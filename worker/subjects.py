"""GPL-3.0: ephemeral, local person-instance selection for RVM.

No identity recognition or image persistence. Ambiguous matches fail closed.
Coordinates for selection use the centered 16:9 live view; instance masks keep
their source-image coordinates so native photo masking is never a bbox crop.
"""
import io
import time
import uuid

import numpy as np
from PIL import Image, ImageFilter


class SubjectBlocked(ValueError):
    pass


def overlap(a, b):
    lo = np.maximum(a[:2], b[:2]); hi = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(0, hi - lo))
    return float(intersection / max(1e-8, np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection))


class PersonDetector:
    def __init__(self, model, threads=2):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(2, threads)); options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def detect(self, image, native=False):
        edge = 640 if native else 320
        w, h = image.size; ratio = min(edge / w, edge / h)
        nw, nh = round(w * ratio), round(h * ratio); px, py = (edge - nw) // 2, (edge - nh) // 2
        padded = Image.new("RGB", (edge, edge), (114, 114, 114))
        padded.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (px, py))
        tensor = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1)[None] / 255
        prediction, proto = self.session.run(None, {self.input: tensor})
        prediction = prediction[0]; proto = proto[0]
        scores = prediction[:, 4] * prediction[:, 5]
        rows = prediction[(scores >= .3) & (prediction[:, 5:85].argmax(axis=1) == 0)]
        if not len(rows): return []
        scores = rows[:, 4] * rows[:, 5]
        boxes = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2, rows[:, :2] + rows[:, 2:4] / 2), axis=1)
        order = scores.argsort()[::-1][:100]; keep = []
        while len(order) and len(keep) < 24:
            chosen = int(order[0]); keep.append(chosen)
            order = np.asarray([j for j in order[1:] if overlap(boxes[chosen], boxes[j]) < .45], dtype=np.int64)
        # Evaluate only person coefficients. Keep low-resolution instance support
        # separate from RVM alpha, which retains native hair/edge precision.
        logits = rows[keep, 85:] @ proto.reshape(32, -1)
        masks = (1 / (1 + np.exp(-np.clip(logits, -30, 30)))).reshape(-1, proto.shape[1], proto.shape[2])
        sw, sh = 256, max(1, round(h * 256 / w))
        small_rgb = np.asarray(image.resize((sw, sh), Image.Resampling.BILINEAR))
        view_w, view_h = min(w, h * 16 / 9), min(h, w * 9 / 16)
        ox, oy = (w - view_w) / 2, (h - view_h) / 2
        people = []
        for index, chosen in enumerate(keep):
            box = boxes[chosen]
            ys, xs = np.ogrid[:proto.shape[1], :proto.shape[2]]
            clipped = masks[index] * ((xs >= box[0] / edge * proto.shape[2]) & (xs < box[2] / edge * proto.shape[2]) & (ys >= box[1] / edge * proto.shape[1]) & (ys < box[3] / edge * proto.shape[1]))
            mask_image = Image.fromarray(np.uint8(clipped * 255)).resize((edge, edge), Image.Resampling.BILINEAR)
            support = np.asarray(mask_image.crop((px, py, px + nw, py + nh)).resize((sw, sh), Image.Resampling.BILINEAR)) > 100
            if support.sum() < 12: continue
            source_box = (box - [px, py, px, py]) / ratio
            view_box = (source_box - [ox, oy, ox, oy]) / [view_w, view_h, view_w, view_h]
            pixels = small_rgb[support].astype(np.int32) // 32
            histogram = np.bincount(pixels[:, 0] * 64 + pixels[:, 1] * 8 + pixels[:, 2], minlength=512).astype(np.float32)
            histogram /= max(1, histogram.sum())
            people.append({"box": view_box, "mask": support, "hist": histogram, "area": float(support.mean()), "score": float(scores[chosen])})
        return people


class SubjectSelector:
    def __init__(self, detector, clock=time.monotonic):
        self.detector = detector; self.clock = clock
        self.reset()

    def reset(self):
        self.policy = None; self.tracks = {}; self.next_id = 1; self.selected = []
        self.lock_id = None; self.stable_since = None; self.last_at = 0; self.last_source = None
        self.lost = False; self.lock_density = {}
        self.state = {"ready": False, "reason": "select", "count": 0, "locked": False}

    def configure(self, policy):
        self.reset(); self.policy = policy

    def lock(self):
        if not self.state["ready"] or self.clock() - self.last_at > .5:
            raise SubjectBlocked("SUBJECT_NOT_READY")
        self.lock_id = self.lock_id or str(uuid.uuid4())
        self.lock_density = {key: self.tracks[key]["area"] / max(1e-6, np.prod(self.tracks[key]["box"][2:] - self.tracks[key]["box"][:2])) for key in self.selected}
        self.state = {**self.state, "locked": True, "lockId": self.lock_id}
        return self.state

    def unlock(self):
        policy = self.policy; self.configure(policy)
        return self.state

    def eligible(self, person):
        x1, _, x2, y2 = person["box"]; zone = self.policy["zone"]
        return zone["x"] <= (x1 + x2) / 2 <= zone["x"] + zone["width"] and zone["y"] <= y2 <= zone["y"] + zone["height"]

    @staticmethod
    def cost(previous, current):
        a, b = previous["box"], current["box"]
        center = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2)
        similarity = np.minimum(previous["hist"], current["hist"]).sum()
        if center > .22 or similarity < .32: return 10.
        return float(.5 * (1 - overlap(a, b)) + .5 * (1 - similarity))

    def match(self, people, native):
        matches = {}; used = set(); ambiguous = False
        # Match the locked cohort first; never let an outsider steal its slot.
        ids = self.selected if native else sorted(self.tracks, key=lambda key: key not in self.selected)
        for identity in ids:
            previous = self.tracks.get(identity)
            if previous is None: continue
            ranked = sorted((self.cost(previous, person), i) for i, person in enumerate(people) if i not in used)
            if not ranked or ranked[0][0] >= .67: continue
            if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < .08:
                if identity in self.selected: ambiguous = True
                continue
            _, index = ranked[0]; used.add(index); matches[identity] = people[index]
        if not native:
            for i, person in enumerate(people):
                if i not in used:
                    matches[self.next_id] = person; self.next_id += 1
        return matches, ambiguous

    def evaluate(self, image, native=False, source=None):
        if self.policy["mode"] == "all":
            now = self.clock()
            # Retain the frame/capture lease without selecting or filtering people.
            if native and (not self.lock_id or now - self.last_at > 15):
                raise SubjectBlocked("SUBJECT_LOCK_EXPIRED")
            if not native:
                self.last_at = now; self.last_source = source
                self.state = {"ready": True, "reason": "ready", "count": 0, "locked": bool(self.lock_id), "lockId": self.lock_id, "mode": "all", "selectionApplied": False}
            return None, {**self.state, "verifiedNative": native}
        people = self.detector.detect(image, native)
        now = self.clock(); matches, ambiguous = self.match(people, native)
        if native and (not self.lock_id or self.lost or now - self.last_at > 15):
            raise SubjectBlocked("SUBJECT_LOCK_EXPIRED")
        if not self.lock_id and not native:
            selected = [identity for identity, person in matches.items() if self.eligible(person)]
            if selected != self.selected: self.stable_since = None
            self.selected = selected
        targets = [matches[key] for key in self.selected if key in matches]
        outsiders = [person for person in people if all(person is not target for target in targets)]
        reason = None
        if not targets: reason = "select"
        elif len(targets) != len(self.selected): reason = "lost"
        elif len(targets) > self.policy["maxPeople"]: reason = "too_many"
        elif ambiguous: reason = "ambiguous"
        elif self.policy["mode"] == "area" and any(not self.eligible(p) for p in targets): reason = "outside_area"
        elif any(p["box"][0] < .005 or p["box"][1] < .005 or p["box"][2] > .995 or p["box"][3] > .995 for p in targets): reason = "clipped"
        for identity in self.selected:
            current = matches.get(identity); previous = self.tracks.get(identity)
            if self.lock_id and current and previous and current["area"] < previous["area"] * .65: reason = "occluded"
            if self.lock_id and current:
                density = current["area"] / max(1e-6, np.prod(current["box"][2:] - current["box"][:2]))
                if density < self.lock_density.get(identity, density) * .65: reason = "occluded"
        for target in targets:
            grown = np.asarray(Image.fromarray(target["mask"].astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0
            if any(np.any(grown & outsider["mask"]) for outsider in outsiders): reason = "occluded"
        if not native:
            # Missing/unsafe observations cannot advance the baseline or reassign
            # locked IDs. After a long gap require explicit group reselection.
            if self.lock_id and (reason in {"select", "lost", "ambiguous"} or (self.last_at and now - self.last_at > 2)): self.lost = True
            if self.lost: reason = "lost"
            if reason or (self.last_at and now - self.last_at > .5): self.stable_since = None
            if reason is None and self.stable_since is None: self.stable_since = now
            ready = reason is None and (now - self.stable_since) * 1000 >= self.policy["stableMs"]
            if not self.lock_id:
                self.tracks = matches
            elif reason is None:
                self.tracks = {key: matches[key] for key in self.selected}
            self.last_at = now; self.last_source = source
            self.state = {"ready": ready, "reason": reason or ("ready" if ready else "stabilizing"), "count": len(targets), "locked": bool(self.lock_id), "lockId": self.lock_id, "mode": self.policy["mode"]}
        else:
            if reason: raise SubjectBlocked("SUBJECT_" + reason.upper())
        if not targets or self.lost or ambiguous:
            support = Image.new("L", (256, max(1, round(image.height * 256 / image.width))))
        else:
            union = np.logical_or.reduce([p["mask"] for p in targets])
            support = Image.fromarray(union.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(.6))
            if outsiders:
                excluded = np.logical_or.reduce([p["mask"] for p in outsiders])
                values = np.array(support); values[excluded] = 0; support = Image.fromarray(values)
        metadata = {**self.state, "verifiedNative": native, "count": len(targets)}
        return support, metadata

    @staticmethod
    def filter_alpha(mask, width, height, support):
        alpha = np.frombuffer(mask, dtype=np.uint8).reshape(height, width)
        gate = np.asarray(support.resize((width, height), Image.Resampling.BILINEAR))
        # uint16 keeps multiplication exact without a full float native tensor.
        result = np.empty_like(alpha)
        for y in range(0, height, 128):
            result[y:y+128] = ((alpha[y:y+128].astype(np.uint16) * gate[y:y+128] + 127) // 255).astype(np.uint8)
        return result.tobytes()
