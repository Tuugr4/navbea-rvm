function windowBoundsForWorkArea(area) {
  const width = Math.max(1, Math.min(1366, area.width));
  const height = Math.max(1, Math.min(900, area.height));
  return {
    width,
    height,
    minWidth: Math.min(640, width),
    minHeight: Math.min(480, height),
    x: area.x + Math.floor((area.width - width) / 2),
    y: area.y + Math.floor((area.height - height) / 2),
  };
}
module.exports = { windowBoundsForWorkArea };
