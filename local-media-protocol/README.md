# Local media protocol

This is the permissively licensed wire-contract implementation used by Navbea
RVM to communicate with a compatible Camera Service. Production transport is a
Windows named pipe or Linux Unix-domain socket; it never opens a TCP listener.

Frames use a versioned 64-byte `NVF1` header followed by a bounded binary
payload. RVM output masks retain the source camera frame sequence number.
