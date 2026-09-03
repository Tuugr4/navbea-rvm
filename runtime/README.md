# Hermetic runtime layout

Release CI creates separate `cpu` and `cuda` packages from the pinned worker
requirements. The selected package is installed below this directory and must
contain its Python executable and all wheels before signing. The production
service never invokes pip or downloads a model at runtime.
