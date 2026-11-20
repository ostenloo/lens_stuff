# Source before any uv/python command on a box whose home partition is small:
# the uv cache, HF weights and temp files all land on the big volume instead.
export UV_CACHE_DIR=/mnt/data/lens_stuff/uv-cache
export HF_HOME=/mnt/data/huggingface
export TMPDIR=/mnt/data/lens_stuff/tmp
mkdir -p "$TMPDIR"
