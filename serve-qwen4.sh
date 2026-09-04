#!/bin/sh
# Serve Qwen3.8-Flash-Next with its built-in MTP block and, when the mmproj
# file is around, image input.
# Binds 127.0.0.1 by default; ds4-server has no authentication, so only expose
# it (DS4_HOST=0.0.0.0) on a trusted network.
#   DS4_TIER=Q8|MXFP4|Q4K|Q2K|IQ2 ./serve-qwen4.sh   weight tier (default Q8)
#   DS4_GGUF_DIR=/path ./serve-qwen4.sh   where the GGUFs live (default gguf/, then ~/ds4-gguf)
#   DS4_HOST=0.0.0.0   ./serve-qwen4.sh   reachable from the network
#   DS4_CTX=65536      ./serve-qwen4.sh   smaller context (model max 262144)
#   DS4_KV_MB=16384    ./serve-qwen4.sh   smaller disk KV budget
#   DS4_SLOTS=4        ./serve-qwen4.sh   N resident sessions, decoded in turn (default 1;
#                                         >1 turns MTP off)
#   DS4_YARN=4         ./serve-qwen4.sh   static YaRN for prompts past the native
#                                         262144 tokens (context becomes 4x that;
#                                         use 2 for 512k), slightly worse on short text
#   DS4_NO_VISION=1    ./serve-qwen4.sh   text only even if the mmproj exists
#   DS4_MODEL=/path.gguf, DS4_VISION=/path-mmproj.gguf   explicit files
#   DS4_TRACE=/path.log ./serve-qwen4.sh   log requests, prompts and outputs to a file
cd "$(dirname "$0")"
MODEL_NAME="Qwen3.8-Flash-Next-${DS4_TIER:-Q8}.gguf"
VISION_NAME=mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
MODEL="${DS4_MODEL:-}"
VISION="${DS4_VISION:-}"
for d in "${DS4_GGUF_DIR:-gguf}" "$HOME/ds4-gguf"; do
  [ -z "$MODEL" ] && [ -f "$d/$MODEL_NAME" ] && MODEL="$d/$MODEL_NAME"
  [ -z "$VISION" ] && [ -f "$d/$VISION_NAME" ] && VISION="$d/$VISION_NAME"
done
[ -n "$MODEL" ] || { echo "serve-qwen4: $MODEL_NAME not found" >&2; exit 1; }
set --
if [ -z "$DS4_NO_VISION" ] && [ -n "$VISION" ]; then
  set -- "$@" --vision "$VISION"
elif [ -z "$DS4_NO_VISION" ]; then
  echo "serve-qwen4: no vision encoder ($VISION_NAME), serving text only" >&2
fi
[ -n "$DS4_TRACE" ] && set -- "$@" --trace "$DS4_TRACE"
CTX="${DS4_CTX:-262144}"
if [ -n "$DS4_YARN" ]; then
  export DS4_QWEN4_YARN_FACTOR="$DS4_YARN"
  [ -z "$DS4_CTX" ] && CTX=$(awk -v f="$DS4_YARN" 'BEGIN { c = int(f * 262144); if (c > 1048576) c = 1048576; print c }')
fi
# MTP verifies two tokens per step on one session; batched slots run plain decode.
case "${DS4_SLOTS:-1}" in
  ''|*[!0-9]*) echo "serve-qwen4: DS4_SLOTS must be a number" >&2; exit 1 ;;
esac
if [ "${DS4_SLOTS:-1}" -gt 1 ]; then
  set -- "$@" --batched-session "$DS4_SLOTS"
else
  set -- "$@" --mtp --mtp-exact-sampling
fi
exec ./ds4-server \
  -m "$MODEL" \
  --metal --ctx "$CTX" "$@" \
  --kv-disk-dir "$HOME/.ds4/server-kv" --kv-disk-space-mb "${DS4_KV_MB:-262144}" \
  --host "${DS4_HOST:-127.0.0.1}" --port "${DS4_PORT:-8080}"
