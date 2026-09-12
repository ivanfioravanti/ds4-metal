# Qwen3.8 Flash Next

[Back to README](../README.md)

Qwen3.8-Flash-Next uses the `qwen4exp` GGUF architecture and a dedicated
Metal graph with gated delta-net, gated GQA, block-sparse attention,
hyper-connections, per-layer n-gram embeddings (PLE), MoE, and MTP.

## Download and run

```sh
make
./download_model.sh qwen38-q2
./ds4 --ple gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --ctx 8192 --prefill-chunk 1024
```

The download installs a combined main/MTP model (41.73 GiB) and a required
external Q4_1 PLE table (29.80 GiB), and updates `ds4flash.gguf`.
The Q2 model uses IQ2_XXS gate/up experts and Q2_K down projections padded
from 640 to 768 columns. `qwen38-q4k` downloads the larger Q4_K option.
Adjust the PLE path when using `DS4_GGUF_DIR`. Resident PLE pages and runtime
buffers consume additional memory; start with the context and chunk above
on a 64 GB machine.

Use the same model and PLE options with `ds4-agent`, `ds4-server`, or
`ds4-bench`. Add `--mtp` for speculative decoding using the built-in MTP
weights. `--nothink` disables thinking. The server exposes
`qwen3.8-flash-next`, `qwen3.8-flash-next-chat`, and
`qwen3.8-flash-next-reasoner` aliases. Tool calls use the native
`<tool_call><function=...><parameter=...>` format.

Disk KV checkpoints include recurrent state. Rewinding to an earlier position
replays the retained prefix on the next evaluation. The native context is
262144 tokens; `DS4_QWEN4_YARN_FACTOR=2` or `=4` enables static YaRN for
longer contexts, with a possible quality cost on shorter prompts.

PLE is demand-paged when memory is tight and prefetched when installed RAM
covers the resident model, sidecar, and a 16 GiB margin.
`DS4_QWEN4_PLE_PREFETCH_FULL=0` or `=1` overrides that choice.

## Conversion

See [GGUF conversion](../gguf-tools/README.md) for conversion from the HF
checkpoint and calibrated low-bit expert quantization.
`gguf-tools/qwen4_pack_to_qwen4exp.py` migrates old DS4 fast packs to the
active `qwen4exp` schema. Its shared pack modules remain conversion dependencies.

## Vision

Images go through the model's Qwen3-VL vision tower. Download llama.cpp's
mmproj file with `./download_model.sh qwen38-vision` (it comes from
`ggml-org/Qwen3.8-Flash-Next-GGUF`; alternatively run
`convert_hf_to_gguf.py --mmproj --outtype f16` on the checkpoint) with
`--vision` and send `image_url` parts as usual. Each image is resized to
multiples of 32 pixels within 64 to 1024 tokens (`DS4_QWEN4_IMAGE_MAX_TOKENS`
raises the cap), encoded on the GPU, and takes the model's 3D rope positions;
live KV reuse keys on the image fingerprints. `make test-qwen4-vision`
compares the tower with the Hugging Face implementation using the same GGUF
weights, dequantized to float32. It separately reports GGUF-versus-original
checkpoint quality and Metal-versus-original agreement. Both comparisons use
the unchanged minimum per-token cosine threshold of 0.99; implementation
parity gates the default exit status. Add `--require-quality` to also fail on
GGUF-versus-original quality loss. A passing implementation check alone does
not mean the quantized encoder matches the original checkpoint.

For a small suite including OCR, diagrams, a photograph, and resizing:

```sh
make tests/test_qwen4_vision
uv run --with numpy --with torch --with torchvision --with pillow \
  --with safetensors --with transformers --with gguf tests/qwen4_vision_ref.py \
  --snapshot /path/to/HF-checkpoint \
  --mmproj gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --image tests/vision-fixtures/qwen38/orbit.png \
  --image tests/vision-fixtures/qwen38/maple.png \
  --image tests/vision-fixtures/glm53/diagram.png \
  --image tests/vision-fixtures/glm53/text.png \
  --image tests/vision-fixtures/glm53/earth.jpg \
  --image tests/vision-fixtures/glm53/screenshot.png \
  --json-report /tmp/qwen38-vision-parity.json
```

Transformers must include `qwen4_exp`; the reference runs on CPU. Repeat
`--image` to add cases. The original single-image `--out` embedding dump is
still supported. Metric checks without model weights run with
`uv run --with numpy python -m unittest discover -s tests -p test_qwen4_vision_ref.py`.
JPEG decoder agreement with Pillow, including progressive scans, subsampling,
and image edges, is checked separately with
`uv run --with numpy --with pillow python -m unittest discover -s tests -p test_jpeg_decode.py`.

To check CLI `/read` image turns and text follow-ups with ordinary and MTP
decode, use two different PNG or JPEG images with the model-backed regression:

```sh
make ds4
uv run tests/test_qwen4_cli_vision.py --model /path/to/main-with-mtp.gguf \
  --ple /path/to/ple.gguf --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --image /path/to/first.png --image /path/to/second.png
```

The test checks that all four turns complete in each mode, including errors
that the interactive CLI can report without a nonzero process exit. It saves
the responses and diagnostics for inspection; it does not grade image content.

Metal only for now. The Metal graph accepts Q8_0, Q4_0, F16, BF16 and F32
dense weights, Q8_0/MXFP4/Q4_0/Q4_K/Q2_K/IQ2_XXS experts, F16/F32/Q8_0
hyper-connection mixers and a Q8_0/Q4_0/MXFP4/F16/F32 n-gram table.
Multi-node tensor parallelism is not implemented yet.


## Validation

```sh
make test-qwen4-kernels test-qwen4-q2 test-qwen4-prefill-reuse test-q8-prefill-variants
make test-frontends
make -B -C gguf-tools quants-shared
python3 -m unittest discover -s gguf-tools/tests -p test_qwen4_pack.py
```

The kernel tests exercise the active Metal API. Vision and end-to-end model
checks additionally require the checkpoints described above.

[Checkpoint-fix benchmark charts and measurements](../speed-bench/qwen38-checkpoints/README.md)
compare prefill, ordinary decode, and MTP decode against the preceding PR head.

Qwen prefill checkpoints include matching logits at each completed chunk,
including cancellation frontiers. To test save/restore and continuation:

```sh
DS4_TEST_MODEL=/path/to/main-with-mtp.gguf DS4_TEST_PLE=/path/to/ple.gguf \
  DS4_TEST_GLM_MTP=1 DS4_TEST_MTP_EXACT=1 \
  ./ds4_test --qwen4-prefill-checkpoints --session-snapshot --session-rewind-resample
python3 tests/test_qwen4_checkpoint_replay.py \
  --model /path/to/main-with-mtp.gguf --ple /path/to/ple.gguf
```

The server regression checks continued disk saves, cache-budget eviction,
and text-answer history with and without tool schemas. It covers clients
that omit reasoning and clients that echo `reasoning_content`, including
server restart after a continuation. For tools-enabled histories, the first
answer keeps the exact disk key; after a client continues without reasoning,
the next checkpoint uses the visible history key for restart reuse.

Qwen directional steering and activation capture support all 48 trunk layers;
see [directional steering](../dir-steering/README.md). Model-backed regressions:

```sh
python3 tests/test_qwen4_steering.py --model /path/to/main-with-mtp.gguf \
  --ple /path/to/ple.gguf
DS4_TEST_PLE=/path/to/ple.gguf \
  ./ds4_agent_test --full-context-save /path/to/main-with-mtp.gguf
make test-web-recovery
```

The agent can save an already-full text transcript without a KV payload.
Restart with a larger `--ctx`, then `/switch` to the saved session and
`/compact` to recover room. Sessions containing images still use the existing
save restriction. Automatic compaction reserves summary space and continues
interrupted generation through the existing agent compaction path.
