# Almost ARCADIA

Almost ARCADIA is a single-user Django control and presentation layer for a trusted research network. The client runs Priority Map, retains media and outputs locally, and calls LLM and SAM3 inference ports directly. Compute hosts expose a small instruction server that starts, stops, lists, and reports logs for processes it owns; it never proxies inference.

## Network architecture

- **Django/client port** (`8000` by default): configuration, uploads, run status, previews, and artifacts.
- **Instruction port** (`9000` by default): lifecycle control on each compute host.
- **Inference ports** (`8081`, `8082`, `8090` in the defaults): direct LLM, Visual LLM, and SAM3 data-plane traffic.

This prototype has no authentication or TLS. Use it only on a trusted local network or VPN and limit firewall access accordingly.

## Installation

Python 3.11 or newer is required.

### macOS Apple Silicon / Metal

```bash
bash scripts/install_macos_metal.sh
```

### Windows / CUDA

```powershell
.\scripts\install_windows_cuda.ps1
```

The scripts create `.venv`, install the Python project, check out the pinned llama.cpp commit from `scripts/llama_cpp_commit.txt`, build native `llama-server`, and run checks. A separately installed binary can be selected with:

```bash
export ARCADIA_LLAMA_SERVER=/absolute/path/to/llama-server
```

Hugging Face downloads are stored on each compute node under:

```text
huggingface/
├── models/  # GGUF models and uploaded sam3.pt checkpoints
└── mmproj/  # vision projector GGUF files
```

Both cache folders are created automatically. `ARCADIA_HUGGINGFACE_DIR` overrides the complete cache root. The deprecated `ARCADIA_MODELS_DIR` is interpreted identically when the new variable is absent; it emits a warning. Existing `workspace/huggingface` files are not moved or deleted automatically—copy them into the new root or upload them again. `ARCADIA_STATE_DIR` controls browser-upload staging and service logs; its default is OS application state, not the repository.

SAM3 uses the Ultralytics-format `sam3.pt` checkpoint. Model Settings **Browse** uploads a `.pt` file to the selected local or remote compute node under `huggingface/models`; save the returned host-local path with the SAM3 settings. The Host page provides the same local-host upload and save flow for its default checkpoint. A configured checkpoint must exist, end in `.pt`, and be under that host's `huggingface/models` directory.

The canonical SAM3 request is:

```json
{"image_base64":"<base64-encoded JPEG/PNG/WebP>","text":"person","confidence":0.25}
```

`confidence` is inclusive from `0.0` through `1.0`; `text` is required and non-empty. The response contains `detections`, each with `label`, `confidence`, optional `box`, and optional `mask_png_base64`, plus `overlay_png_base64`. Legacy `image`, `prompt`, and `prompts` inputs remain accepted by the inference server only while callers migrate.

## Start the application

```bash
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

The real Django `runserver` child automatically starts one owned instruction server using the saved Host settings. The initial address is `127.0.0.1:9000`. Open **Host** to bind it to a LAN or VPN IPv4 address assigned to the computer.

## Model settings

The LLM interface deliberately exposes only settings that Almost ARCADIA must understand:

- Compute node
- Inference port
- Inference IP
- Hugging Face model source
- Vision toggle
- Hugging Face projector source
- Context size
- Max output tokens
- Temperature
- Additional native `llama-server` arguments

The inference IP defaults to the selected compute node IP and remains editable. This allows an instruction address and an inference address to differ, but the selected IP must actually exist and be reachable on the compute host.

### Hugging Face sources

The model and projector fields accept either a repository:

```text
unsloth/Qwen3.5-2B-GGUF
```

or an exact GGUF link:

```text
https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/blob/main/Qwen3.5-2B-IQ4_XS.gguf
```

A repository is selected automatically only when it has exactly one usable GGUF model (or one split-model family). For ambiguous repositories, paste the exact file link. Split GGUFs are supported when the exact link points to shard `00001`.

Vision requires a projector source. A separate Visual LLM always has vision enabled. In **Same as Logical LLM** mode, Logical LLM must have vision enabled and supplies both roles from one process.

### Additional llama-server arguments

Separate arguments with spaces, commas, or line breaks:

```text
--flash-attn on,
--batch-size 2048
--ubatch-size 512, --cache-type-k q8_0
--cache-type-v q8_0
--mlock
```

Commas inside a value must be quoted:

```text
--tensor-split "1,1"
--chat-template-kwargs '{"preserve_thinking": false}'
```

The resulting token list is passed directly to `subprocess.Popen(..., shell=False)`. Almost ARCADIA rejects flags that override app-owned model, projector, host, port, context, temperature, or output-token settings, plus protocol-changing or unsafe server modes. The app does not set `--alias`; an optional native `--alias` may be supplied in this field. Other native flags are passed through and validated by the pinned `llama-server` itself. Startup errors appear in service logs.

The app-owned command shape is:

```text
llama-server
  --host <inference-ip>
  --port <inference-port>
  --model <downloaded-gguf>
  --ctx-size <context>
  [--mmproj <downloaded-projector>]
  <additional arguments...>
```

Temperature and max output tokens are request-time defaults sent in `/v1/chat/completions`; they are not startup flags. Before the first request, the client reads `/v1/models` and uses the model ID actually exposed by `llama-server`, preserving its native or user-supplied alias.

## Remote compute host

On every compute machine:

1. Install this same code and native dependencies.
2. Run Django locally.
3. Open **Host** and save the machine's reachable LAN/VPN IPv4 address and instruction port.
4. Allow the instruction port and selected inference ports through the trusted-network firewall.
5. Add the machine under **Model settings → Compute nodes** on the client.

Remote model files are downloaded and cached on the machine that runs the service. The instruction server accepts typed service specifications, not arbitrary commands or executable paths. Remote starts may legitimately take several minutes while a model downloads or loads; the control client waits up to 11 minutes, reuses an identical running service, and returns the remote startup error plus the tail of the model log instead of a generic HTTP 500.

## Priority Map

The `pipeline` optional dependency pins Priority Map at:

```text
ea6d1064175b20c1e90dd3f1ffb0b4173f68e03d
```

Almost ARCADIA temporarily substitutes direct-client implementations for `SceneUnderstanding`, `Segment`, and `GraphAgent` while constructing the pinned runner, then restores the original symbols. The Graph Agent adapter implements the pinned asynchronous no-argument lifecycle and applies returned score deltas through the graph builder.

Install the full pipeline and SAM dependencies with:

```bash
pip install -e ".[all]"
```

Configure a real SAM3 checkpoint in the SAM3 card. The card now includes **Test SAM3**: upload a JPEG, PNG, or WebP image, enter a concept such as `car` or `red backpack`, and run the saved configuration. The test starts or reuses the configured local/remote SAM3 service, sends the text concept through the same SAM client used by Priority Map, and returns an overlaid PNG with masks, contours, labels, confidence values, and available boxes. No fake masks or synthetic successful results are used in production paths.

### Manual SAM3 integration procedure

1. On the compute host, activate `almost-arcadia`, run migrations if needed, then start the Host Django app with `python manage.py runserver <host-ip>:8000`.
2. Open **Host**, use **Browse or upload** to stream `sam3.pt` to that host, then click **Save checkpoint**. Confirm the status is **Ready**.
3. Open **Model Settings**, select that host, save the SAM3 configuration using the uploaded checkpoint, then run **Test SAM3**. This starts the service if necessary.
4. Confirm `GET http://<host-ip>:8090/health` returns `{"status":"ready","service_type":"sam3"}`.
5. Upload a JPEG, PNG, or WebP test image, enter a non-empty search term, and confirm the segmented PNG is displayed and downloadable.
6. From a remote client, configure the remote node, upload `sam3.pt` through Model Settings, save SAM3 settings, and repeat the image test.
7. On the selected host, confirm the checkpoint is in `huggingface/models/` and that neither project-level `workspace/` nor `uploads/` was created.

## Run an analysis

1. Save Logical LLM, Visual mode/settings, and SAM3 settings.
2. Open **Analysis** and save pipeline options.
3. Provide a local path or stage an upload.
4. Start the analysis.
5. Open **Results** for the live MJPEG preview and output artifacts.

Only one analysis runs at a time. Outputs are retained under:

```text
outputs/<run-id>/
├── effective_settings.json
├── analysis.log
├── observations.csv
└── Priority Map images/videos/graphs
```

## Checks

```bash
python -m pytest -q
ruff check .
python manage.py check
mypy core web project manage.py
```

Real model, SAM3 checkpoint, and complete moving-sequence smoke tests require local hardware and weights; unit tests use bounded fakes around external processes and HTTP calls.

## Important limitations

- Trusted single-user LAN/VPN prototype; no auth, TLS, or multi-user isolation.
- One active analysis and one serialized SAM predictor per host.
- No persistent recovery of model subprocesses after the owning process exits.
- Raw native argument values are intentionally not mirrored in Django validation.
- Draft-model repositories supplied through native speculative-decoding flags are handled by llama.cpp, not downloaded through Almost ARCADIA's main/projector resolver.
- SAM masks currently cross the network as JSON arrays because Priority Map needs raw masks. This is simple and testable but inefficient for high-resolution images; a future protocol should use compressed mask PNGs or run-length encoding.
- A remote inference IP can be syntactically validated by the client, but only the compute host can determine whether it is assigned and bindable.

See [`REVIEW_NOTES.md`](REVIEW_NOTES.md) for design tradeoffs and the remaining heavyweight validation work.
