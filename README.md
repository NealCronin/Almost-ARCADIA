# Almost ARCADIA

Almost ARCADIA is a Django control and presentation layer for a trusted-user research pipeline. It starts owned LLM and SAM3 HTTP services, sends inference directly to those services, runs Priority Map on the client machine, and retains logs and outputs locally.

This is a single-user prototype for a trusted LAN or VPN. It deliberately has no authentication, authorization, TLS management, database service registry, scheduler, Celery, container orchestration, hardware detection, or persistent host-side recovery.

## Architecture

There are three network-facing port types:

* **Instruction port**: FastAPI control API on a compute host (`9000` in the example). It accepts typed service specifications and can start, replace, stop, list, and return bounded logs for processes owned by that instruction-server process. It never proxies inference.
* **Inference ports**: one LLM port (for example `8081`) and one SAM3 port (for example `8090`). Django and the pipeline call these ports directly at `/v1/chat/completions` and `/v1/predict`.
* **Django port**: the presentation plane, normally `8000`, running on the client machine.

The Priority Map pipeline runs on the client. Image folders remain in place. Videos are decoded into the analysis output directory on the client before the external pipeline reads the frames. Only encoded frames needed for LLM or SAM requests cross the network. Pipeline outputs, effective settings, analysis logs, and service logs are local files.

A service is identified by `(host, inference port)`. Reconfiguring an occupied port replaces only the process previously launched and still owned by this application. No process is killed by scanning operating-system port ownership.

Replacement is stop-then-start. If a replacement fails its new child is terminated, its log handle is closed, and the port is removed from the owned-service registry; the prior service is **not** restored. The UI and instruction API report that the port is left without a running service.

## Installation

Python 3.11 or newer is required. Create a virtual environment first.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,sam,pipeline]"
```

Linux/macOS:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,sam,pipeline]"
```

The `pipeline` extra installs Priority Map at commit `ea6d1064175b20c1e90dd3f1ffb0b4173f68e03d`, whose `PriorityMapRunner` constructor imports and instantiates `priority_map.runner.SceneUnderstanding` and `Segment`. The `sam` extra installs the Ultralytics package used by the current Priority Map SAM3 implementation. If the external SAM3 source has a newer installation procedure, install that source in the same environment and keep the checkpoint path in `config.json` accurate.

### llama-cpp-python and Hugging Face models

Almost ARCADIA pins `llama-cpp-python[server]==0.3.34`. The LLM builder launches its verified server flags directly, including context, GPU/CPU, batch, cache, memory, tensor split, RoPE/YaRN, model alias, and MMProj controls. It does not expose unsupported `--n_parallel`, and it does not use `--hf_model_repo_id`: Almost ARCADIA resolves GGUF files itself.

Choose a Hugging Face `owner/repository`; normal Hugging Face login/environment credentials are honored for private or gated repositories. Each compute machine owns its cache at `workspace/models/huggingface` and `workspace/models/mmproj`, or beneath `ARCADIA_MODELS_DIR` when set. Caches are never sent to or deleted by another machine. An uncached offline or unauthenticated repository cannot start, but remains saveable.

Model discovery considers only unsplit `.gguf` files. MMProj/projector names are excluded from the main model selection; split shards are rejected. A single usable file is selected automatically. Multiple candidates require **Advanced settings → Model file pattern**; vision similarly requires a unique projector pattern when needed. Request-time generation controls—temperature, top-k, min-p, and top-p—are sent on every Priority Map LLM request, not as server startup flags.

### SAM3 checkpoint

Install the real SAM3-capable package used by Priority Map, then manually place a compatible checkpoint outside source control, for example:

```text
checkpoints/sam3.pt
```

Set `services.sam3.settings.checkpoint` to that path. A missing or unloadable checkpoint is a startup error. Runtime SAM3 responses are never mocked or synthesized.

## Configuration

The Host portal configures this computer's instruction listener. It starts automatically with Django `runserver` on `127.0.0.1:9000` unless `config.json` contains another `host_listener` value. The IP must be assigned to a local interface: use `127.0.0.1` for local-only control, a LAN IP for trusted local-network clients, or a VPN address such as Tailscale for VPN clients.

Saving Host settings serializes replacement and persistence: it stops only the Django-owned instruction-server child, starts a replacement bound directly to the saved IP and port, and waits for `/health`. The configuration is saved only after the replacement is healthy; a persistence failure restores the prior listener when possible and leaves the prior saved configuration unchanged. If automatic startup fails, Django remains available so Host can repair the listener configuration.

Restarting the instruction server stops any LLM or SAM processes owned by that instruction-server process. The next Priority Map run can automatically reprovision configured services. It does not intentionally stop unrelated Django-owned local services.

Remote clients must update their saved instruction-host IP and port when this listener changes. Allow the instruction port and direct inference ports through the relevant host firewall for trusted LAN/VPN clients.


### Priority Map Models page

**Client → Priority Map → Model settings** retains local and remote compute-node CRUD. Remote node fields are **Instruction-server IP** and **Instruction port**; those control the FastAPI listener. LLM **Inference bind host** is separate: local runs accept only an IPv4 address assigned to this computer and default to `127.0.0.1`; remote runs always bind exactly to the selected node address and do not accept a browser-supplied override.

The LLM card has quick controls for compute node, inference port, repository, context size, temperature, top-k, min-p, and top-p. Advanced settings contain model pattern/alias/chat format, optional MMProj vision, local networking, performance, memory/cache, context extension, timeout, and safe additional server arguments. Repository inspection lists bounded filename suggestions without downloading files or changing saved settings.

Direct inference ports require trusted LAN/VPN firewall access. The remote instruction server must run this same code with `--host` and `--public-host` equal to its configured node IP; it rejects commands, cache paths, unknown LLM launch fields, and a bind-host mismatch.

```json
{
  "host_listener": {
    "host": "127.0.0.1",
    "port": 9000
  },
  "nodes": {
    "local": {"mode": "local", "host": "127.0.0.1"},
    "example_remote": {"mode": "remote", "host": "192.168.1.20", "instruction_port": 9000}
  },
  "services": {
    "llm": {
      "node": "local",
      "service_type": "llm",
      "port": 8081,
      "settings": {
        "model_path": "models/model.gguf",
        "bind_host": "0.0.0.0",
        "startup_timeout": 600,
        "n_ctx": 32768,
        "n_gpu_layers": -1,
        "extra_args": ["--n_batch", "2048", "--n_ubatch", "512", "--flash_attn", "true"]
      }
    },
    "sam3": {
      "node": "local",
      "service_type": "sam3",
      "port": 8090,
      "settings": {"checkpoint": "checkpoints/sam3.pt", "bind_host": "0.0.0.0", "confidence": 0.25}
    }
  },
  "pipeline": {"sam_step": 5, "run_at_source_fps": false, "sam_resize": null},
  "output_root": "outputs"
}
```

Existing settings dictionaries remain compatible. The UI saves configuration atomically through a same-directory temporary file and replacement. `ARCADIA_CONFIG` can point Django at another JSON file. No live process state is stored in JSON or SQLite.

Useful manual LLM settings include `bind_host`, `startup_timeout`, `n_ctx`, `n_gpu_layers`, `chat_format`, `model_alias`, and `extra_args`; exact Hugging Face sources use `hf_repo`, `hf_file`, and optional `hf_cache_dir`. The remote instruction API rejects executable, server-module, shell, and arbitrary command fields. Unit tests alone use the command escape hatch.

## Automatic host instruction server

When Django runs through `python manage.py runserver`, the real runserver child starts one Django-owned instruction server using the saved Host settings. Management commands, migrations, checks, imports, pytest, and the autoreloader parent do not start it. `--noreload` starts the same one listener directly.

The listener command is equivalent to:

```powershell
python -m core.services.instruction_server --host <saved-ip> --public-host <saved-ip> --port <saved-port> --log-dir logs/instruction
```

The listener binds directly to the saved IP; it never silently falls back to `0.0.0.0`. Visit **Host** to change the IP or port. The page shows the current listener state, address, uptime, and replacement failures. Other machines must run their own Django Host portal before they can expose their own listener; this application does not remotely bootstrap them.

## Start Django

From the client environment:

```powershell
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

Open <http://127.0.0.1:8000/>. Use **Host** to expose this computer, **Services** to configure LLM/SAM3 service settings, and **Analysis** to set pipeline options and run one analysis. State-changing actions use POST and successful forms redirect.

## Start and test services

The UI starts the configured service and waits for its readiness endpoint. A successful LLM start means `/v1/models` responded successfully. A successful SAM3 start means `/health` responded successfully. A child that exits or times out during loading is terminated and reported.

Direct LLM request (data plane, not the instruction port):

```powershell
$body = @{model="local-model"; messages=@(@{role="user"; content=@(@{type="text"; text="Describe this research prototype in one sentence."})})} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8081/v1/chat/completions -ContentType application/json -Body $body
```

Direct SAM request requires a base64-encoded image. This local helper sends one request directly to the SAM service:

```powershell
$image = [Convert]::ToBase64String([IO.File]::ReadAllBytes("sample.jpg"))
$body = @{image=$image; prompts=@("car"); confidence=0.25} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8090/v1/predict -ContentType application/json -Body $body
```

The response contains `masks`, `labels`, `confidences`, and `bounding_boxes`. SAM inference is serialized by one lock around one loaded predictor.

## Run an analysis

1. Put an image folder, image file, or video on the client machine.
2. Confirm both configured services are reachable.
3. Open **Analysis**, set the task, `sam_step`, confidence, resize, and output options, then save.
4. Enter the local input path and start one analysis.
5. Follow progress on the analysis page or `/analysis/status/`.
6. Open **Results** after completion.

The coordinator permits one active analysis. It saves `effective_settings.json` and `analysis.log` before work begins. Output directories include UTC microseconds and are reserved atomically so rapid sequential runs do not share them. Incremental Priority Map output is preserved after failures. When an `LLMClient` or `SAMClient` identifies its failing service, only that service is reprovisioned and the pipeline is retried once. Unknown service failures conservatively restart both. A second failure ends the analysis.

Outputs are under:

```text
outputs/<UTC timestamp>/
├── effective_settings.json
├── analysis.log
├── input_frames/        # video/single-image input preparation when needed
└── <Priority Map outputs>
```

Service logs are under `logs/` on the host running the service. Remote log tails are available from the instruction API and the UI's service log links.

## Tests and checks

```powershell
python -m pytest -q
ruff format --check .
ruff check .
mypy core web project manage.py
python manage.py check
```

Unit coverage is boundary-focused: subprocess lifecycle, direct HTTP clients, endpoint serialization, Priority Map class substitution, and optical-flow propagation are tested without model weights. The latest validation run reported `60 passed, 1 warning`; the warning is Starlette's external `TestClient` deprecation notice for `httpx2`. `ruff format --check .`, `ruff check .`, `mypy core web project manage.py`, and `python manage.py check` all passed.

### Observed local validation environment

Validation ran in the isolated `almost_arcadia_gpt` conda environment with Python `3.12.13`, Django `5.2.16`, FastAPI `0.139.0`, Uvicorn `0.51.0`, Requests `2.34.2`, NumPy `2.5.1`, OpenCV `4.13.0.92`, llama-cpp-python `0.3.34`, huggingface-hub `0.36.2`, Ultralytics `8.4.96`, Priority Map `0.1.0` at the pinned commit, pytest `8.4.2`, Ruff `0.15.21`, and mypy `1.20.2`. The validation commands set `PYTHONNOUSERSITE=1` to exclude user-site packages.

### Observed Django and instruction-server smoke tests

These commands were run locally:

```powershell
python manage.py migrate
python manage.py runserver 127.0.0.1:8000 --noreload
```

With the automatic listener, `runserver --noreload` starts one listener from `host_listener`. A current smoke run confirmed `GET /health` returned `200 {"status":"ok","service":"instruction"}`, the Host page displayed its running status, a save from port `9000` to `9010` stopped the old listener and made the replacement healthy, and an unassigned IP was rejected while the `9010` listener remained healthy. Owned-child shutdown and failed-replacement rollback are unit-tested; this smoke run did not launch a real model child.

### Heavyweight runtime status

The conda environment includes `llama_cpp`, `huggingface_hub`, `ultralytics`, and Priority Map. `SAM3SemanticPredictor` and the Priority Map runner symbols imported successfully.

Two real LLM lifecycle smoke tests passed with the public, non-gated `afrideva/Tinystories-gpt-0.1-3m-GGUF` file `tinystories-gpt-0.1-3m.Q2_K.gguf`:

1. **Local path:** `models/tinystories/tinystories-gpt-0.1-3m.Q2_K.gguf` started through `ServiceController` on port `8081`; `/v1/models` returned `200` and one direct `/v1/chat/completions` request returned `200`; `controller.stop()` then reported `running_after_stop False`.
2. **Exact Hugging Face source:** `hf_repo=afrideva/Tinystories-gpt-0.1-3m-GGUF` with `hf_file=tinystories-gpt-0.1-3m.Q2_K.gguf` started through `ServiceController` on port `8082`; the runtime downloaded the exact file with `token=False`, `/v1/models` returned `200`, one direct chat-completion request returned `200`, and stop again reported `running_after_stop False`.

The 3M TinyStories model is a lifecycle test artifact rather than a quality benchmark; its generated text was syntactically poor, but both server lifecycle and direct OpenAI-compatible request paths were exercised.

SAM endpoint tests still use an injected test predictor. A real SAM smoke test remains blocked by the absence of a compatible checkpoint; no real checkpoint load or prediction is claimed. Priority Map is installed and its adapter symbols imported, but no complete Priority Map analysis or moving sequence with `sam_step > 1` was run because real SAM output remains unavailable. The adapter's local propagation behavior is unit-tested: it computes DIS flow per frame, remaps retained masks and centroids, tracks median displacement, and replaces the propagated set exactly once on the next SAM frame.

## Remote smoke test

On `192.168.1.20`, run Django, open **Host**, and save the local LAN/VPN address and instruction port:

```powershell
.venv\Scripts\Activate.ps1
python manage.py runserver 127.0.0.1:8000
```

The Host portal on that machine starts:

```text
python -m core.services.instruction_server --host 192.168.1.20 --public-host 192.168.1.20 --port 9000 --log-dir logs/instruction
```

On the client:

```powershell
Invoke-RestMethod http://192.168.1.20:9000/health
python manage.py runserver 127.0.0.1:8000
```

Configure the remote node details in that client's saved configuration or tool/model settings, then start services through the preserved remote instruction-client architecture. The instruction port must be reachable from the client, and the selected inference ports must be reachable directly from the client. No inference request should be sent to port `9000`.

## Troubleshooting

* **LLM startup timeout**: inspect `logs/llm-<port>.log`; verify one GGUF path or the exact Hugging Face repo/file and increase `startup_timeout`.
* **SAM checkpoint error**: verify the file exists on the machine that runs SAM3 and that its installed SAM3 package accepts the checkpoint format.
* **Remote start rejected**: remove `command`, `shell`, and executable fields; remote control accepts service settings, not shell arrays.
* **Direct inference connection refused**: check the service's bind host, firewall, VPN route, and that the returned endpoint uses the host's reachable LAN/VPN address.
* **Priority Map import error**: install the `pipeline` extra and its heavyweight dependencies in the same environment as Django.
* **No images found**: Priority Map accepts image folders; image/video preparation must produce readable image files and ignores macOS metadata files.
* **Unexpected partial output**: inspect `analysis.log` and the output directory. Existing files are intentionally retained after a failed run.

## Prototype limitations and integration decision

Priority Map commit `ea6d1064175b20c1e90dd3f1ffb0b4173f68e03d` exposes `PriorityMapRunner` but does not accept LLM/SAM client objects. Almost ARCADIA verifies that `PriorityMapRunner`, `SceneUnderstanding`, and `Segment` exist, temporarily substitutes only the latter two while constructing the runner, and restores both symbols even if construction fails. The adapter supplies direct `LLMClient` and `SAMClient` implementations while leaving frame loading, local DIS optical flow, clustering, heatmaps, graph construction, and output writing to Priority Map. This narrow integration is unit-tested against a compatible fake module, not a real installed Priority Map environment.

The prototype supports one analysis at a time and one serialized SAM predictor per host. It does not implement authentication, TLS, multi-user isolation, concurrent analyses, model download management, hardware discovery, arbitrary remote commands, or persistent service recovery. The committed example contains no secrets and no user-specific absolute paths.
