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

The `pipeline` extra installs the current Priority Map repository directly from GitHub. The `sam` extra installs the Ultralytics package used by the current Priority Map SAM3 implementation. If the external SAM3 source has a newer installation procedure, install that source in the same environment and keep the checkpoint path in `config.json` accurate.

### llama-cpp-python variants

`llama-cpp-python` is a base dependency. Select the build appropriate to the machine before or instead of the editable install if a prebuilt wheel is unavailable.

CPU:

```powershell
$env:CMAKE_ARGS="-DGGML_NATIVE=OFF"
pip install llama-cpp-python
```

CUDA (a compatible CUDA toolkit and Visual C++ build tools are required on Windows):

```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python
```

Apple Metal:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

Do not use split GGUF files. Configure either one exact `model_path`, or both `hf_repo` and `hf_file`. Gated Hugging Face repositories are not supported by this prototype.

### SAM3 checkpoint

Install the real SAM3-capable package used by Priority Map, then manually place a compatible checkpoint outside source control, for example:

```text
checkpoints/sam3.pt
```

Set `services.sam3.settings.checkpoint` to that path. A missing or unloadable checkpoint is a startup error. Runtime SAM3 responses are never mocked or synthesized.

## Configuration

`config.json` is a committed, non-secret example. Edit the model and checkpoint paths for the machine. A service entry has a node name, service type, inference port, and runtime settings:

```json
{
  "nodes": {
    "local": {"mode": "local", "host": "127.0.0.1"},
    "example_remote": {"mode": "remote", "host": "192.168.1.20", "instruction_port": 9000}
  },
  "services": {
    "llm": {
      "node": "local",
      "service_type": "llm",
      "port": 8081,
      "settings": {"model_path": "models/model.gguf", "bind_host": "0.0.0.0"}
    },
    "sam3": {
      "node": "local",
      "service_type": "sam3",
      "port": 8090,
      "settings": {"checkpoint": "checkpoints/sam3.pt", "bind_host": "0.0.0.0"}
    }
  },
  "pipeline": {"sam_step": 5, "run_at_source_fps": false, "sam_resize": null},
  "output_root": "outputs"
}
```

The UI saves configuration atomically through a same-directory temporary file and replacement. `ARCADIA_CONFIG` can point Django at another JSON file. No live process state is stored in JSON or SQLite.

Supported useful runtime settings include `bind_host`, `python_executable` for local trusted setup, `startup_timeout`, `n_ctx`, `n_gpu_layers`, `chat_format`, `model_alias`, `extra_args`, and the exact model source fields. The remote instruction API rejects executable and arbitrary command fields. Unit tests alone use the command escape hatch.

## Start a host instruction server

Run this on the remote compute host. `--host` is the bind address; `--public-host` is the address returned to the client for direct inference:

```powershell
python -m core.services.instruction_server --host 0.0.0.0 --public-host 192.168.1.20 --port 9000 --log-dir logs
```

The host must have the Almost ARCADIA environment, llama-cpp-python, SAM3 dependencies, and the checkpoint/model paths configured for that host. The server owns children until it exits. On normal shutdown it stops owned children. A host restart loses live state; the client starts the desired service again when needed.

Health check from the client:

```powershell
Invoke-RestMethod http://192.168.1.20:9000/health
```

## Start Django

From the client environment:

```powershell
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

Open <http://127.0.0.1:8000/>. Use **Nodes** to define local or remote hosts, **Services** to start or replace LLM/SAM3 services, and **Analysis** to set pipeline options and run one analysis. State-changing actions use POST and successful forms redirect.

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

The coordinator permits one active analysis. It saves `effective_settings.json` and `analysis.log` before work begins. Incremental Priority Map output is preserved after failures. If a direct service failure is detected, the coordinator reprovisions both configured services once and retries the pipeline once; a second failure ends the analysis.

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

The repository tests mock subprocesses, HTTP responses, and SAM predictors at their boundaries. They do not download model weights or claim that a heavyweight checkpoint is installed.

## Remote smoke test

On `192.168.1.20`:

```powershell
.venv\Scripts\Activate.ps1
python -m core.services.instruction_server --host 0.0.0.0 --public-host 192.168.1.20 --port 9000 --log-dir logs
```

On the client:

```powershell
Invoke-RestMethod http://192.168.1.20:9000/health
python manage.py runserver 127.0.0.1:8000
```

In the UI, add the remote node, assign either service to that node, start it, then use the direct inference commands above with the endpoint returned by the server. The instruction port must be reachable from the client, and the selected inference ports must be reachable directly from the client. No inference request should be sent to port `9000`.

## Troubleshooting

* **LLM startup timeout**: inspect `logs/llm-<port>.log`; verify one GGUF path or the exact Hugging Face repo/file and increase `startup_timeout`.
* **SAM checkpoint error**: verify the file exists on the machine that runs SAM3 and that its installed SAM3 package accepts the checkpoint format.
* **Remote start rejected**: remove `command`, `shell`, and executable fields; remote control accepts service settings, not shell arrays.
* **Direct inference connection refused**: check the service's bind host, firewall, VPN route, and that the returned endpoint uses the host's reachable LAN/VPN address.
* **Priority Map import error**: install the `pipeline` extra and its heavyweight dependencies in the same environment as Django.
* **No images found**: Priority Map accepts image folders; image/video preparation must produce readable image files and ignores macOS metadata files.
* **Unexpected partial output**: inspect `analysis.log` and the output directory. Existing files are intentionally retained after a failed run.

## Prototype limitations and integration decision

The current Priority Map repository exposes `PriorityMapRunner` but does not accept LLM/SAM client objects. Almost ARCADIA therefore patches only the runner's imported `SceneUnderstanding` and `Segment` class symbols during runner construction. The adapter supplies direct `LLMClient` and `SAMClient` implementations and leaves frame loading, clustering, heatmaps, graph construction, and output writing to Priority Map. This keeps the integration narrow but depends on the external runner's current class names and constructor shape.

The prototype supports one analysis at a time and one serialized SAM predictor per host. It does not implement authentication, TLS, multi-user isolation, concurrent analyses, model download management, hardware discovery, arbitrary remote commands, or persistent service recovery. The committed example contains no secrets and no user-specific absolute paths.
