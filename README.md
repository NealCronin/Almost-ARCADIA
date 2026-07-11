# Almost ARCADIA

A distributed orchestration interface for drone target tracking and inference workloads. The framework splits compute between a **Client** device (coordinates the pipeline, reads datasets locally) and a **Host** device (inference compute over LAN/VPN).

## Architecture

```
+-----------------------+                    +-----------------------+
|   Client              |                    |   Host                |
|   Device              |<------LAN/VPN----->|   Device              |
|                       |  HTTP (GET/POST)   |                       |
| - Dataset Reader      |  Inference         | - LLM Server          |
| - Stream Viewer       |  Requests          | - SAM3 Model          |
| - UI Controls         |                    | - Stateless API       |
| - Optional local LLM  |                    | - Managed services    |
| - Optional local SAM  |                    | - External services   |
+-----------------------+                    +-----------------------+
```

### Key Design Decisions

- **Explicit Client/Host architecture** — the user chooses which computer runs each workload.
- **Independent LLM and SAM routing** — LLM and SAM can run locally or remotely independently.
- **Persistent JSON settings** — all configuration is saved to a JSON file outside the repository.
- **Managed and external services** — the application can either manage the process lifecycle or connect to an already-running server.
- **No hard-coded model families** — users can select any compatible model file without source changes.
- **Raw argument support** — advanced users can supply arbitrary backend flags.

## Configuration

### Settings File Location

Settings are persisted to a versioned JSON file. The location depends on the platform:

| Platform | Path |
|----------|------|
| Windows  | `%APPDATA%\AlmostARCADIA\settings.json` |
| macOS    | `~/Library/Application Support/AlmostARCADIA/settings.json` |
| Linux    | `$XDG_CONFIG_HOME/almost-arcadia/settings.json` or `~/.config/almost-arcadia/settings.json` |

Override with the `ALMOST_ARCADIA_CONFIG_DIR` environment variable:

```bash
export ALMOST_ARCADIA_CONFIG_DIR=/path/to/custom/config
```

### Settings Precedence

1. Explicit JSON settings file (saved via UI or API)
2. Environment variables (when no saved value exists)
3. Hard-coded application defaults

Saved configuration is NOT overwritten by environment variables on restart.

### Settings Schema (version 1)

```json
{
  "version": 1,
  "client": {
    "llm_mode": "local",
    "sam3_mode": "remote",
    "dataset_path": "",
    "remote_host": {
      "host": "127.0.0.1",
      "port": 8080,
      "scheme": "http"
    },
    "local_llm": {
      "service_mode": "managed",
      "executable": "llama-server",
      "model_path": "/path/to/model.gguf",
      "base_url": "",
      "api_format": "llama-completion",
      "port": 8081,
      "arguments": []
    },
    "local_sam3": {
      "service_mode": "managed",
      "weights_path": "/path/to/sam3.pt",
      "arguments": []
    }
  },
  "host": {
    "listen_ip": "0.0.0.0",
    "listen_port": 8080,
    "llm": { "...same shape as client LLM..." },
    "sam3": { "...same shape as client SAM..." }
  }
}
```

## Routing

LLM and SAM each have an independent execution mode. All four combinations are supported:

| LLM Mode | SAM Mode | Behavior |
|----------|----------|----------|
| local    | local    | Both run on this machine |
| local    | remote   | LLM local, SAM sent to Host |
| remote   | local    | LLM sent to Host, SAM local |
| remote   | remote   | Both sent to Host |

Legacy `routing_mode` (single global value) is automatically migrated on first load.

## Service Modes

### Managed by Almost ARCADIA
The application launches and controls the backend process:
- **Start / Stop / Restart** buttons in the UI
- Configurable executable, model path, host, port, and raw arguments
- Startup timeout and health checks
- Bounded log output (2000 lines max)
- Process status visible in the UI
- Command preview shows the generated launch command

### Connect to Existing Server
The user manages the backend independently:
- Only need to configure the base URL and model ID
- Almost ARCADIA does not start or stop the process
- Health checks verify connectivity

Toggling between modes preserves the configuration for both.

## Managed Service Lifecycle

Services have unique identities: `client:llm`, `client:sam3`, `host:llm`, `host:sam3`.

- **Start**: spawns the process, waits for health check, reports status
- **Stop**: terminates the process gracefully (SIGTERM, then SIGKILL after 5s)
- **Restart**: stops then starts
- **Status**: reports state (stopped, starting, running, unhealthy, stopping, failed, external)
- **Logs**: recent output (up to 2000 lines)

Inference requests use the already-running service. Starting a service per request is no longer done.

## Host API

The Host API uses **correct HTTP methods**:
- `GET /api/host/status/` — listener and service status
- `POST /api/host/evaluate-llm/` — LLM inference (uses Host's own configuration)
- `POST /api/host/evaluate-sam3/` — SAM3 inference (uses Host's own weights path)

The Host listener uses `ThreadingHTTPServer` for concurrent request handling. Status requests work while inference is running.

## Raw Arguments

Raw arguments are stored as a JSON array and appended after structured arguments. The backend's normal last-flag-wins precedence applies. The generated command is built as a `list[str]` for `subprocess.Popen` with `shell=False`.

## Settings API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/settings/` | GET | Return current normalized settings |
| `/api/settings/` | PUT | Merge request body and save |
| `/api/settings/reset/` | POST | Reset to factory defaults |

## Running

```bash
# Create environment
conda create -n almost_arcadia python=3.10 -y
conda activate almost_arcadia

# Install
pip install django requests opencv-python numpy python-dotenv

# Run
python manage.py runserver 0.0.0.0:8000
```

## Tests

```bash
# Run all tests
python -m unittest discover -s core_orchestrator/tests -p "test_*.py" -v

# Django system checks
python manage.py check

# Syntax validation
python -m compileall .
```

Tests do not require model files, GPU, or a remote computer. All heavyweight components are mocked.

## Recovering from Malformed Settings

If the settings JSON file becomes malformed:
1. The application logs the parsing failure
2. It attempts to load the last valid backup (`settings.json.bak`)
3. If backup is also malformed, factory defaults are loaded
4. The malformed file is preserved as `settings.json.malformed` for inspection
5. A warning is logged so the admin can investigate

## API Endpoints

### Host API (Background Listener)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/host/status/` | GET | Listener + service status |
| `/api/host/evaluate-llm/` | POST | LLM inference (Host-owned config) |
| `/api/host/evaluate-sam3/` | POST | SAM3 segmentation (Host-owned weights) |

### Django Views

| URL | View | Description |
|-----|------|-------------|
| `/` | `landing_page` | Landing page with role selection |
| `/host/` | `host_portal` | Host configuration and service management |
| `/client/` | `client_portal` | Client tool selection |
| `/client/heatmap/` | `heatmap_dashboard` | Heatmap stream with routing config |
| `/stream/heatmap/` | `heatmap_stream` | MJPEG video stream with overlays |
| `/api/settings/` | `settings_view` | Settings GET/PUT |
| `/api/settings/reset/` | `settings_reset` | Settings reset |