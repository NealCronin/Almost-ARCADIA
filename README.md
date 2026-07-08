# Drone Orchestrator

A distributed orchestration interface for drone target tracking pipelines. The framework splits compute load between a **Client** device (coordinates the pipeline, reads datasets locally) and a **Host** device (stateless compute box evaluating LLM or SAM3 inferences over LAN/VPN).

## Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│   Client        │                    │   Host          │
│   Device        │◄──────LAN/VPN─────►│   Device        │
│                 │                    │                 │
│ - Dataset Reader│   HTTP POST        │ - LLM Server    │
│ - Stream Viewer │   Inference        │ - SAM3 Model    │
│ - UI Controls   │   Requests         │ - Stateless API │
└─────────────────┘                    └─────────────────┘
```

## Phase 1: Environment Setup

Create an isolated Conda environment with all dependencies:

```bash
# Create and activate environment
conda create -n drone_orchestrator python=3.10 -y
conda activate drone_orchestrator

# Install dependencies
pip install django requests opencv-python pandas numpy python-dotenv llama-cpp-python torch torchvision

# Optional: Install segment-anything for SAM3 support
pip install segment-anything
```

## Phase 2: Project Structure

```
Almost-ARCADIA/
├── manage.py
├── drone_orchestrator/           # Project settings
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── core_orchestrator/            # Main application
│   ├── views.py                  # Django views & streaming
│   ├── urls.py                   # URL routing
│   ├── apps.py
│   ├── utils/
│   │   ├── model_host/
│   │   │   ├── process_manager.py       # Background process management
│   │   │   ├── llama_server_helper.py   # LLM inference helper
│   │   │   ├── sam3_server_helper.py    # SAM3 segmentation helper
│   │   │   ├── vpn_tunnel_helper.py     # Network connectivity checks
│   │   │   └── remote_client_helper.py  # Client-to-host HTTP calls
│   │   └── __init__.py
│   └── templates/
│       └── core_orchestrator/
│           ├── base.html                # Base template with TailwindCSS
│           ├── index.html               # Landing page (role selection)
│           ├── host_portal.html         # Host configuration dashboard
│           ├── tool_selection.html      # Client tool launcher
│           └── heatmap_dashboard.html   # Live stream dashboard
└── README.md
```

## Phase 3: Running the Server

### Starting the Development Server

```bash
# Activate environment
conda activate drone_orchestrator

# Run Django development server
python manage.py runserver 0.0.0.0:8000
```

The server will start at `http://localhost:8000`

### Accessing the Portals

1. **Landing Page** (`/`) - Choose between Host Portal or Client Portal
2. **Host Portal** (`/host/`) - Configure model paths and monitor services
3. **Client Portal** (`/client/`) - Access available tools
4. **Heatmap Dashboard** (`/client/heatmap/`) - View live video stream with target tracking

## Core Components

### Process Manager (`process_manager.py`)

Thread-safe utility for managing background binaries:
- Start/stop processes with PID tracking
- Capture stdout/stderr streams
- Safe termination (SIGTERM) and forced kill (SIGKILL)

### Llama Server Helper (`llama_server_helper.py`)

LLM inference wrapper using `llama-cpp-python`:
- Load GGUF model weights from path
- Start background `llama-server` process
- Synchronous `evaluate(prompt, context)` method

### SAM3 Server Helper (`sam3_server_helper.py`)

Segment Anything Model 3 wrapper:
- Load `sam3.pt` checkpoint
- Predict masks from points or boxes
- Extract target coordinates from segmentation

### VPN Tunnel Helper (`vpn_tunnel_helper.py`)

Network diagnostics:
- Check host reachability via TCP socket
- Enumerate VPN interfaces
- Verify tunnel connectivity

### Remote Client Helper (`remote_client_helper.py`)

Client-to-host communication:
- Serialize frames via HTTP POST
- Retry logic with exponential backoff
- Base64 image encoding/decoding

## API Endpoints

### Host API (Stateless)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/host/evaluate-llm/` | POST | Evaluate LLM prompt |
| `/api/host/evaluate-sam3/` | POST | Run SAM3 segmentation |
| `/api/host/status/` | GET | Get service status |

#### LLM Evaluation Request

```json
{
  "prompt": "Your question here",
  "context": "Optional context",
  "temperature": 0.7,
  "max_tokens": 512
}
```

#### SAM3 Evaluation Request

```json
{
  "frame_b64": "base64-encoded JPEG",
  "input_points": [[x1, y1], [x2, y2]],
  "input_boxes": [[x1, y1, x2, y2]]
}
```

### Streaming Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream/heatmap/` | GET | MJPEG video stream with heatmap overlay |

## Configuration

### Host Configuration

Set via Host Portal form or environment variables:

- `HOST_LISTEN_HOST` (default: `0.0.0.0`)
- `HOST_LISTEN_PORT` (default: `8080`)
- `model_path` - Path to LLM GGUF file
- `weights_path` - Path to SAM3 checkpoint

### Client Configuration

Set via Heatmap Dashboard controls:

- **Dataset Path**: Video file path or leave empty for camera
- **SAM3 Mode**: `local` or `remote`
- **LLM Mode**: `local` or `remote`
- **Host IP/Port**: Remote server address (when using remote mode)

## MJPEG Streaming

The heatmap dashboard uses Django's `StreamingHttpResponse` to serve MJPEG frames:

1. Camera/video source is opened with OpenCV
2. Each frame is processed through SAM3 inference
3. Target coordinates are extracted and drawn as heatmap overlays
4. Frames are encoded as JPEG and sent with `multipart/x-mixed-replace` boundary

Stream URL format:
```
/stream/heatmap/?sam3_mode=remote&host_ip=192.168.1.100&host_port=8080
```

## Security Notes

- CSRF protection is enabled for all POST endpoints
- Use `@csrf_exempt` only for API endpoints that receive external requests
- In production, configure `ALLOWED_HOSTS` and use HTTPS

## Troubleshooting

### Model Loading Failures

- Ensure model paths are absolute and files exist
- Check file permissions
- Verify model format (GGUF for LLM, PT for SAM3)

### Stream Not Loading

- Verify camera/video source is accessible
- Check firewall settings for MJPEG port
- Ensure OpenCV is properly installed

### Remote Host Unreachable

- Verify VPN/LAN connectivity
- Check host IP and port configuration
- Test with `vpn_tunnel_helper.verify_tunnel()`

## Future Enhancements

- WebSocket support for real-time bidirectional communication
- Multi-drone tracking with individual heatmaps
- Persistent logging and analytics dashboard
- Authentication and access control
