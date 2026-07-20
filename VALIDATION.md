# Validation report

Validation was run against the exported source tree with automatic listener startup disabled.

```text
pytest -q
51 passed

ruff check .
All checks passed

mypy core web project manage.py
Success: no issues found in 38 source files

python manage.py check
System check identified no issues (0 silenced)
```

Additional checks performed:

- Rendered the home, client, Host, model settings, analysis, results, and endpoint-test pages with Django's test client.
- Rendered model settings in both shared-Visual and separate-Visual modes.
- Parsed rendered HTML and confirmed two separate LLM forms in separate mode with no nested forms.
- Built a wheel and confirmed Django templates and static assets were included.
- Installed the project in editable mode and imported the core packages.

Not executed in this environment:

- Native llama.cpp compilation.
- Real GGUF or projector downloads.
- Real LLM inference.
- Real SAM3 checkpoint loading/inference.
- Full Priority Map run with physical model services.
- Remote-host firewall and VPN testing.
- Real SAM3 text-prompt segmentation and visual inspection of returned masks.


Additional regression coverage in this revision verifies:

- the app does not emit or store a forced model alias;
- a native `--alias` supplied through Additional Arguments is passed through;
- the inference client discovers the server's actual model ID from `/v1/models`;
- identical remote services are reused rather than restarted;
- remote model startup uses the extended timeout;
- remote startup failures return their real detail instead of an unhandled HTTP 500.

Additional regression coverage in version 0.5.0 verifies:

- the default cache tree is `huggingface/models` plus `huggingface/mmproj`, with `ARCADIA_HUGGINGFACE_DIR` precedence and deprecated `ARCADIA_MODELS_DIR` compatibility;
- browser uploads and service logs use `ARCADIA_STATE_DIR`, not repository `workspace/` or `uploads/`;
- the Model Settings and Host pages expose SAM3 checkpoint upload/save controls;
- local checkpoint uploads are streamed into `huggingface/models`, and remote instruction servers return the real remote compute-node path;
- checkpoint writes are atomic, require `.pt`, and enforce a configurable size limit;
- SAM3 validates canonical `image_base64`, `text`, and bounded `confidence` fields while retaining FastAPI validation details;
- invalid base64 and non-image bytes return JSON 4xx details;
- compact PNG masks support empty, one-mask, and multiple-mask detections, including masks with no box/class array;
- the SAM client sends the canonical request for every Priority Map search term and preserves server error bodies;
- the browser SAM3 test returns a rendered PNG overlay and the model alias remains untouched;
- JavaScript syntax passes `node --check web/static/web/app.js`.

The browser-to-remote transfer was validated with test payloads, not the full production `sam3.pt` file.
