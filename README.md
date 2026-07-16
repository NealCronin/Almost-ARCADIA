# Almost ARCADIA — GPT branch prototype

Clean prototype for starting model services locally or remotely, calling
their inference endpoints directly, and later connecting them to the
Priority Map pipeline.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows:

```powershell
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Start the Django UI

```bash
python manage.py migrate
python manage.py runserver
```

## Start a remote instruction server

```bash
python -m core.services.instruction_server --host 192.168.1.20 --port 9000
```

The instruction server starts or stops model services. Inference requests
go directly to the model service ports.
