"""
llama_server_helper.py

Manages a local ``llama.cpp`` HTTP server (via ``llama-server`` or the
Python ``llama_cpp`` bindings) for LLM inference.

Responsibilities
----------------
* Spawn / stop the server process through :class:`ProcessManager`.
* Parse a user-supplied model-path string.
* Expose a synchronous ``evaluate(prompt)`` method that hits the
  local server and returns the generated text.
"""

import json
import logging
import time
from typing import Optional

import requests

from .process_manager import ProcessManager

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_MODEL_PATH = ""
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8081
SERVER_STARTUP_TIMEOUT = 30  # seconds


class LlamaServerError(Exception):
    """Raised when the Llama server cannot be reached or returns an error."""


class LlamaServerHelper:
    """
    Wrapper around a local llama.cpp server instance.

    Parameters
    ----------
    model_path : str
        Path to the GGUF model file.
    host : str
        Bind address for the server.
    port : int
        TCP port for the server.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.model_path = model_path
        self.host = host
        self.port = port
        self._pm = ProcessManager.instance()
        self._server_ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start_server(self) -> bool:
        """
        Launch ``llama-server`` in the background and wait until it
        responds to health-check requests.

        Returns
        -------
        bool
            ``True`` if the server became healthy within the timeout.
        """
        if not self.model_path:
            logger.warning("No model path configured; server will not start")
            return False

        if self._pm.is_running("llama_server"):
            self._server_ready = True
            return True

        cmd = [
            "llama-server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--n-predict", "256",
            "--embedding",
        ]

        try:
            self._pm.start(name="llama_server", cmd=cmd, capture_output=True)
        except Exception as exc:
            logger.error("Failed to launch llama-server: %s", exc)
            return False

        # Poll until healthy
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            try:
                resp = requests.get(f"http://{self.host}:{self.port}/health", timeout=1)
                if resp.status_code == 200:
                    self._server_ready = True
                    logger.info("llama-server healthy at %s:%d", self.host, self.port)
                    return True
            except requests.RequestException:
                pass
            time.sleep(0.5)

        logger.error("llama-server failed to start within %ds", SERVER_STARTUP_TIMEOUT)
        return False

    def stop_server(self) -> None:
        """Gracefully shut down the background server."""
        self._pm.stop("llama_server", force=False)
        self._server_ready = False

    def is_running(self) -> bool:
        return self._pm.is_running("llama_server")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def evaluate(self, prompt: str, **kwargs) -> str:
        """
        Send *prompt* to the local server and return generated text.

        Parameters
        ----------
        prompt : str
            The user prompt.
        **kwargs
            Extra keys forwarded to the server (e.g. ``temperature``,
            ``top_p``, ``n_predict``).

        Returns
        -------
        str
            The generated completion text.

        Raises
        ------
        LlamaServerError
            If the server is unreachable or returns a non-200 status.
        """
        if not self.is_running():
            raise LlamaServerError("llama-server is not running. Call start_server() first.")

        payload = {
            "prompt": prompt,
            "n_predict": kwargs.get("n_predict", 256),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False,
        }

        url = f"http://{self.host}:{self.port}/completion"
        resp = requests.post(url, json=payload, timeout=kwargs.get("timeout", 60))
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", data.get("generation", ""))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def status(self) -> dict:
        info: dict = {
            "model_path": self.model_path,
            "host": self.host,
            "port": self.port,
            "running": self.is_running(),
            "ready": self._server_ready,
        }
        if self.is_running():
            record = self._pm.get("llama_server")
            if record:
                info["pid"] = record.pid
        return info
