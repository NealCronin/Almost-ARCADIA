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
    Helper class for Llama.cpp server management and inference.

    Attributes
    ----------
    model_path : str
        Path to the GGUF model file.
    host : str
        Host to bind the server to.
    port : int
        Port to bind the server to.
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
        self._server_process_name = "llama_server"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start_server(self) -> bool:
        """
        Start the llama-server background process.

        Returns True if the server started successfully and responded to health check.
        """
        if not self.model_path:
            raise LlamaServerError("Model path not configured")

        cmd = [
            "llama-server",
            "--model",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        try:
            pid = self._pm.start(self._server_process_name, cmd)
            logger.info("Waiting for server to be ready...")

            # Wait for server to be ready
            base_url = f"http://{self.host}:{self.port}"
            start_time = time.time()
            while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
                try:
                    resp = requests.get(f"{base_url}/health", timeout=2)
                    if resp.status_code == 200:
                        logger.info("Server is ready at %s", base_url)
                        self._server_ready = True
                        return True
                except requests.RequestException:
                    pass
                time.sleep(1)

            logger.error("Server failed to start within timeout")
            self._pm.stop(self._server_process_name, force=True)
            return False
        except Exception as exc:
            logger.exception("Failed to start server: %s", exc)
            return False

    def stop_server(self) -> None:
        """Gracefully shut down the background server."""
        self._pm.stop(self._server_process_name, force=False)
        self._server_ready = False

    def is_running(self) -> bool:
        """Check if the server process is running."""
        return self._pm.is_running(self._server_process_name)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def evaluate(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to the Llama server and return the generated response.

        Parameters
        ----------
        prompt : str
            The input prompt text.
        **kwargs
            Additional generation parameters (max_tokens, temperature, etc.).

        Returns
        -------
        str
            The generated text response.

        Raises
        ------
        LlamaServerError
            If the server is not running or the request fails.
        """
        if not self._server_ready and not self.is_running():
            raise LlamaServerError("Server is not running")

        base_url = f"http://{self.host}:{self.port}"

        # Build request payload
        payload = {
            "prompt": prompt,
            "n_predict": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stop": kwargs.get("stop", ["</s>"]),
        }

        try:
            resp = requests.post(
                f"{base_url}/completion",
                json=payload,
                timeout=kwargs.get("timeout", 120),
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("content", data.get("generation", ""))
        except requests.RequestException as exc:
            raise LlamaServerError(f"Server request failed: {exc}") from exc
        except (json.JSONDecodeError, KeyError) as exc:
            raise LlamaServerError(f"Invalid response from server: {exc}") from exc

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def status(self) -> dict:
        """Return the current server status and configuration."""
        info: dict = {
            "model_path": self.model_path,
            "host": self.host,
            "port": self.port,
            "running": self.is_running(),
            "ready": self._server_ready,
        }

        if self.is_running():
            info["pid"] = self._pm.get_pid(self._server_process_name)

        return info

    def evaluate_with_context(self, prompt: str, context: str, **kwargs) -> str:
        """
        Evaluate a prompt with additional context.

        This wraps the prompt and context into a single formatted input.

        Parameters
        ----------
        prompt : str
            The main question or instruction.
        context : str
            Additional context or background information.
        **kwargs
            Additional generation parameters.

        Returns
        -------
        str
            The generated response.
        """
        formatted_prompt = f"""Context:
{context}

Question:
{prompt}

Answer:"""
        return self.evaluate(formatted_prompt, **kwargs)
