from dataclasses import dataclass, field
from collections import deque
from subprocess import Popen
from pathlib import Path


@dataclass
class ProcessSpec:
    command: list[str]
    working_directory: Path | None = None
    environment: dict[str, str] | None = None


@dataclass
class RunningProcess:
    process: Popen
    spec: ProcessSpec
    output_buffer_size: int = 200
    stdout_lines: deque[str] = field(init=False)
    stderr_lines: deque[str] = field(init=False)

    def __post_init__(self):
        self.stdout_lines = deque(maxlen=self.output_buffer_size)
        self.stderr_lines = deque(maxlen=self.output_buffer_size)

    @property
    def process_id(self) -> int:
        return self.process.pid
