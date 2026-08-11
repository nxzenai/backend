"""
Execution Process

Responsible for spawning and managing an isolated Python
subprocess for code execution.

This module MUST NOT know anything about:

- FastAPI
- MongoDB
- Notebook Repository
- Authentication

It simply executes Python code and returns stdout/stderr.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from app.modules.execution.constants import DEFAULT_TIMEOUT_SECONDS
from app.modules.execution.exceptions import (
    ExecutionFailed,
    ExecutionTimeout,
)


class ExecutionProcess:
    """
    Manages an isolated Python subprocess.

    Each execution runs inside a fresh Python interpreter.

    Future versions may replace this with:

        - Docker
        - Firecracker
        - Kubernetes
        - Remote Workers
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:

        self.timeout = timeout

    def execute(
        self,
        source: str,
    ) -> tuple[str, str, int]:
        """
        Execute Python source code.

        Returns
        -------
        tuple
            (
                stdout,
                stderr,
                return_code,
            )
        """

        temp_file = self._create_temp_script(source)

        try:

            process = subprocess.run(
                [sys.executable, str(temp_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return (
                process.stdout,
                process.stderr,
                process.returncode,
            )

        except subprocess.TimeoutExpired as exc:

            raise ExecutionTimeout() from exc

        except Exception as exc:

            raise ExecutionFailed() from exc

        finally:

            if temp_file.exists():
                temp_file.unlink()

    @staticmethod
    def _create_temp_script(
        source: str,
    ) -> Path:
        """
        Write source code into a temporary Python file.
        """

        temp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        )

        temp.write(source)

        temp.close()

        return Path(temp.name)