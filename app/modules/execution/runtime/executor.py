"""
Execution Executor

Responsible for converting subprocess execution results into
NxZenAI Studio execution models.

The executor does NOT know anything about:

- FastAPI
- MongoDB
- Notebook Repository

It simply executes code using ExecutionProcess and returns
ExecutionResult objects.
"""

from __future__ import annotations

import time
import traceback

from app.modules.execution.models import (
    ExecutionOutput,
    ExecutionRequest,
    ExecutionResult,
)

from app.modules.execution.runtime.process import (
    ExecutionProcess,
)


class ExecutionExecutor:
    """
    Executes Python code through an ExecutionProcess.

    Converts subprocess output into AI Studio runtime models.
    """

    def __init__(
        self,
        process: ExecutionProcess,
    ) -> None:

        self.process = process

    def execute(
        self,
        request: ExecutionRequest,
        execution_count: int,
    ) -> ExecutionResult:
        """
        Execute a notebook cell.

        Parameters
        ----------
        request
            Execution request.

        execution_count
            Current notebook execution count.

        Returns
        -------
        ExecutionResult
        """

        start = time.perf_counter()

        try:

            stdout, stderr, return_code = self.process.execute(
                request.source
            )

            outputs: list[ExecutionOutput] = []

            if stdout:

                outputs.append(
                    ExecutionOutput(
                        output_type="stream",
                        content=stdout,
                    )
                )

            if stderr:

                if return_code == 0:

                    outputs.append(
                        ExecutionOutput(
                            output_type="stderr",
                            content=stderr,
                        )
                    )

                else:

                    outputs.append(
                        ExecutionOutput(
                            output_type="error",
                            content={
                                "stderr": stderr,
                            },
                        )
                    )

            execution_time = (
                time.perf_counter() - start
            ) * 1000

            return ExecutionResult(
                success=return_code == 0,
                execution_count=execution_count,
                outputs=outputs,
                execution_time_ms=execution_time,
            )

        except Exception as exc:

            execution_time = (
                time.perf_counter() - start
            ) * 1000

            return ExecutionResult(
                success=False,
                execution_count=execution_count,
                outputs=[
                    ExecutionOutput(
                        output_type="error",
                        content={
                            "ename": exc.__class__.__name__,
                            "evalue": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                ],
                execution_time_ms=execution_time,
            )