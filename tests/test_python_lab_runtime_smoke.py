import pytest

from app.modules.execution.kernel_manager import KernelManager
from app.modules.execution.environment import detect_runtime_environment


@pytest.mark.asyncio
async def test_cpu_notebook_runtime_smoke(tmp_path):
    manager = KernelManager(
        workspace_root=str(tmp_path), output_max_bytes=2 * 1024 * 1024
    )
    notebook_id = "runtime-smoke"
    await manager.start_kernel(notebook_id)
    try:
        outputs, count = await manager.execute(notebook_id, 'print("Python Lab ready")')
        assert count == 1
        assert any("Python Lab ready" in str(output.content) for output in outputs)

        outputs, _ = await manager.execute(
            notebook_id,
            "import numpy as np\nimport pandas as pd\nprint(np.__version__, pd.__version__)",
        )
        assert not any(output.output_type == "error" for output in outputs)

        outputs, _ = await manager.execute(
            notebook_id,
            "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3], [1, 4, 9])\nplt.show()",
        )
        assert any(
            isinstance(output.content, dict)
            and "image/png" in output.content.get("data", {})
            for output in outputs
        )

        outputs, _ = await manager.execute(
            notebook_id,
            "from sklearn.datasets import load_iris\nfrom sklearn.ensemble import RandomForestClassifier\nX, y = load_iris(return_X_y=True)\nprint(RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y).score(X, y))",
        )
        assert not any(output.output_type == "error" for output in outputs)

        module_names = {
            "torch": "torch",
            "tensorflow": "tensorflow",
            "transformers": "transformers",
            "datasets": "datasets",
            "spacy": "spacy",
            "nltk": "nltk",
        }
        status = {
            item["name"]: item for item in detect_runtime_environment()["packages"]
        }
        available = [
            module
            for distribution, module in module_names.items()
            if status[distribution]["installed"]
        ]
        source = "\n".join(
            f"import {name}; print('{name}', getattr({name}, '__version__', 'installed'))"
            for name in available
        )
        if source:
            outputs, _ = await manager.execute(notebook_id, source)
            assert not any(output.output_type == "error" for output in outputs)
    finally:
        await manager.shutdown_kernel(notebook_id)
