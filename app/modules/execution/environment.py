from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
from functools import lru_cache

SUPPORTED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scikit-learn": "sklearn",
    "jupyter-client": "jupyter_client",
    "ipython": "IPython",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "transformers": "transformers",
    "datasets": "datasets",
    "spacy": "spacy",
    "nltk": "nltk",
}


@lru_cache(maxsize=1)
def detect_runtime_environment() -> dict:
    packages = []
    imported_modules = {}
    for distribution, module in SUPPORTED_PACKAGES.items():
        installed = False
        version = None
        error = None
        if importlib.util.find_spec(module) is not None:
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = None
            try:
                imported = importlib.import_module(module)
                imported_modules[module] = imported
                installed = True
                if version is None:
                    version = str(getattr(imported, "__version__", "unknown"))
            except Exception as exc:
                error = f"{type(exc).__name__}: import failed"
                installed = False
                if version is None:
                    version = "unknown"
        packages.append(
            {
                "name": distribution,
                "installed": installed,
                "version": version,
                "error": error,
            }
        )

    gpu_available = False
    gpu_details = None
    if "torch" in imported_modules:
        try:
            torch = imported_modules["torch"]
            if bool(torch.cuda.is_available()):
                gpu_available = True
                gpu_details = str(torch.cuda.get_device_name(0))
        except Exception:
            pass
    if not gpu_available and "tensorflow" in imported_modules:
        try:
            tensorflow = imported_modules["tensorflow"]
            devices = tensorflow.config.list_physical_devices("GPU")
            if devices:
                gpu_available = True
                gpu_details = str(devices[0].name)
        except Exception:
            pass

    return {
        "python_version": platform.python_version(),
        "packages": packages,
        "gpu_available": gpu_available,
        "gpu_details": gpu_details,
    }
