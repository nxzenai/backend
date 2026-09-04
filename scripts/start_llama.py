import os
import subprocess
import time

from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = os.getenv(
    "GENAI_FAST_MODEL_REPO",
    "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
)

QUANT = os.getenv(
    "GENAI_FAST_MODEL_QUANT",
    "Q4_K_M",
)

MODEL_ALIAS = os.getenv(
    "GENAI_FAST_MODEL",
    "llama-3.2-1b-instruct",
)

MODEL_DIR = os.getenv(
    "GENAI_MODEL_DIR",
    "/opt/models",
)

LLAMA_PORT = os.getenv(
    "GENAI_LLAMA_PORT",
    "8081",
)


def find_model_file() -> str:
    print(f"[GenAI] Searching repository: {REPO_ID}", flush=True)

    files = list_repo_files(REPO_ID)

    candidates = [
        filename
        for filename in files
        if filename.lower().endswith(".gguf")
        and QUANT.lower() in filename.lower()
    ]

    if not candidates:
        raise RuntimeError(
            f"No {QUANT} GGUF model found in {REPO_ID}"
        )

    return candidates[0]


def resolve_model_path(filename: str) -> str:
    model_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(model_path):
        print(
            f"[GenAI] Model ready: {model_path}",
            flush=True,
        )
        return model_path

    print(
        "[GenAI] Model not found locally. Downloading...",
        flush=True,
    )

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=MODEL_DIR,
    )

    print(
        f"[GenAI] Model ready: {model_path}",
        flush=True,
    )

    return model_path


def start_llama_server(model_path: str) -> subprocess.Popen:
    command = [
        "/usr/local/bin/llama-server",
        "--model",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        LLAMA_PORT,
        "--ctx-size",
        "2048",
        "--threads",
        "2",
        "--parallel",
        "1",
        "--alias",
        MODEL_ALIAS,
    ]

    print(
        f"[GenAI] Starting llama-server on "
        f"127.0.0.1:{LLAMA_PORT}",
        flush=True,
    )

    return subprocess.Popen(command)


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    filename = find_model_file()

    print(
        f"[GenAI] Selected model: {filename}",
        flush=True,
    )

    model_path = resolve_model_path(filename)

    process = start_llama_server(model_path)

    while True:
        return_code = process.poll()

        if return_code is not None:
            raise RuntimeError(
                f"llama-server exited with code {return_code}"
            )

        time.sleep(5)


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            f"[GenAI] Inference startup failed: {exc}",
            flush=True,
        )
        raise
