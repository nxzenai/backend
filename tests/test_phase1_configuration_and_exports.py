import json
import os
import re
import subprocess
from pathlib import Path

import nbformat
import pytest
from pydantic import ValidationError

os.environ["DEBUG"] = "false"

from app.core.config.settings import Settings

BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BACKEND_ROOT.parent
FRONTEND_ROOT = WORKSPACE_ROOT / "frontend-main"


def production_settings(**overrides):
    values = {
        "ENVIRONMENT": "production",
        "APP_DEBUG": False,
        "SECRET_KEY": "a-production-secret-that-is-longer-than-thirty-two-characters",
        "MONGODB_URL": "mongodb+srv://database.example.invalid/app",
        "MINIO_ACCESS_KEY": "deployed-access-key",
        "MINIO_SECRET_KEY": "deployed-secret-key",
        "MINIO_SECURE": True,
        "FRONTEND_URL": "https://app.example.com",
        "CORS_ORIGINS": "https://app.example.com,https://admin.example.com",
        "LOG_LEVEL": "INFO",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_production_settings_accept_explicit_secure_configuration():
    configured = production_settings()
    assert configured.allowed_origins == [
        "https://app.example.com",
        "https://admin.example.com",
    ]


def test_staging_also_rejects_development_defaults():
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings(_env_file=None, ENVIRONMENT="staging")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"APP_DEBUG": True}, "DEBUG"),
        ({"SECRET_KEY": "supersecretkey123"}, "SECRET_KEY"),
        ({"MONGODB_URL": "mongodb://localhost:27017"}, "MONGODB_URL"),
        ({"MONGODB_URL": ""}, "MONGODB_URL"),
        ({"MINIO_SECURE": False}, "MINIO_SECURE"),
        ({"MINIO_SECRET_KEY": "minioadmin"}, "MinIO"),
        ({"CORS_ORIGINS": "http://app.example.com"}, "HTTPS"),
        ({"LOG_LEVEL": "DEBUG"}, "LOG_LEVEL"),
    ],
)
def test_production_settings_fail_fast_on_unsafe_values(override, message):
    with pytest.raises(ValidationError, match=message):
        production_settings(**override)


def test_example_environment_has_unique_keys_and_no_embedded_credentials():
    lines = (BACKEND_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    keys = [
        line.split("=", 1)[0]
        for line in lines
        if line and not line.startswith("#") and "=" in line
    ]
    assert len(keys) == len(set(keys))
    content = "\n".join(lines)
    assert not re.search(r"mongodb(?:\+srv)?://[^/\s:@]+:[^@\s]+@", content)


def test_tracked_files_do_not_contain_common_live_secret_shapes():
    patterns = {
        "credentialed MongoDB URI": re.compile(
            r"mongodb(?:\+srv)?://[^/\s:@]+:[^@\s]+@"
        ),
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    }
    findings = []
    for repository in (BACKEND_ROOT, FRONTEND_ROOT):
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for relative in tracked:
            path = repository / relative
            if not path.is_file() or path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".ico",
                ".lock",
            }:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in patterns.items():
                if pattern.search(content):
                    findings.append(f"{repository.name}/{relative}: {label}")
    assert findings == []


def test_frontend_ipynb_builder_produces_nbformat_valid_notebook():
    exporter = (FRONTEND_ROOT / "src" / "utils" / "exportNotebook.ts").as_uri()
    script = f"""
      import {{ buildIPYNB }} from {json.dumps(exporter)};
      const notebook = {{ title: 'Release test', description: null }};
      const cells = [
        {{ id: 'code:1', cell_type: 'code', source: 'print(1)', metadata: {{}}, execution_count: 1,
           outputs: [{{ output_type: 'stream', content: {{ name: 'stderr', text: 'warning' }}, metadata: {{ name: 'stderr' }} }}] }},
        {{ id: 'markdown-1', cell_type: 'markdown', source: '# Heading', metadata: {{}}, execution_count: null, outputs: [] }}
      ];
      process.stdout.write(JSON.stringify(buildIPYNB(notebook, cells)));
    """
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=FRONTEND_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    document = nbformat.from_dict(json.loads(completed.stdout))
    nbformat.validate(document)
    assert document.cells[0].outputs[0].name == "stderr"
