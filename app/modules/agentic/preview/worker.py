from __future__ import annotations

import argparse
import asyncio
import socket
import uuid

from app.core.config.settings import settings
from app.core.database.mongodb import MongoDB
from app.modules.agentic.build.repository import AgenticBuildRepository
from app.modules.agentic.build.service import AgenticBuildService
from app.modules.agentic.preview.repository import AgenticPreviewRepository
from app.modules.agentic.preview.service import AgenticPreviewService
from app.modules.agentic.repository import AgenticRepository
from app.modules.agentic.service import AgenticService
from app.modules.agentic.version_service import AgenticVersionService
from app.modules.genai.repository import GenAIRepository


async def run_worker(*, once: bool = False) -> None:
    await MongoDB.connect()
    try:
        if MongoDB.database is None:
            raise RuntimeError("MongoDB is unavailable.")
        database = MongoDB.database
        agentic = AgenticRepository(database)
        builds = AgenticBuildRepository(database)
        previews = AgenticPreviewRepository(database)
        await agentic.ensure_indexes()
        await builds.ensure_indexes()
        await previews.ensure_indexes()
        planning = AgenticService(agentic, GenAIRepository(database))
        versions = AgenticVersionService(agentic, planning)
        build_service = AgenticBuildService(builds, agentic, planning, versions)
        service = AgenticPreviewService(previews, builds, planning, build_service)
        worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:12]}"
        while True:
            await service.cleanup_expired()
            preview = await previews.claim_starting(worker_id)
            if preview:
                await service.launch_claimed(preview, worker_id)
            if once:
                return
            if not preview:
                await asyncio.sleep(settings.agentic_preview_cleanup_poll_seconds)
    finally:
        await MongoDB.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Agentic Docker preview worker.")
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(run_worker(once=arguments.once))
