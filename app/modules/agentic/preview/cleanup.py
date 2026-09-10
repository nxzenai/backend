from __future__ import annotations

import argparse
import asyncio

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


async def run_cleanup(*, once: bool = False) -> None:
    await MongoDB.connect()
    try:
        if MongoDB.database is None:
            raise RuntimeError("MongoDB is unavailable.")
        database = MongoDB.database
        agentic = AgenticRepository(database)
        builds = AgenticBuildRepository(database)
        previews = AgenticPreviewRepository(database)
        await previews.ensure_indexes()
        planning = AgenticService(agentic, GenAIRepository(database))
        versions = AgenticVersionService(agentic, planning)
        build_service = AgenticBuildService(builds, agentic, planning, versions)
        service = AgenticPreviewService(previews, builds, planning, build_service)
        while True:
            await service.cleanup_expired()
            if once:
                return
            await asyncio.sleep(settings.agentic_preview_cleanup_poll_seconds)
    finally:
        await MongoDB.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean expired Agentic Docker previews.")
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(run_cleanup(once=arguments.once))
