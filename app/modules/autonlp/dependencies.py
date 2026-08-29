from __future__ import annotations

from app.modules.autonlp.service import AutoNLPService


def get_autonlp_service() -> AutoNLPService:
    return AutoNLPService()


__all__ = ["get_autonlp_service"]
