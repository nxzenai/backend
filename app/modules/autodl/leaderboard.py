"""
NxZen AI Studio

AutoDL Leaderboard
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class DLLeaderboardEntry:
    rank: int
    model_name: str
    score: float
    training_time: float

class DLLeaderboardEngine:
    def generate(self, training_results: list[Any]) -> list[DLLeaderboardEntry]:
        entries = []
        for res in training_results:
            entries.append(DLLeaderboardEntry(rank=1, model_name=res.model_name, score=res.accuracy, training_time=res.training_time))
        entries.sort(key=lambda x: x.score, reverse=True)
        for i, entry in enumerate(entries, start=1):
            entry.rank = i
        return entries