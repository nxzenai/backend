import pytest

from app.modules.genai.repository import GenAIRepository


class Collection:
    def __init__(self, rows):
        self.rows = rows

    def find(self, query):
        def matches(row):
            return all(row.get(key) in value["$in"] if isinstance(value, dict)
                       else row.get(key) == value for key, value in query.items())
        return Cursor([row for row in self.rows if matches(row)])


class Cursor:
    def __init__(self, rows):
        self.rows = rows

    def sort(self, key, direction):
        self.rows.sort(key=lambda row: row[key], reverse=direction < 0)
        return self

    def limit(self, count):
        self.rows = self.rows[:count]
        return self

    async def to_list(self, length):
        return self.rows[:length]


def repository():
    repo = object.__new__(GenAIRepository)
    repo.attachments = Collection([
        {"_id": aid, "owner_id": owner, "chunk_count": 50}
        for aid, owner in [("pdf", "user"), ("docx", "user"), ("private", "other")]
    ])
    repo.attachment_chunks = Collection([
        {"_id": f"{aid}-{index}", "owner_id": owner, "attachment_id": aid,
         "filename": aid, "chunk_index": index,
         "content": "Helium" if index == 17 else "Astronomy research findings."}
        for aid, owner in [("pdf", "user"), ("docx", "user"), ("private", "other")]
        for index in reversed(range(50))
    ])
    return repo


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "summarize this document", "what is this paper about",
    "what is this document about?", "give me the summary of the document",
    "Please provide an overview of the attached document.",
])
async def test_summary_covers_document_in_original_order(query):
    chunks = await repository().search_attachment_chunks("user", ["pdf"], query)
    indices = [chunk["chunk_index"] for chunk in chunks]
    assert len(chunks) == 8
    assert indices == sorted(set(indices))
    assert indices[0] == 0 and indices[-1] == 49
    assert any(20 <= index <= 30 for index in indices)
    assert all("owner_id" not in chunk and "_id" not in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_multiple_documents_share_budget_and_keep_order():
    chunks = await repository().search_attachment_chunks("user", ["pdf", "docx"], "summarize these documents")
    assert len(chunks) == 8
    for aid in ["pdf", "docx"]:
        indices = [chunk["chunk_index"] for chunk in chunks if chunk["attachment_id"] == aid]
        assert indices == [0, 16, 33, 49]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["What does the document say about helium?", "Summarize helium in this document"])
async def test_specific_questions_still_use_relevance(query):
    chunks = await repository().search_attachment_chunks("user", ["pdf"], query)
    assert [chunk["chunk_index"] for chunk in chunks] == [17]
    assert await repository().search_attachment_chunks("user", ["pdf"], "What about xenon?") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [["missing"], ["private"], ["pdf", "private"]])
async def test_invalid_or_unowned_selection_has_no_evidence(ids):
    repo = repository()
    assert await repo.search_attachment_chunks("user", ids, "summarize this document") == []
    # Existing chat binding rejects the selection before any summary retrieval.
    assert await repo.attach_files_to_conversation("user", ids, "conversation", None) == []


@pytest.mark.asyncio
async def test_empty_document_retains_evidence_guard():
    repo = repository()
    repo.attachment_chunks = Collection([])
    assert await repo.search_attachment_chunks("user", ["pdf"], "summarize this document") == []
