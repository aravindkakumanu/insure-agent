"""
test_tools.py — Artifact-returning retrieval tools
===================================================
Offline: a fake vector store stands in for ChromaDB + OpenAI embeddings.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents import tools
from tests.fakes import make_doc, store_with


def run_tool(tool, query: str = "water damage") -> ToolMessage:
    """Invoke a tool the way ToolNode does, so the artifact comes back too."""
    return tool.invoke({"name": tool.name, "args": {"query": query}, "id": "t1", "type": "tool_call"})


@pytest.fixture
def use_store(monkeypatch: pytest.MonkeyPatch):
    def _use(*docs, **kwargs):
        store = store_with(*docs, **kwargs)
        monkeypatch.setattr(tools, "get_vector_store", lambda: store)
        return store

    return _use


# ---------------------------------------------------------------------------
# Exclusion language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Losses caused by freezing are excluded.",
        "SECTION I - EXCLUSIONS",
        "We do not cover loss caused by flood.",
        "This exclusion applies regardless of cause.",
    ],
)
def test_exclusion_language_detected(text: str) -> None:
    assert tools.has_exclusion_language(text)


@pytest.mark.parametrize(
    "text",
    [
        "We will cover loss caused by water or steam not otherwise excluded.",
        "Your dwelling is protected against fire. NOT EXCLUDED perils are covered.",
        "Your dwelling is protected against fire.",
    ],
)
def test_coverage_wording_is_not_exclusion_language(text: str) -> None:
    """Regression: the burst-pipe coverage grant says "not otherwise excluded"."""
    assert not tools.has_exclusion_language(text)


def test_real_exclusion_next_to_not_otherwise_excluded_still_counts() -> None:
    text = "Water not otherwise excluded is covered, but flood damage is excluded."
    assert tools.has_exclusion_language(text)


# ---------------------------------------------------------------------------
# Evidence records
# ---------------------------------------------------------------------------


def test_evidence_carries_metadata_distance_and_id(use_store) -> None:
    use_store(
        make_doc(
            "We cover fire.",
            chunk_id="abc",
            source="x.pdf",
            page=8,
            page_end=9,
            section="Section I—Your Property",
            subsection="Losses We Cover Under Coverage C",
            kind="coverage",
        ),
        distance=0.25,
    )

    (evidence,) = run_tool(tools.search_policy).artifact

    assert evidence["chunk_id"] == "abc"
    assert (evidence["source"], evidence["page"], evidence["page_end"]) == ("x.pdf", 8, 9)
    assert evidence["subsection"] == "Losses We Cover Under Coverage C"
    assert evidence["kind"] == "coverage"
    assert evidence["distance"] == pytest.approx(0.25)
    assert evidence["exclusion_language"] is False


def test_chunks_ingested_before_metadata_existed_still_work(use_store) -> None:
    """Old ingests only had {"source": ...}: every other field must default."""
    from langchain_core.documents import Document

    use_store(Document(id="old", page_content="We do not cover flood.", metadata={"source": "x.pdf"}))

    (evidence,) = run_tool(tools.search_exclusions).artifact

    assert (evidence["page"], evidence["page_end"], evidence["section"], evidence["kind"]) == (0, 0, "", "")
    assert evidence["exclusion_language"] is True  # the keyword test alone decides


def test_chunk_without_id_gets_a_content_hash(use_store) -> None:
    from langchain_core.documents import Document

    use_store(Document(page_content="We cover fire.", metadata={"source": "x.pdf"}))
    (evidence,) = run_tool(tools.search_policy).artifact
    assert len(evidence["chunk_id"]) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# search_policy
# ---------------------------------------------------------------------------


def test_search_policy_returns_text_and_matching_artifact(use_store) -> None:
    use_store(make_doc("We cover fire.", page=8, subsection="Losses We Cover"))

    message = run_tool(tools.search_policy)

    assert "We cover fire." in message.content
    assert "[Source: p.pdf, page 8, Losses We Cover]" in message.content
    assert len(message.artifact) == 1


def test_label_shows_a_page_range(use_store) -> None:
    use_store(make_doc("text", page=13, page_end=15, section="Section I—Your Property"))
    assert "[Source: p.pdf, pages 13-15, Section I—Your Property]" in run_tool(tools.search_policy).content


def test_label_omits_unknown_page_and_section(use_store) -> None:
    from langchain_core.documents import Document

    use_store(Document(id="x", page_content="text", metadata={"source": "p.pdf"}))
    assert "[Source: p.pdf]" in run_tool(tools.search_policy).content


def test_search_policy_empty(use_store) -> None:
    use_store()
    message = run_tool(tools.search_policy)
    assert message.content == "No relevant policy content found for this query."
    assert message.artifact == []


def test_search_policy_asks_for_four_chunks(use_store) -> None:
    store = use_store(make_doc("a"))
    run_tool(tools.search_policy, "burst pipe")
    store.similarity_search_with_score.assert_called_once_with("burst pipe", k=tools.POLICY_TOP)


# ---------------------------------------------------------------------------
# search_exclusions
# ---------------------------------------------------------------------------


def test_exclusions_section_chunks_come_before_keyword_only_hits(use_store) -> None:
    keyword_only = make_doc("Freezing losses are excluded.", chunk_id="kw", kind="coverage")
    in_section = make_doc("Land.", chunk_id="sec", kind="exclusions")  # no keyword at all
    use_store(keyword_only, in_section)  # keyword_only is the CLOSER match

    artifact = run_tool(tools.search_exclusions).artifact

    assert [e["chunk_id"] for e in artifact] == ["sec", "kw"]
    assert all(e["exclusion_language"] for e in artifact)


def test_chunk_overlapping_an_exclusions_region_counts(use_store) -> None:
    straddler = make_doc("Each person who sustains injury...", chunk_id="s", kind="coverage", has_exclusions=True)
    use_store(straddler)
    (evidence,) = run_tool(tools.search_exclusions).artifact
    assert evidence["exclusion_language"] is True


def test_coverage_grant_is_not_returned_as_an_exclusion(use_store) -> None:
    """The burst-pipe regression: a grant saying 'not otherwise excluded' must not
    be treated as an exclusion chunk. With nothing else, it comes back only as an
    unflagged fallback."""
    grant = make_doc(
        "we cover the direct physical damage caused by the water or steam. If loss is "
        "caused by water or steam not otherwise excluded, we will cover tearing out.",
        kind="coverage",
    )
    use_store(grant)

    (evidence,) = run_tool(tools.search_exclusions).artifact

    assert evidence["exclusion_language"] is False


def test_exclusions_fall_back_to_closest_unflagged_chunks(use_store) -> None:
    docs = [make_doc(f"Your dwelling is protected {i}.", chunk_id=f"d{i}") for i in range(10)]
    use_store(*docs)

    artifact = run_tool(tools.search_exclusions).artifact

    assert [e["chunk_id"] for e in artifact] == ["d0", "d1", "d2", "d3"]  # POLICY_TOP
    assert not any(e["exclusion_language"] for e in artifact)


def test_exclusions_return_at_most_exclusion_top(use_store) -> None:
    use_store(*[make_doc(f"Loss {i} is excluded.", chunk_id=f"e{i}") for i in range(12)])
    assert len(run_tool(tools.search_exclusions).artifact) == tools.EXCLUSION_TOP


def test_exclusions_search_uses_a_wide_pool(use_store) -> None:
    store = use_store(make_doc("excluded"))
    run_tool(tools.search_exclusions, "flood")
    store.similarity_search_with_score.assert_called_once_with("flood", k=tools.EXCLUSION_POOL)


def test_exclusions_empty(use_store) -> None:
    use_store()
    message = run_tool(tools.search_exclusions)
    assert message.content == "No exclusion-related content found in the policy."
    assert message.artifact == []


# ---------------------------------------------------------------------------
# Through the real ToolNode (what the specialists use)
# ---------------------------------------------------------------------------


def test_toolnode_keeps_the_artifact_on_the_tool_message(use_store) -> None:
    use_store(make_doc("We do not cover flood.", chunk_id="f1", kind="exclusions"))
    call = AIMessage(
        content="",
        tool_calls=[{"name": "search_exclusions", "args": {"query": "flood"}, "id": "1"}],
    )

    # ToolNode needs graph context, so run it as the only node of a tiny graph
    # - exactly how the specialists use it.
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([tools.search_exclusions]))
    graph.set_entry_point("tools")
    graph.add_edge("tools", END)
    out = graph.compile().invoke({"messages": [call]})

    message = out["messages"][-1]
    assert message.name == "search_exclusions"
    assert [e["chunk_id"] for e in message.artifact] == ["f1"]


def test_tool_names_match_the_stage_two_tools() -> None:
    import src.agents.langgraph_agent as stage2

    assert tools.search_policy.name == stage2.search_policy.name
    assert tools.search_exclusions.name == stage2.search_exclusions.name
    assert tools.search_policy is not stage2.search_policy  # Stage 2 file left alone
