"""
test_sessions.py — Durable sessions (SQLite)
=============================================
Fully offline: fake LLMs and a fake vector store, with the session database in a
temporary file. A "restart" is simulated the way a real one works: every in-memory
object (checkpointer connections, compiled graph, router maps) is thrown away and
rebuilt from the same file.
"""

import os
import stat
import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents import coverage_agent, exclusions_agent, router, sessions, tools
from src.agents.sessions import SessionMap, create_checkpointer, thread_id
from tests.fakes import FakeRouterLLM, OneToolThenAnswerLLM, make_doc, store_with


class RecordingLLM:
    """Answers directly (no tools) and keeps every message list it was given."""

    def __init__(self) -> None:
        self.seen: list[list] = []

    def invoke(self, messages: list) -> AIMessage:
        self.seen.append(list(messages))
        return AIMessage(content=f"answer {len(self.seen)}")

    def humans(self, call: int = -1) -> list[str]:
        return [m.content for m in self.seen[call] if isinstance(m, HumanMessage)]


@pytest.fixture
def db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point sessions at a real file (conftest defaults them to :memory:)."""
    path = str(tmp_path / "sessions.db")
    monkeypatch.setenv("SESSIONS_DB", path)
    return path


def restart(module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebuild a specialist's saver and graph from the file, as a new process would."""
    monkeypatch.setattr(module, "_checkpointer", create_checkpointer())
    monkeypatch.setattr(module, "_agent", module._build_graph())


# ---------------------------------------------------------------------------
# Configuration and the file itself
# ---------------------------------------------------------------------------


def test_default_location_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESSIONS_DB", raising=False)
    assert sessions.db_path() == "data/sessions.db"
    monkeypatch.setenv("SESSIONS_DB", "/somewhere/else.db")
    assert sessions.db_path() == "/somewhere/else.db"


def test_the_test_suite_never_touches_the_real_database() -> None:
    assert os.environ["SESSIONS_DB"] == ":memory:"


def test_the_database_is_created_owner_only_and_its_folder_too(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested" / "dir" / "sessions.db"
    monkeypatch.setenv("SESSIONS_DB", str(path))

    SessionMap("intent")["s"] = "coverage"

    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


# ---------------------------------------------------------------------------
# SessionMap (the router's state)
# ---------------------------------------------------------------------------


def test_session_map_behaves_like_the_dict_it_replaces(db: str) -> None:
    m = SessionMap("intent")
    assert m.get("s") is None
    assert m.get("s", "fallback") == "fallback"
    assert "s" not in m

    m["s"] = "coverage"
    m["s"] = "claims"  # overwrite

    assert m.get("s") == "claims"
    assert "s" in m


def test_session_map_survives_a_restart(db: str) -> None:
    SessionMap("question")["s"] = "Is mold covered?"

    assert SessionMap("question").get("s") == "Is mold covered?"  # new instance


def test_clear_only_clears_its_own_kind(db: str) -> None:
    intent, question = SessionMap("intent"), SessionMap("question")
    intent["s"], question["s"] = "coverage", "q?"

    intent.clear()

    assert intent.get("s") is None
    assert question.get("s") == "q?"


# ---------------------------------------------------------------------------
# Specialist history
# ---------------------------------------------------------------------------


def test_thread_ids_are_namespaced_by_agent() -> None:
    assert thread_id("coverage", "abc") != thread_id("exclusions", "abc")
    assert thread_id("coverage", "abc").endswith("abc")


def test_history_survives_a_restart(db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "get_vector_store", lambda: store_with(make_doc("We cover fire.")))
    restart(coverage_agent, monkeypatch)

    first = OneToolThenAnswerLLM("search_policy", answer="fire is covered")
    monkeypatch.setattr(coverage_agent, "_get_llm", lambda: first)
    coverage_agent.answer_question("Is fire covered?", "durable")

    restart(coverage_agent, monkeypatch)  # the "new process"
    after = RecordingLLM()
    monkeypatch.setattr(coverage_agent, "_get_llm", lambda: after)
    coverage_agent.answer_question("And smoke?", "durable")

    assert after.humans() == ["Is fire covered?", "And smoke?"]
    # The earlier turn's tool result came back WITH its evidence artifact.
    tool_messages = [m for m in after.seen[0] if isinstance(m, ToolMessage)]
    assert tool_messages and tool_messages[0].artifact


def test_agents_sharing_a_file_do_not_share_history(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    restart(coverage_agent, monkeypatch)
    restart(exclusions_agent, monkeypatch)
    coverage_llm, exclusions_llm = RecordingLLM(), RecordingLLM()
    monkeypatch.setattr(coverage_agent, "_get_llm", lambda: coverage_llm)
    monkeypatch.setattr(exclusions_agent, "_get_llm", lambda: exclusions_llm)

    coverage_agent.answer_question("COVERAGE question", "same-session")
    exclusions_agent.answer_question("EXCLUSIONS question", "same-session")

    assert exclusions_llm.humans() == ["EXCLUSIONS question"]
    assert coverage_llm.humans() == ["COVERAGE question"]


def test_a_new_session_starts_empty(db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    restart(coverage_agent, monkeypatch)
    llm = RecordingLLM()
    monkeypatch.setattr(coverage_agent, "_get_llm", lambda: llm)

    coverage_agent.answer_question("first", "session-one")
    coverage_agent.answer_question("second", "session-two")

    assert llm.humans() == ["second"]


def test_concurrent_writers_on_one_file_do_not_hit_database_is_locked(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coverage fan-out has two specialists writing the same file at once."""
    restart(coverage_agent, monkeypatch)
    restart(exclusions_agent, monkeypatch)
    llms = {coverage_agent: RecordingLLM(), exclusions_agent: RecordingLLM()}
    for module, llm in llms.items():
        monkeypatch.setattr(module, "_get_llm", lambda llm=llm: llm)

    errors: list[BaseException] = []

    def hammer(module, tag: str) -> None:
        try:
            for i in range(25):
                module.answer_question(f"{tag} {i}", f"{tag}-session")
        except BaseException as exc:  # noqa: BLE001 - the test reports any failure
            errors.append(exc)

    threads = [
        threading.Thread(target=hammer, args=(coverage_agent, "c")),
        threading.Thread(target=hammer, args=(exclusions_agent, "e")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # 25 turns each, all kept: the last call saw the whole (trimmed) history.
    assert llms[coverage_agent].humans()[-1] == "c 24"


# ---------------------------------------------------------------------------
# Router follow-ups
# ---------------------------------------------------------------------------


def reconnect_router_maps() -> None:
    """Drop the router maps' connections so the next use reopens the file (a restart)."""
    for state in (router._session_intent, router._session_question):
        state._conn = None


def test_the_router_keeps_its_state_in_the_session_database() -> None:
    assert isinstance(router._session_intent, SessionMap)
    assert isinstance(router._session_question, SessionMap)


def test_a_bare_follow_up_still_routes_after_a_restart(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRouterLLM("exclusions", "COV?", "What exclusions apply to mold?")
    monkeypatch.setattr(router, "_get_router_llm", lambda: fake)
    monkeypatch.setattr(exclusions_agent, "_get_llm", lambda: RecordingLLM())
    reconnect_router_maps()  # the router's own maps, now pointed at the file

    router.route_and_answer("What about mold?", session_id="follow-up")
    reconnect_router_maps()  # restart: only the file survives
    router.route_and_answer("and fungus?", session_id="follow-up")

    prompt = fake.seen[-1][-1].content
    assert "previous_intent: exclusions" in prompt
    assert "previous_question: What exclusions apply to mold?" in prompt
    reconnect_router_maps()  # leave the maps on :memory: for the next test
