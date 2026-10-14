"""
langgraph_agent.py — Stage 2: LangGraph ReAct Agent
=====================================================
Replaces the Stage 1 LCEL chain with a proper agent that can:
  - Decide which tool to call (or both) based on the question
  - Loop: call a tool, read the result, call another tool if needed
  - Remember the last conversation turns per session

ARCHITECTURE — the graph has two nodes connected in a cycle:

    START
      ↓
  ┌─────────────────────────────────────────────────────┐
  │  agent node (call_model)                            │
  │  • Reads conversation history                       │
  │  • Calls GPT-4o with tools bound                    │
  │  • LLM decides: call a tool? or answer directly?    │
  └──────────────┬──────────────────────────────────────┘
                 │
         ┌───────┴────────────┐
         │ conditional edge:  │
         │ tool_calls in msg? │
         └──┬─────────────────┘
            │ yes                   no
            ↓                       ↓
  ┌─────────────────┐            END
  │  tools node     │
  │  (ToolNode)     │
  │  • Runs the     │
  │    chosen tool  │
  │  • Appends      │
  │    result to    │
  │    messages     │
  └────────┬────────┘
           │ always loop back
           ↓
       agent node   (repeat until LLM stops calling tools)

STATE — a single dict that every node reads and writes:
    { "messages": [HumanMessage, AIMessage, ToolMessage, ...] }

The `add_messages` annotation means new messages are APPENDED,
not replaced. MemorySaver persists this state by thread_id (session_id)
so follow-up questions see the full conversation history.

MEMORY WINDOW — we trim to MEMORY_WINDOW messages before calling the LLM
to prevent context from growing unbounded. The full history stays in
MemorySaver; we just limit what the LLM sees per call.
"""

import logging
import re
import uuid
from typing import Annotated, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.rag.embedder import get_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# How many messages to pass to the LLM per call.
# Each Q&A turn with tool use produces ~3 messages:
#   HumanMessage → AIMessage(tool_call) → ToolMessage → AIMessage(answer)
# So MEMORY_WINDOW = 10 ≈ last 3–4 full conversation turns.
MEMORY_WINDOW = 10

SYSTEM_PROMPT = """\
You are an expert insurance policy analyst with access to two search tools:

  • search_policy     — searches all policy content for general information
  • search_exclusions — searches specifically for exclusions, limitations,
                        and things NOT covered by the policy

How to decide which tool(s) to call:
  - "What does my policy cover for X?"         → search_policy
  - "What's NOT covered / excluded for X?"     → search_exclusions
  - "Tell me everything about X" / broad query → call BOTH tools
  - Follow-up like "what about mold?" after a  → use same tool(s) as before
    water damage question

Rules for answering:
  - Base answers ONLY on retrieved policy text — never make things up
  - Quote the relevant policy section when possible
  - Explain insurance jargon in plain English after quoting
  - If retrieved text doesn't answer the question, say so clearly
"""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Keywords that signal an exclusion or limitation clause.
# Using stems (e.g. "exclud") catches "exclude", "excluded", "exclusion".
_EXCLUSION_KEYWORDS = [
    "exclud",
    "not covered",
    "we do not cover",
    "does not cover",
    "no coverage",
    "limitation",
    "we will not pay",
    "we do not insure",
    "this policy does not",
]


@tool
def search_policy(query: str) -> str:
    """
    Search all insurance policy content for information relevant to a question.

    Use this tool for questions about what the policy COVERS or for general
    policy terms, definitions, and procedures.

    Args:
        query: A search query describing what policy information is needed.

    Returns:
        The most semantically similar policy excerpts, with source labels.
    """
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=4)

    if not docs:
        return "No relevant policy content found for this query."

    parts = [
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    ]
    return "\n\n---\n\n".join(parts)


@tool
def search_exclusions(query: str) -> str:
    """
    Search specifically for exclusions, limitations, and things NOT covered
    by the insurance policy.

    Use this tool when the question is about what is excluded, not covered,
    or limited under the policy.

    Args:
        query: A search query describing the exclusion or limitation to find.

    Returns:
        Policy excerpts that contain exclusion or limitation language.
    """
    vector_store = get_vector_store()

    # Step 1: Retrieve a large candidate pool (k=20) via semantic similarity.
    # We cast a wide net because exclusion chunks may not be the *most* similar
    # to the query — they use negation language that shifts their embedding.
    all_docs = vector_store.similarity_search(query, k=20)

    # Step 2: Post-filter for chunks that contain exclusion language.
    filtered = [
        doc for doc in all_docs
        if any(kw in doc.page_content.lower() for kw in _EXCLUSION_KEYWORDS)
    ]

    # Step 3: Fall back to top unfiltered results if no exclusion chunks found.
    # This prevents a silent empty response when the policy uses unusual phrasing.
    docs_to_use = filtered[:4] if filtered else all_docs[:4]

    if not docs_to_use:
        return "No exclusion-related content found in the policy."

    parts = [
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs_to_use
    ]
    return "\n\n---\n\n".join(parts)


TOOLS = [search_policy, search_exclusions]

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    The state object passed between every node in the graph.

    `messages` uses the `add_messages` annotation from LangGraph, which means
    returning {"messages": [new_msg]} APPENDS new_msg instead of replacing
    the list. This is how conversation history accumulates automatically.
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# LLM — lazy-initialized so the module can be imported without OPENAI_API_KEY
# in the environment (the key is loaded from .env by main.py before first use)
# ---------------------------------------------------------------------------

_llm = None  # populated on first call to _get_llm()


def _get_llm():
    """Return the LLM singleton, creating it on first call."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(TOOLS)
    return _llm

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def call_model(state: AgentState) -> dict:
    """
    Agent node — the brain of the graph.

    Reads the conversation history, trims it to MEMORY_WINDOW messages so
    context doesn't grow unbounded, prepends the system prompt, then calls
    the LLM. The LLM response is either:
      • An AIMessage with tool_calls  → graph will route to the tools node
      • An AIMessage with just content → graph will end
    """
    messages = state["messages"]

    # Trim to last MEMORY_WINDOW messages.
    # We slice from the end rather than using trim_messages here for simplicity.
    # We then scan backwards to find the first HumanMessage so we never start
    # the context mid-tool-call (which confuses the LLM).
    if len(messages) > MEMORY_WINDOW:
        messages = messages[-MEMORY_WINDOW:]
        # Walk forward until we find a HumanMessage to start from
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                messages = messages[i:]
                break

    response = _get_llm().invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    logger.info(
        "LLM response: tool_calls=%s",
        [tc["name"] for tc in response.tool_calls] if response.tool_calls else "none",
    )
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Conditional edge — the graph's routing decision.

    Checks the last message in state:
      • If it has tool_calls → route to "tools" node (execute the tool)
      • Otherwise           → route to END (return answer to user)

    This is what makes the agent loop: tool result → back to agent → maybe
    another tool → back to agent → final answer → END.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Build the compiled graph (once at module load — not per request)
# ---------------------------------------------------------------------------

# MemorySaver is an in-process dict keyed by thread_id.
# Sessions persist as long as the server process is running.
# On server restart, sessions are lost — use SqliteSaver or RedisSaver
# in production for durable persistence.
_checkpointer = MemorySaver()


def _build_graph() -> object:
    """
    Construct the LangGraph StateGraph and compile it with the checkpointer.

    Graph structure:
        START → agent → (tools → agent)* → END
    """
    graph = StateGraph(AgentState)

    # --- Nodes ---
    graph.add_node("agent", call_model)
    # ToolNode automatically handles executing whatever tool the LLM called
    # and appending the result as a ToolMessage to state["messages"].
    graph.add_node("tools", ToolNode(TOOLS))

    # --- Entry point ---
    graph.set_entry_point("agent")

    # --- Edges ---
    # Conditional edge from agent: go to tools or END
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    # Fixed edge from tools: always loop back to agent
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=_checkpointer)


_agent = _build_graph()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(question: str, session_id: Optional[str] = None) -> dict:
    """
    Answer an insurance policy question using the LangGraph agent.

    If session_id is provided and matches an existing session, the agent
    sees the full conversation history for that session (up to MEMORY_WINDOW
    messages in the LLM context). This enables follow-up questions like:
        1st: "Does my policy cover water damage?"
        2nd: "What about mold?" ← agent knows you mean mold coverage

    Args:
        question:   The user's question in plain English.
        session_id: Optional session ID string. Pass None to start a new
                    session (a UUID will be created and returned).

    Returns:
        Dict with:
            "answer"     — GPT-4o's response
            "session_id" — the session ID (new or existing)
            "sources"    — list of source PDF filenames cited by the tools
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info("New session created: %s", session_id)
    else:
        logger.info("Resuming session: %s", session_id)

    config = {"configurable": {"thread_id": session_id}}

    result = _agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    # The last message in the graph's output is always the final AI answer
    answer = result["messages"][-1].content

    # Extract source filenames from [Source: filename] labels we embed in
    # tool output strings. Using a set deduplicates repeated source citations.
    sources: set[str] = set()
    for msg in result["messages"]:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            found = re.findall(r"\[Source:\s*([^\]]+)\]", msg.content)
            sources.update(s.strip() for s in found)

    logger.info("Answered (session=%s, sources=%s)", session_id, sorted(sources))
    return {
        "answer": answer,
        "session_id": session_id,
        "sources": sorted(sources),
    }
