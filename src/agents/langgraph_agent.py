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
    trim_messages,
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

# Doom-loop guard: max agent-node visits per question. A legitimate question
# needs only a handful; hitting this means the agent is stuck re-calling tools.
MAX_ITERATIONS = 25

# Hard backstop below LangGraph's own limit. LangGraph counts every node run
# (agent AND tools) as a step, so one agent visit with a tool call = 2 steps.
# Set explicitly because the default varies by version (10007 in langgraph
# 1.2.x, 25 in older releases); it must stay above 2 * MAX_ITERATIONS or
# GraphRecursionError would pre-empt our friendlier fallback.
RECURSION_LIMIT = 2 * MAX_ITERATIONS + 5

FALLBACK_ANSWER = (
    "I wasn't able to complete this request - I kept searching the policy "
    "without reaching an answer. Please try rephrasing your question or "
    "asking about one specific topic at a time."
)

SYSTEM_PROMPT = """\
You are an expert insurance policy analyst with access to two search tools:

  • search_policy     — searches all policy content for general information,
                        coverage grants, definitions, and procedures
  • search_exclusions — searches specifically for exclusions, limitations,
                        and things NOT covered by the policy

DEFAULT BEHAVIOR — call BOTH tools for any question about a specific topic:
  When a user asks about water damage, mold, theft, fire, liability, or any
  other specific subject, ALWAYS call both search_policy AND search_exclusions.
  Then synthesize the results into a complete answer that covers:
    1. What IS covered (from search_policy)
    2. What is NOT covered or excluded (from search_exclusions)
    3. Any dollar limits or conditions mentioned in either result

EXCEPTIONS — call only one tool when the question is explicitly one-sided:
  - "What does my policy cover for X?" (no mention of exclusions) → search_policy only
  - "What's excluded for X?" / "What's not covered?" → search_exclusions only

Follow-up questions: if the user asks a follow-up like "what about mold?"
after a water damage question, apply the same default — call both tools
in the context of the previous topic.

Rules for answering:
  - Base answers ONLY on retrieved policy text — never make things up
  - Quote the relevant policy section, then explain it in plain English
  - Clearly separate what IS covered from what IS NOT covered
  - If retrieved text doesn't answer part of the question, say so explicitly
  - Never leave the user wondering "but what's the catch?" — always include limits
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
    # Agent-node visits for the current question. No reducer, so the value
    # passed in at invoke time OVERWRITES the checkpointed one - answer_question
    # sends 0 on every new question so the count never leaks across turns.
    step_counter: int


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

    Doom-loop guard: every visit increments step_counter. Past MAX_ITERATIONS
    we return a fallback answer WITHOUT calling the LLM; it has no tool_calls,
    so should_continue routes to END.
    """
    step = state.get("step_counter", 0) + 1
    if step > MAX_ITERATIONS:
        logger.warning("Doom-loop guard tripped at step %d; returning fallback", step)
        return {"messages": [AIMessage(content=FALLBACK_ANSWER)], "step_counter": step}

    # Keep the last MEMORY_WINDOW messages (token_counter=len counts each
    # message as 1). start_on="human" guarantees the window never begins
    # mid-tool-call, which OpenAI rejects (a ToolMessage with no preceding
    # tool_call).
    all_messages = state["messages"]
    messages = trim_messages(
        all_messages,
        max_tokens=MEMORY_WINDOW,
        token_counter=len,
        strategy="last",
        start_on="human",
    )
    if not messages:
        # The current turn alone exceeds the window (many tool round trips), so
        # no HumanMessage fit. Keep the whole current turn rather than lose the
        # question.
        last_human = max(
            i for i, m in enumerate(all_messages) if isinstance(m, HumanMessage)
        )
        messages = all_messages[last_human:]

    response = _get_llm().invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    logger.info(
        "LLM response: tool_calls=%s",
        [tc["name"] for tc in response.tool_calls] if response.tool_calls else "none",
    )
    return {"messages": [response], "step_counter": step}


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

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    result = _agent.invoke(
        {"messages": [HumanMessage(content=question)], "step_counter": 0},
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
