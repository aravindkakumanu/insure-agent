"""
claims_agent.py — Stage 3: Claims specialist
Explains the claims process and procedures. Tool subset: search_policy
and search_exclusions (there is no separate claims section in the vector
store, so claims content is found with claims-focused queries).

Same pattern as langgraph_agent.py (the Stage 2 reference; see its docstring for
the graph diagram): its own compiled graph, its own MemorySaver, its own
answer_question(), and the same MAX_ITERATIONS doom-loop guard. The only things
that differ between the three specialists are the system prompt and the tool
subset.
"""

import logging
import uuid
from typing import Annotated, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.agents.tools import search_exclusions, search_policy
from src.agents.results import (
    AgentResult,
    assess_confidence,
    current_turn,
    extract_evidence,
    extract_queries,
    find_unverified_quotes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENT_NAME = "claims"
TOOLS = [search_policy, search_exclusions]

MEMORY_WINDOW = 10
MAX_ITERATIONS = 25
RECURSION_LIMIT = 2 * MAX_ITERATIONS + 5

FALLBACK_ANSWER = (
    "I wasn't able to complete this request - I kept searching the policy "
    "without reaching an answer. Please try rephrasing your question or "
    "asking about one specific topic at a time."
)

SYSTEM_PROMPT = """\
You are the Claims specialist in an insurance policy Q&A system. Your job is
to explain how to make a claim and what the policy requires of the insured.

You have two tools:
  - search_policy: searches all policy content
  - search_exclusions: searches exclusions, limitations, and conditions

How to answer:
  - The policy has no dedicated claims section. Search with claims-focused
    wording such as "duties after loss", "notice of claim", "proof of loss",
    "appraisal", "payment of loss", and "suit against us"
  - Cover the process in order: what to do after a loss, who to notify and by
    when, what documents or proof are required, how the claim is settled, and
    any time limits
  - Use search_exclusions to check for conditions that could bar or reduce a
    claim, such as late notice, and mention them
  - Quote the relevant policy section, then explain it in plain English
  - Base answers ONLY on retrieved policy text; never make things up
  - Use quotation marks ONLY for text copied word for word from the retrieved
    policy text. Use "..." where you skip words. If you are summarizing or
    rewording, write it without quotation marks
  - If the retrieved text doesn't answer part of the question, say so clearly
"""

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Messages accumulate (add_messages); step_counter is reset per question."""

    messages: Annotated[list[BaseMessage], add_messages]
    step_counter: int


_llm = None  # lazy: ChatOpenAI needs OPENAI_API_KEY at construction time


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
    """Agent node: guard, trim history, call the LLM with this agent's tools."""
    step = state.get("step_counter", 0) + 1
    if step > MAX_ITERATIONS:
        logger.warning("[%s] doom-loop guard tripped at step %d", AGENT_NAME, step)
        return {"messages": [AIMessage(content=FALLBACK_ANSWER)], "step_counter": step}

    all_messages = state["messages"]
    messages = trim_messages(
        all_messages,
        max_tokens=MEMORY_WINDOW,
        token_counter=len,
        strategy="last",
        start_on="human",
    )
    if not messages:
        # The current turn alone exceeds the window: keep all of it.
        last_human = max(
            i for i, m in enumerate(all_messages) if isinstance(m, HumanMessage)
        )
        messages = all_messages[last_human:]

    response = _get_llm().invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    logger.info(
        "[%s] LLM response: tool_calls=%s",
        AGENT_NAME,
        [tc["name"] for tc in response.tool_calls] if response.tool_calls else "none",
    )
    return {"messages": [response], "step_counter": step}


def should_continue(state: AgentState) -> str:
    """Route to the tools node if the LLM asked for a tool, otherwise finish."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph — compiled once at import, with this agent's own MemorySaver
# ---------------------------------------------------------------------------

_checkpointer = MemorySaver()


def _build_graph() -> object:
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=_checkpointer)


_agent = _build_graph()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def answer_question(question: str, session_id: Optional[str] = None) -> AgentResult:
    """
    Answer a question with this specialist.

    Returns an AgentResult. Evidence, sources, confidence and the quote check
    use THIS turn's tool results only, so earlier questions in the session
    don't leak in.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info("[%s] new session: %s", AGENT_NAME, session_id)

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    result = _agent.invoke(
        {"messages": [HumanMessage(content=question)], "step_counter": 0},
        config=config,
    )

    messages = result["messages"]
    turn = current_turn(messages)
    answer = messages[-1].content
    fell_back = result["step_counter"] > MAX_ITERATIONS

    evidence = extract_evidence(turn)
    unverified = () if fell_back else find_unverified_quotes(answer, evidence)
    confidence = assess_confidence(turn, fell_back, unverified)
    sources = sorted({e["source"] for e in evidence})
    queries = extract_queries(turn)

    logger.info(
        "[%s] answered (session=%s, sources=%s, confidence=%s, unverified_quotes=%d, queries=%s)",
        AGENT_NAME,
        session_id,
        sources,
        confidence,
        len(unverified),
        list(queries),
    )
    return AgentResult(
        answer=answer,
        sources=sources,
        agent_name=AGENT_NAME,
        confidence=confidence,
        session_id=session_id,
        evidence=tuple(evidence),
        unverified_quotes=unverified,
        queries=queries,
    )
