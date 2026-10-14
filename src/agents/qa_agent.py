"""
qa_agent.py — Stage 1 LCEL RAG Chain (kept for comparison)
=============================================================
⚠️  STAGE 1 ONLY — Not used by the /ask endpoint as of Stage 2.
    The production agent is now src/agents/langgraph_agent.py.
    This file is preserved so you can compare the simple LCEL
    pipeline against the full LangGraph agent side-by-side.

HOW IT WORKS (RAG in one sentence):
  Retrieve relevant chunks → Inject them into a prompt → Ask the LLM to answer.

LCEL (LangChain Expression Language):
In LangChain 1.x, the old RetrievalQA chain was removed. The modern way is
to compose steps using the pipe operator (|), called LCEL. It reads like a
Unix pipeline and makes data flow explicit:

    retriever | format_docs | prompt | llm | output_parser

Each step receives the output of the previous step.
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.rag.retriever import get_retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template — tells the LLM its role and how to use the context
# ---------------------------------------------------------------------------
PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """\
You are an expert insurance policy analyst. Your job is to answer questions
about insurance policies clearly and accurately, based ONLY on the provided
policy document excerpts.

Rules:
- If the answer is in the context, quote the relevant section and explain it in plain English.
- If the answer is NOT in the context, say: "I couldn't find that information in the provided policy documents."
- Never make up information. Accuracy is critical in insurance.
- Be concise but complete.

Context (policy document excerpts):
{context}""",
    ),
    ("human", "{question}"),
])


def _format_docs(docs: list) -> str:
    """
    Join retrieved Document chunks into a single string for the prompt.

    Each chunk is separated by a divider so the LLM can tell where one
    excerpt ends and the next begins.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_qa_chain():
    """
    Build and return a LangChain LCEL RAG chain.

    The chain (read left to right via the | pipe operator):
      1. RunnablePassthrough — passes the question straight through unchanged
      2. retriever          — finds top-k relevant chunks from ChromaDB
      3. _format_docs       — joins chunks into one context string
      4. PROMPT             — fills in {context} and {question}
      5. llm                — GPT-4o generates the answer
      6. StrOutputParser    — extracts the plain string from the LLM response

    Returns a runnable you call with: chain.invoke({"question": "..."})
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,   # 0 = deterministic; we want factual accuracy, not creativity
    )
    retriever = get_retriever()

    # LCEL chain — each | passes output to the next step
    chain = (
        {
            # Run retriever on the question, format results as context string
            "context": retriever | _format_docs,
            # Pass the question through unchanged so the prompt can use it
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info("QA chain ready (model=gpt-4o, LCEL)")
    return chain


def answer_question(question: str) -> dict:
    """
    Answer a single insurance policy question.

    Args:
        question: Plain-English question about a policy.

    Returns:
        Dict with keys:
            - "answer" (str): The LLM's response.
            - "sources" (list[str]): Source document names used.
    """
    # Build retriever separately so we can capture source docs
    retriever = get_retriever()
    source_docs = retriever.invoke(question)
    sources = list({doc.metadata.get("source", "unknown") for doc in source_docs})

    # Run the full chain for the answer
    chain = build_qa_chain()
    answer = chain.invoke(question)

    logger.info("Answered question with %d source(s): %s", len(sources), sources)
    return {
        "answer": answer,
        "sources": sources,
    }
