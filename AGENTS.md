## Learned User Preferences

- User wants every AI decision explained step-by-step as code is written — treat this as a teaching project.
- User runs commands in their own terminal; provide exact copy-paste shell commands rather than executing on their behalf.
- When introducing a new concept (tokens vs characters, semantic vs fixed chunking), always explain the distinction before making code changes.

## Learned Workspace Facts

- `langchain-experimental` is installed and required for `SemanticChunker` (`from langchain_experimental.text_splitter import SemanticChunker`).
- `RecursiveCharacterTextSplitter` measures **characters** by default; use `.from_tiktoken_encoder()` for true token-based splitting.
- `CHUNK_STRATEGY` flag lives at the top of `src/rag/loader.py`; valid values: `"semantic"` (default, production) and `"fixed_token"` (baseline/debug).
- After any `CHUNK_STRATEGY` change, always wipe and re-ingest: `rm -rf data/chroma_db` then POST to `/ingest`.
- Active PDF for testing: `data/policies/apc220-ho-policy-jacket.pdf` (26 pages); semantic ingestion produces ~69 chunks, fixed-token 512-tok produces ~61 chunks.
- Semantic chunking is confirmed superior for this domain: it keeps coverage-grant and tear-out clauses in the same chunk, which fixed-token splits across a boundary.
- `langchain-text-splitters` package must be installed separately; import as `from langchain_text_splitters import RecursiveCharacterTextSplitter` (not `langchain.text_splitter`).
