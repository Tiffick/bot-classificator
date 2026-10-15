# Professional knowledge runtime assets

This directory contains versioned, reviewable source catalogs for the local
Knowledge/Retrieval path. Built search structures are derived data; the JSON
catalogs and manifests remain authoritative.

The first weight-management release is explicitly a pilot author synthesis,
not expert-validated clinical knowledge. Human Experience admission means only
that a card passed the declared research-schema and dialogue-safety checks. It
does not turn a card into evidence about a user.

Retrieval is semantic-only. It uses the local ONNX model
`intfloat/multilingual-e5-small` and a versioned
dense index. It has no FTS, BM25, keywords, regular expressions, synonym lists,
or lexical fallback.

E5 retrieval uses its required `query:` prefix for conversational context and
`passage:` for every indexed semantic card view.

Core mechanism and lived-experience views are the primary ranking signal.
Broader applicability and inquiry/limits views have lower semantic weights so
generic supporting language cannot outrank a direct mechanism match.

The E5-specific similarity floors are recorded in the release manifest. They
were calibrated against the real local index rather than copied from the
previous MiniLM implementation. A relative band around the best candidate plus
the existing token budget avoids both forced matches and a fixed card count.

After installing requirements, prepare the model and index once:

```
python -m ai.knowledge_retrieval --prepare-index
python -m ai.knowledge_retrieval --verify-index
```

The generated cache and index live outside the repository under local
application data. `KNOWLEDGE_RETRIEVAL_MODEL_CACHE` and
`KNOWLEDGE_RETRIEVAL_INDEX_PATH` can override those locations.

Semantic retrieval is enabled when `KNOWLEDGE_RETRIEVAL_ENABLED` is absent.
Set it explicitly to `false` for the emergency rollback to the previous
production assembly; in that state the embedding model is not loaded.
Retrieval failure logs a diagnostic and falls back to the previous assembly
without changing user state or attempting lexical retrieval.

The local development build writes one detailed Cognitive Core test trace by
default because it contains the evidence needed to debug live consultations.
Start the bot normally:

```
python main.py
```

Telegram `/start` replaces the previous trace and begins one clean test
session. Every later turn atomically updates the same Git-ignored file:
`diagnostics/latest_test_dialog.json`. It may contain personal conversation
content and must be shared deliberately. Disable it explicitly with
`COGNITIVE_CORE_DIAGNOSTICS_ENABLED=false`, or override its location with
`COGNITIVE_CORE_DIAGNOSTICS_PATH`. Inspect its path/content or clear it:

```
python -m ai.cognitive_diagnostics --path
python -m ai.cognitive_diagnostics --show
python -m ai.cognitive_diagnostics --purge
```
