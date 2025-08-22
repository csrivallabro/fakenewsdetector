# Fake News Detector — Free & Local (ReAct-style)

This project implements a **free, local** fake-news detector inspired by the AWS Machine Learning Blog post
*Harness large language models in fake news detection* (Chain-of-Thought & ReAct approaches).  
It mirrors the **ReAct** logic using **Wikipedia retrieval + Natural Language Inference (NLI)** with **open-source models** on your machine — **no paid APIs required**.

> Key differences from the AWS post: we **don’t use Amazon Bedrock** or paid LLMs.  
> Instead, we use free Hugging Face models locally to emulate the same reasoning pattern (retrieve facts, check entailment/contradiction).

---

## What it does

1. **Takes a claim or article text.**
2. **Retrieves candidate evidence** from Wikipedia (free API).
3. **Ranks sentences** by semantic similarity to the claim.
4. **Runs NLI** on the top evidence sentences to decide if they **support** (entail) or **refute** (contradict) the claim.
5. Produces a **verdict**: `TRUE`, `FALSE`, or `UNVERIFIABLE`, with **explanations and citations**.

This follows the **ReAct idea** from the AWS post: reason + act (call a search tool), then reason again with the evidence.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### CLI usage

```bash
python cli.py --text "Barbara Liskov was the first woman to earn a PhD in computer science." --mode react --k 5 --m 3
```

- `--mode react` (default) uses retrieval + NLI (recommended).  
- `--k` = number of Wikipedia pages to fetch (default 5).  
- `--m` = number of evidence sentences to check (default 5).

**Sample output (truncated):**
```json
{
  "verdict": "FALSE",
  "support_score": 0.12,
  "refute_score": 0.78,
  "explanation": "Top evidence contradicts the claim.",
  "evidence": [
    {"title": "...", "url": "...", "sentence": "...", "stance": "contradiction", "p": 0.83},
    ...
  ]
}
```

### Optional: Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## Why this is free

- Models are **open-source** and run **locally**:
  - NLI: `cross-encoder/nli-deberta-v3-base` (~180MB)
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (~80MB)
- Evidence comes from the **Wikipedia** REST API (free).

> The first run will download model weights from Hugging Face (free).

---

## Requirements

See `requirements.txt`. This runs on CPU; GPU is optional.

---

## Notes & limitations

- **Heuristic thresholds** determine the verdict; you can tune them in `detector.py`.
- Works best for **single factual claims**. For long articles, paste a **single key claim** or run multiple times for different sentences.
- Wikipedia may not have up-to-date info for breaking news; verdict may be `UNVERIFIABLE`.
- This is a **reference/education** project, not production advice.

---

## License

MIT
