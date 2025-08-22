# detector.py
# Core logic: Wikipedia retrieval + NLI stance detection (entail/contradict/neutral).

from __future__ import annotations
import re
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np
import wikipedia
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util as st_util


@dataclass
class Evidence:
    title: str
    url: str
    sentence: str
    similarity: float
    stance: str | None = None
    probs: Dict[str, float] | None = None


@dataclass
class Verdict:
    verdict: str
    support_score: float
    refute_score: float
    explanation: str
    evidence: List[Evidence]


def split_into_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter
    text = re.sub(r'\s+', ' ', text).strip()
    # Split on ., ?, ! while keeping abbreviations somewhat intact
    parts = re.split(r'(?<=[.!?])\s+', text)
    # Clean small/empty
    return [p.strip() for p in parts if len(p.strip()) > 30]  # keep reasonably informative sentences


class FakeNewsDetector:
    def __init__(self, device: str | int = "auto"):
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        # Embeddings for sentence similarity
        self.embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=("cuda" if self.device==0 else "cpu"))

        # Cross-encoder NLI for stance (entail/contradict/neutral)
        self.nli_name = "cross-encoder/nli-deberta-v3-base"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_name)
        self.nli_model.to("cuda" if self.device==0 else "cpu")
        self.nli_model.eval()

        # Label mapping from model config
        cfg = self.nli_model.config
        self.id2label = getattr(cfg, "id2label", {0:"CONTRADICTION", 1:"NEUTRAL", 2:"ENTAILMENT"})
        # Normalize to lowercase keywords we use later
        self.label_norm = {i: self.id2label[i].lower() for i in self.id2label}

    # ---------------- Wikipedia tools ----------------

    def wiki_search(self, query: str, k: int = 5) -> List[Dict]:
        titles = wikipedia.search(query, results=k)
        pages = []
        for t in titles:
            try:
                page = wikipedia.page(t, auto_suggest=False)
                pages.append({"title": page.title, "url": page.url, "content": page.content})
            except Exception:
                # Disambiguation or fetch error — skip
                continue
        return pages

    def _pick_top_sentences(self, claim: str, pages: List[Dict], m: int = 5) -> List[Evidence]:
        # Gather sentences with page context
        sentences = []
        for p in pages:
            for s in split_into_sentences(p["content"][:5000]):  # limit per page for speed
                sentences.append((p["title"], p["url"], s))

        if not sentences:
            return []

        # Rank by semantic similarity
        sent_texts = [s[2] for s in sentences]
        claim_emb = self.embed.encode([claim], convert_to_tensor=True, normalize_embeddings=True)
        sent_emb = self.embed.encode(sent_texts, convert_to_tensor=True, normalize_embeddings=True)
        sims = st_util.cos_sim(claim_emb, sent_emb).cpu().numpy().ravel()
        idx = np.argsort(-sims)[:m]

        ev = []
        for i in idx:
            title, url, s = sentences[i]
            ev.append(Evidence(title=title, url=url, sentence=s, similarity=float(sims[i])))
        return ev

    # ---------------- NLI ----------------

    def nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        inputs = self.nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.nli_model.device)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        # Map to standardized keys
        out = {self.label_norm[i]: float(probs[i]) for i in range(len(probs))}
        # Ensure keys exist
        for k in ["entailment","contradiction","neutral"]:
            if k not in out:
                # Try alternate capitalization
                for kk, vv in list(out.items()):
                    if kk.lower().startswith(k):
                        out[k] = vv
                out.setdefault(k, 0.0)
        return out

    # ---------------- Public API ----------------

    def classify_react(self, text: str, k_pages: int = 5, m_sentences: int = 5, entail_thresh: float = 0.55, contra_thresh: float = 0.55) -> Verdict:
        """
        ReAct-style: search Wikipedia, pick relevant sentences, run NLI to score support vs refute.
        """
        pages = self.wiki_search(text, k=k_pages)
        if not pages:
            return Verdict(
                verdict="UNVERIFIABLE",
                support_score=0.0,
                refute_score=0.0,
                explanation="No suitable Wikipedia evidence found.",
                evidence=[]
            )

        evidence = self._pick_top_sentences(text, pages, m=m_sentences)

        support = 0.0
        refute = 0.0
        best_support = None
        best_refute = None

        for ev in evidence:
            probs = self.nli(premise=ev.sentence, hypothesis=text)
            ev.probs = probs
            # stance = argmax label
            stance = max(probs, key=probs.get)
            ev.stance = stance

            if probs.get("entailment", 0.0) > support:
                support = probs["entailment"]
                best_support = ev
            if probs.get("contradiction", 0.0) > refute:
                refute = probs["contradiction"]
                best_refute = ev

        # Simple decision rule (tunable)
        if support >= entail_thresh and support - refute >= 0.15:
            verdict = "TRUE"
            explanation = "Top evidence sentences entail the claim."
        elif refute >= contra_thresh and refute - support >= 0.15:
            verdict = "FALSE"
            explanation = "Top evidence sentences contradict the claim."
        else:
            verdict = "UNVERIFIABLE"
            explanation = "Evidence is mixed or insufficient."

        # Sort evidence for readability (highest of max(entail,contradict))
        evidence_sorted = sorted(
            evidence,
            key=lambda e: max(e.probs.get("entailment",0.0), e.probs.get("contradiction",0.0)),
            reverse=True
        )

        return Verdict(
            verdict=verdict,
            support_score=float(support),
            refute_score=float(refute),
            explanation=explanation,
            evidence=evidence_sorted
        )


def pretty_print(verdict: Verdict) -> str:
    data = asdict(verdict)
    # Convert dataclasses in list
    data["evidence"] = [asdict(e) for e in verdict.evidence]
    return json.dumps(data, indent=2, ensure_ascii=False)
