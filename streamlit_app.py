# streamlit_app.py
import streamlit as st
from detector import FakeNewsDetector, pretty_print

st.set_page_config(page_title="Fake News Detector (Free & Local)", layout="wide")

st.title("📰 Fake News Detector — Free & Local")
st.caption("ReAct-style: Wikipedia retrieval + NLI stance checking. Inspired by AWS ML Blog.")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Wikipedia pages (k)", 1, 10, 5)
    m = st.slider("Evidence sentences (m)", 1, 15, 5)
    entail = st.slider("Entailment threshold", 0.0, 1.0, 0.55, 0.01)
    contra = st.slider("Contradiction threshold", 0.0, 1.0, 0.55, 0.01)

text = st.text_area("Paste a single claim or short statement:", height=140, placeholder='e.g. "Barbara Liskov was the first woman to earn a PhD in computer science."')

if st.button("Check factuality", type="primary") and text.strip():
    with st.spinner("Retrieving evidence and running NLI..."):
        det = FakeNewsDetector()
        verdict = det.classify_react(text.strip(), k_pages=k, m_sentences=m, entail_thresh=entail, contra_thresh=contra)
    st.subheader(f"Verdict: {verdict.verdict}")
    st.write(verdict.explanation)
    st.write(f"Support score: **{verdict.support_score:.2f}** • Refute score: **{verdict.refute_score:.2f}**")

    st.divider()
    st.subheader("Evidence")
    for i, ev in enumerate(verdict.evidence, start=1):
        with st.expander(f"{i}. {ev.title} — {ev.stance.upper()} (sim {ev.similarity:.2f})"):
            st.write(f"[{ev.title}]({ev.url})")
            st.write(ev.sentence)
            st.json(ev.probs)
