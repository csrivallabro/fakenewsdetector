# cli.py
import argparse, json
from detector import FakeNewsDetector, pretty_print

def main():
    ap = argparse.ArgumentParser(description="Free, local fake news detector (ReAct-style)")
    ap.add_argument("--text", required=True, help="Claim or short passage to check")
    ap.add_argument("--mode", default="react", choices=["react"], help="Detection mode (react only in this free build)")
    ap.add_argument("--k", type=int, default=5, help="Wikipedia pages to fetch")
    ap.add_argument("--m", type=int, default=5, help="Evidence sentences to test")
    ap.add_argument("--entail", type=float, default=0.55, help="Entailment threshold for TRUE")
    ap.add_argument("--contra", type=float, default=0.55, help="Contradiction threshold for FALSE")
    args = ap.parse_args()

    det = FakeNewsDetector()
    res = det.classify_react(args.text, k_pages=args.k, m_sentences=args.m, entail_thresh=args.entail, contra_thresh=args.contra)
    print(pretty_print(res))

if __name__ == "__main__":
    main()
