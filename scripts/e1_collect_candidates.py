"""Collect natural candidate answers from the local models (E1 step 1).

fm questions:    closed-book for every model; for the three fastest models also RAG with the top-4 passages
                 (fm-knowledge-assistant's own prompt) and RAG with passages ranked 4-7 (a simulated retrieval miss).
                 Four passages rather than the assistant's six, and three models rather than five, because a
                 2,000-token prompt takes 10-30 s to read on the 4 GB GPU.
sgbus questions (all 180 for qwen2.5:3b, the odd-numbered 90 for the others, to fit the time budget):
                 with a four-row reference table that holds the answer among distractor rows (every model), and
                 closed-book (the three fast models only; the two 4B models read prompts about 4x slower here).

Models run one after another (the 4 GB GPU holds one at a time). Output: results/e1/candidates.jsonl.
Requires local/fm_passages.jsonl from scripts/e1_fm_retrieve.py.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab.items import fact_card, load_jsonl, write_jsonl  # noqa: E402
from evalab.llm import Cache, chat, unload  # noqa: E402

MODELS = ["qwen2.5:3b", "qwen3.5:4b", "llama3.2:3b", "gemma3:4b", "phi4-mini:3.8b"]
RAG_MODELS = {"qwen2.5:3b", "llama3.2:3b", "phi4-mini:3.8b"}

# fm-knowledge-assistant's answer prompt (src/fmka/answer.py), used unchanged
FM_SYSTEM = (
    "You answer questions about Singapore building and facilities-management codes using ONLY the numbered passages "
    "provided. Cite every statement with the passage number in square brackets, e.g. [2]. Quote numbers and units "
    "exactly as written. If the passages do not contain the answer, reply exactly NOT_FOUND. Be concise (at most 3 "
    "sentences).")
FM_CLOSED = ("You answer questions about Singapore building and facilities-management codes from your own knowledge. "
             "Give the specific value the code requires. Be concise (at most 3 sentences).")
BUS_CLOSED = "You answer questions about Singapore's public bus network. Be concise (one or two sentences)."
BUS_CARD = ("You answer questions about Singapore's public bus network using the reference table provided. "
            "Be concise (one or two sentences).")


def jobs(model):
    fm = load_jsonl(ROOT / "data" / "e1" / "fm_gold.jsonl")
    passages = {r["id"]: r for r in load_jsonl(ROOT / "local" / "fm_passages.jsonl")}
    bus = load_jsonl(ROOT / "data" / "e3" / "sgbus_items.jsonl")
    for q in fm:
        for cfg, key in (("rag_top4", "top6"), ("rag_miss", "drop3")):
            if model not in RAG_MODELS:
                continue
            numbered = "\n\n".join(f"[{n}] {t}" for n, t in enumerate(passages[q["id"]][key][:4], start=1))
            yield q["id"], "fm", cfg, [{"role": "system", "content": FM_SYSTEM},
                                       {"role": "user", "content": f"Passages:\n{numbered}\n\nQuestion: {q['question']}"}]
        yield q["id"], "fm", "closed", [{"role": "system", "content": FM_CLOSED},
                                        {"role": "user", "content": q["question"]}]
    for q in bus:
        # qwen2.5:3b (run first) answered all 180; to fit the time budget the other models answer the odd-numbered
        # half (90 questions, every template)
        if model != "qwen2.5:3b" and int(q["id"][-4:]) % 2 == 0:
            continue
        if model in RAG_MODELS:
            yield q["id"], "sgbus", "closed", [{"role": "system", "content": BUS_CLOSED},
                                               {"role": "user", "content": q["question"]}]
        yield q["id"], "sgbus", "card", [{"role": "system", "content": BUS_CARD},
                                         {"role": "user", "content": f"Reference table:\n{fact_card(q)}\n\n"
                                                                     f"Question: {q['question']}"}]


def main():
    cache = Cache(ROOT / "cache" / "candidates.jsonl")
    rows = []
    for m in MODELS:
        todo = list(jobs(m))
        for n, (qid, src, cfg, msgs) in enumerate(todo, start=1):
            r = chat(m, msgs, cache=cache, num_predict=160, options={"num_ctx": 8192 if src == "fm" else 4096})
            rows.append({"cand_id": f"{qid}|{m}|{cfg}", "qid": qid, "source": src, "generator": m, "config": cfg,
                         "answer": r.text.strip(), "prompt_tokens": r.prompt_tokens,
                         "output_tokens": r.output_tokens, "seconds": r.seconds})
            if n % 100 == 0:
                print(m, n, len(todo), flush=True)
        unload(m)
    write_jsonl(ROOT / "results" / "e1" / "candidates.jsonl", rows)
    print("candidates", len(rows))


if __name__ == "__main__":
    main()
