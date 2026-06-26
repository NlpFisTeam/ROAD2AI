"""Sub-query decomposition for long legal questions."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

DEFAULT_THRESHOLD_1 = 30
DEFAULT_THRESHOLD_2 = 58

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def load_embed_tokenizer(embed_model_path: Path):
    return AutoTokenizer.from_pretrained(str(embed_model_path))


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def num_subqueries(
    token_count: int,
    *,
    threshold_1: int = DEFAULT_THRESHOLD_1,
    threshold_2: int = DEFAULT_THRESHOLD_2,
) -> int:
    if token_count < threshold_1:
        return 1
    if token_count < threshold_2:
        return 2
    return 3


def plan_queries(
    question: str,
    tokenizer,
    *,
    threshold_1: int = DEFAULT_THRESHOLD_1,
    threshold_2: int = DEFAULT_THRESHOLD_2,
) -> tuple[int, list[str] | None]:
    """Return (n_subqueries, None) — decomposition filled in later by LLM."""
    n = num_subqueries(
        count_tokens(question, tokenizer),
        threshold_1=threshold_1,
        threshold_2=threshold_2,
    )
    if n == 1:
        return 1, [question]
    return n, None


def _build_decompose_prompt(tokenizer, question: str, n: int) -> str:
    system = (
        "Bạn là chuyên gia pháp luật Việt Nam. "
        f"Tách câu hỏi pháp luật thành đúng {n} sub-query độc lập, "
        "mỗi sub-query tập trung một khía cạnh (điều kiện, mức phạt, thủ tục, thời hạn, v.v.). "
        "Chỉ trả về JSON array các chuỗi, không giải thích."
    )
    user = (
        f"Câu hỏi: {question}\n\n"
        f'Trả về đúng {n} sub-query dạng JSON array, ví dụ: ["sub-query 1", "sub-query 2"]'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _parse_subqueries(text: str, n: int, fallback: str) -> list[str]:
    text = text.strip()
    match = _JSON_ARRAY_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                subs = [str(s).strip() for s in parsed if str(s).strip()]
                if subs:
                    return subs[:n] if len(subs) >= n else subs + [fallback] * (n - len(subs))
        except json.JSONDecodeError:
            pass
    lines = [line.strip(" \"'-•") for line in text.splitlines() if line.strip()]
    subs = [line for line in lines if len(line) > 10]
    if len(subs) >= n:
        return subs[:n]
    return [fallback]


def decompose_question(
    question: str,
    n: int,
    model,
    tokenizer,
    eos_ids: list[int],
    *,
    max_new_tokens: int = 256,
) -> list[str]:
    if n <= 1:
        return [question]

    prompt = _build_decompose_prompt(tokenizer, question, n)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    raw = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    subs = _parse_subqueries(raw, n, question)
    return subs


def batch_decompose_questions(
    questions: list[dict],
    model,
    tokenizer,
    eos_ids: list[int],
    embed_tokenizer,
    *,
    threshold_1: int = DEFAULT_THRESHOLD_1,
    threshold_2: int = DEFAULT_THRESHOLD_2,
) -> dict[int, list[str]]:
    """Return question id -> list of query strings (including single-query cases)."""
    result: dict[int, list[str]] = {}
    pending: list[tuple[dict, int]] = []

    for q in questions:
        qid = q["id"]
        n, preset = plan_queries(
            q["question"],
            embed_tokenizer,
            threshold_1=threshold_1,
            threshold_2=threshold_2,
        )
        if preset is not None:
            result[qid] = preset
        else:
            pending.append((q, n))

    for q, n in pending:
        subs = decompose_question(
            q["question"],
            n,
            model,
            tokenizer,
            eos_ids,
        )
        result[q["id"]] = subs
        print(f"  Decompose id={q['id']} → {n} sub-query")

    return result
