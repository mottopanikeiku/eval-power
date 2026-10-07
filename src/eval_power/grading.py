"""Explicit prompt and exact-match rules for the new inference runs."""

import re
from decimal import Decimal, InvalidOperation


def make_prompt(benchmark, row):
    if benchmark == "gsm8k":
        reference = row["answer"].split("####")[-1].strip()
        return (
            row["question"]
            + "\nSolve this problem. End with #### followed by the numeric answer.",
            reference,
        )
    if benchmark == "arc":
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(row["choices"]["label"], row["choices"]["text"], strict=True)
        )
        return (
            row["question"]
            + "\n"
            + choices
            + "\nChoose one option. End with Answer: followed by its label.",
            row["answerKey"],
        )
    raise ValueError(f"Unknown benchmark: {benchmark}")


def numeric_value(text):
    try:
        value = Decimal(text.replace(",", "").strip())
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def grade_answer(benchmark, text, reference):
    if benchmark == "gsm8k":
        if "####" not in text:
            return None, False
        suffix = text.rsplit("####", 1)[1].strip()
        match = re.match(r"^([-+]?\d[\d,]*(?:\.\d+)?)\s*[.!]?\s*$", suffix)
        prediction = match.group(1).replace(",", "") if match else None
        return prediction, prediction is not None and numeric_value(prediction) == numeric_value(
            reference
        )
    if benchmark == "arc":
        marker = re.search(r"\bAnswer\s*:\s*([A-E1-5])\b", text, re.IGNORECASE)
        plain = re.fullmatch(r"\s*([A-E1-5])[.!]?\s*", text, re.IGNORECASE)
        match = marker or plain
        prediction = match.group(1).upper() if match else None
        return prediction, prediction == reference.upper()
    raise ValueError(f"Unknown benchmark: {benchmark}")
