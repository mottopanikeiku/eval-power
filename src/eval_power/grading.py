"""Explicit prompt and exact-match rules for the new inference runs."""

import re
from decimal import Decimal, InvalidOperation


def make_prompt(benchmark, row):
    if benchmark == "gsm8k":
        reference = row["answer"].split("####")[-1].strip()
        return (
            row["question"] + "\nSolve this problem. End with #### followed by the numeric answer.",
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


def grade_strict_answer(benchmark, text, reference):
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


def grade_answer(benchmark, text, reference):
    """Flexible exact match; the last number follows lm-eval's GSM8K convention."""
    if benchmark == "gsm8k":
        numbers = re.findall(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)", text.replace("−", "-"))
        if not numbers:
            return None, False
        value = numeric_value(numbers[-1])
        prediction = format(value.normalize(), "f") if value is not None else None
        return prediction, value is not None and value == numeric_value(reference)
    if benchmark == "arc":
        labels = "12345" if reference.isdecimal() else "ABCDE"
        markers = re.findall(
            r"\b(?:answer|option|choice)\s*(?:is|:|=)?\s*[\(\[]?([A-E1-5])\b",
            text,
            re.IGNORECASE,
        )
        markers = [label.upper() for label in markers if label.upper() in labels]
        parenthesized = re.findall(r"\(([A-E1-5])\)", text, re.IGNORECASE)
        listed = re.findall(r"(?<!\w)([A-E1-5])[.)]\s+\S", text, re.IGNORECASE)
        explicit = {label.upper() for label in [*listed, *parenthesized] if label.upper() in labels}
        final = re.search(r"\b([A-E1-5])[.!]?\s*$", text, re.IGNORECASE)
        prediction = (
            markers[-1]
            if markers
            else next(iter(explicit))
            if len(explicit) == 1
            else final.group(1)
            if final and final.group(1).upper() in labels
            else None
        )
        prediction = prediction.upper() if prediction else None
        return prediction, prediction == reference.upper()
    raise ValueError(f"Unknown benchmark: {benchmark}")


def generation_seed(protocol, model_index, benchmark, dataset_index, phase):
    """Reserve k child seeds per item; vLLM's n children use parent_seed + index."""
    return (
        protocol["generation_seed"]
        + model_index * 1000000
        + (100000 if phase == "confirm" else 0)
        + (10000 if benchmark == "arc" else 0)
        + dataset_index * protocol["seed_stride"]
    )
