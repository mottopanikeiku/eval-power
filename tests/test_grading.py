import pytest

from eval_power.grading import grade_answer, make_prompt


@pytest.mark.parametrize(
    ("text", "reference", "correct"),
    [
        ("Work.\n#### 1,234", "1234", True),
        ("#### 2.0", "2", True),
        ("#### -3", "-3", True),
        ("#### 4.\n", "4", True),
        ("The answer is 4", "4", False),
        ("#### 4 or 5", "4", False),
        ("#### 5\n#### 4", "4", True),
        ("#### NaN", "NaN", False),
        ("#### 4", "5", False),
    ],
)
def test_numeric_exact_match(text, reference, correct):
    assert grade_answer("gsm8k", text, reference)[1] is correct


@pytest.mark.parametrize(
    ("text", "reference", "correct"),
    [
        ("Reasoning. Answer: B", "B", True),
        ("b", "B", True),
        ("Answer: 2", "2", True),
        ("The answer might be B", "B", False),
        ("Answer: B. Answer: C", "C", False),
        ("Answer: C", "B", False),
    ],
)
def test_choice_exact_match(text, reference, correct):
    assert grade_answer("arc", text, reference)[1] is correct


def test_arc_prompt_retains_labels():
    row = {
        "question": "Which?",
        "choices": {"label": ["1", "2"], "text": ["first", "second"]},
        "answerKey": "2",
    }
    prompt, reference = make_prompt("arc", row)
    assert "1. first\n2. second" in prompt
    assert reference == "2"


def test_unknown_benchmark_rejected():
    with pytest.raises(ValueError):
        grade_answer("unknown", "x", "x")
