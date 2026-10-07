import pytest

from eval_power.grading import generation_seed, grade_answer, grade_strict_answer, make_prompt


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
def test_strict_numeric_exact_match(text, reference, correct):
    assert grade_strict_answer("gsm8k", text, reference)[1] is correct


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
def test_strict_choice_exact_match(text, reference, correct):
    assert grade_strict_answer("arc", text, reference)[1] is correct


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


@pytest.mark.parametrize(
    ("text", "reference", "correct"),
    [
        ("#### The numeric answer is 10.", "10", True),
        ("Earlier 40 + 40 = 80. Final answer: $1,234.00.", "1234", True),
        ("The result is −3.5", "-3.5", True),
        ("The answer is .5", "0.5", True),
        ("#### 4 or 5", "4", False),
        ("No number here", "4", False),
        ("Calculation 9; final answer 10", "10", True),
    ],
)
def test_flexible_numeric_extraction(text, reference, correct):
    assert grade_answer("gsm8k", text, reference)[1] is correct


@pytest.mark.parametrize(
    ("text", "reference", "correct"),
    [
        ("The correct option is B.", "B", True),
        ("I initially considered B. The final answer is C.", "C", True),
        ("Answer: B. Answer: C", "C", True),
        ("Therefore (D).", "D", True),
        ("Choice 2 is right.", "2", True),
        ("No option was chosen.", "B", False),
        ("A. lightning.", "A", True),
        ("Nitrogen returns to soil by A. lightning.", "A", True),
        ("B. flammable\nAnswer: flammable", "B", True),
        ("A. first option; B. second option", "B", False),
        ("B. flammable\n1. It can burn.\n2. This is a chemical change.", "B", True),
    ],
)
def test_flexible_choice_extraction(text, reference, correct):
    assert grade_answer("arc", text, reference)[1] is correct


def test_child_rng_streams_do_not_overlap_between_items_models_benchmarks_or_phases():
    protocol = {"generation_seed": 1707, "seed_stride": 5}
    seeds = [
        generation_seed(protocol, model, benchmark, item, phase) + child
        for model in range(6)
        for benchmark in ("gsm8k", "arc")
        for phase in ("pilot", "confirm")
        for item in (0, 1, 2, 3, 4, 1171, 1318)
        for child in range(5)
    ]
    assert len(seeds) == len(set(seeds))
    assert generation_seed(protocol, 0, "gsm8k", 1, "pilot") == 1712
