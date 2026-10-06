# eval-power

This checks whether a small, same-item LLM pilot can reliably plan a larger accuracy comparison.

**Question:** how many items do two models need, and how often does a pilot get that budget right?

This extends [Kotawala’s paired audit](https://arxiv.org/abs/2605.30315) and [Basile et al.’s planning toolkit](https://aclanthology.org/2026.lrec-1.353/) with disjoint-pilot calibration—not new power formulas. Data are tinyBenchmarks’ **selected 395-model, 2024 Open LLM Leaderboard population**, not current frontier models ([source and licenses](data/manifest.json)).

[stats.py](src/eval_power/stats.py) contains paired, unpaired and clustered errors plus an item-count planner. [analysis.py](src/eval_power/analysis.py) fits small pilots, then checks their predictions using heldout item subsampling and exact McNemar tests.

**Result:** plans targeting 80% power from a 128-item pilot’s observed gap delivered only **49.3–78.6% median heldout detection**, depending on benchmark ([calibration](results/calibration.csv)). A noisy pilot gap is not a safe design target.

## Planning budgets and leaderboard noise

Rounded medians, **items/model (paired / unpaired)**: prespecified differences, empirical non-identical adjacent-pair variances, two-sided α = 0.05, 80% Gaussian power—not simultaneous leaderboard budgets. Interquartile ranges and more differences: [lookup.csv](results/lookup.csv).

| Benchmark | 1 percentage point | 2 percentage points | Adjacent gaps not distinguishable |
|---|---:|---:|---:|
| ARC-Challenge | 13.0k / 37.3k | 3.3k / 9.3k | 99.7% |
| GSM8K | 15.1k / 24.4k | 3.8k / 6.1k | 100% |
| WinoGrande | 13.9k / 28.0k | 3.5k / 7.0k | 100% |
| HellaSwag | 5.7k / 22.4k | 1.4k / 5.6k | 98.7% |
| MMLU | 17.0k / 37.6k | 4.3k / 9.4k | 99.7% |

Noise counts use exact paired McNemar tests and Holm correction over **all 77,815 unordered pairs per benchmark**, before extracting the selected population’s 394 adjacencies. Rankings use item-weighted accuracy, with deterministic tie ordering ([audit](results/audit.csv), [all pair tests](results/all_pair_tests.npz)). Not distinguishable does **not** mean equal.

Measured adjacent-item correlations were 0.34–0.73; pairing reduced median standard errors by 19–48%. For MMLU, clustering its 57 subjects increased median adjacent SE by 1.60× and left no adjacent gap distinguishable ([audit](results/audit.csv)).

## How well do pilots predict power?

Pairs are selected by indices, not scores: 40 pairs per benchmark, five disjoint item splits, and 1,000 trials per point ([configuration](results/configuration.json)). Each pilot-gap plan targets 80%; the table reports its **median observed** detection under the empirical heldout IID distribution:

| Benchmark | Pilot: 128 items | Pilot: 256 items | 128-item plans below target* |
|---|---:|---:|---:|
| ARC-Challenge | 67.1% | 65.7% | 56.5% |
| GSM8K | 49.3% | 53.0% | 75.0% |
| WinoGrande | 50.4% | 80.8% | 60.1% |
| HellaSwag | 51.4% | 62.4% | 65.9% |
| MMLU | 78.6% | 67.1% | 48.5% |

*Among nonzero-gap, positive-variance plans: upper Monte Carlo 95% bounds below 80%; not future-population guarantees.*

Larger pilots did not uniformly improve this gap-based rule. Pilot-predicted rejection curves had 10.0–25.4 percentage-point mean absolute error at the smaller pilot size; a heldout-fitted normal reference had 1.9–4.4 points ([calibration](results/calibration.csv)). [Raw curves](results/validation_curves.csv.gz) and [plans](results/pilot_plans.csv.gz) include Monte Carlo intervals and unattainable finite-pool budgets. Without-replacement subsampling is reported separately; its finite-population correction is never used to erase future-item uncertainty.

![Out-of-pilot planning](figures/pilot_planning.svg)

[Calibration curves](figures/pilot_calibration.svg) · [Minimum detectable differences](figures/minimum_difference.svg) · [Leaderboard noise](figures/leaderboard_noise.svg)

## Reproduce and plan

Local CPU only; $0 paid compute, no model inference. Recorded hardware: Ryzen AI 5 PRO 340, single-threaded numerical libraries and Python 3.11 ([environment](results/environment.json)). Committed compressed matrices suffice; the optional [importer](scripts/import_data.py) pins and caches the original correctness artifact.

On the owner's shared machine, use `pp-run heavy` in place of `nice -n 19`; that scheduling wrapper is not a package dependency.

```sh
uv sync --locked
nice -n 19 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python scripts/analyze.py && uv run python scripts/figures.py
uv run eval-power --difference .01 --variance .2
```

The last command illustrates fixed-effect planning. Alternatively, `--pilot pilot.npy` takes an item-by-two-model binary matrix and reports paired and unpaired budgets. Choose the smallest meaningful difference **before** evaluating; a zero-variance pilot is refused. Optional group labels report clustered SE, not an IID item-budget extrapolation. [Tests](tests/) cover scalar references, exact binomial tests, cluster sums, integer power inversion and safe data import; [CI](.github/workflows/ci.yml) checks the committed results.

## Limits and prior work

- These are selected historical models, often related fine-tunes or merges—not independent families.
- Original document IDs and evaluation revisions are absent. MMLU alignment inherits the exporter’s ordering assumption; it was not independently ID-audited.
- Empirical IID trials resample heldout scores; they are not newly answered questions or proof of future power. One response per item cannot measure decoding variability.
- Subject clustering assumes independently sampled subjects. Adding questions to existing subjects is not adding independent subjects.
- Lookup medians hide pair variation; observed identical vectors do not justify a zero-item budget.

Foundations: [Miller 2024](https://arxiv.org/abs/2411.00640), [Madaan et al. 2024](https://arxiv.org/abs/2406.10229), [Polo et al. 2024](https://proceedings.mlr.press/v235/maia-polo24a.html), [Card et al. 2020](https://aclanthology.org/2020.emnlp-main.745/), and [Neuhof & Benjamini 2026](https://arxiv.org/abs/2607.16259). [Prior-work notes](docs/PRIOR_WORK.md) explain the overlap and extension. Code is MIT; data terms are recorded separately.

Written with AI coding assistance.
