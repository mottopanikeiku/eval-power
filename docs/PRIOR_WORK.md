# Prior work: how many items distinguish models?

**EXT** Closest predecessors: Kotawala's [Resolution Diagnostics for Paired LLM Evaluation](https://arxiv.org/html/2605.30315) and Basile et al.'s [How Many Samples Do We Need?](https://aclanthology.org/2026.lrec-1.353.pdf) already provide power-aware LLM audits; both full texts were opened.

**INFERENCE** Practical question: given a pilot of $n$ shared items, how many full-evaluation items does it predict, and how often is that prediction right? The expanded audit itself is not novel.

Labels: **EXT** = opened external source; **INFERENCE** = derivation or proposal; **MEAS**/**SIM** = project measurements/simulations. No project experiments were run for this memo.

## 1. Score estimation versus difference detection

**EXT** Miller [1, §§4–5] explicitly analyzes paired differences and assumes paired analysis in his sample-size equation. Appendix C covers clustered planning. Calling his contribution only an unpaired Gaussian formula, as Kotawala's comparison does, misses these parts.

**EXT** Card et al. [4, §§2–3] already simulate paired-accuracy power using accuracy difference, model agreement, and McNemar testing. They recommend validation/pilot parameter estimates and assumed-effect sensitivity, and warn against observed-effect post-hoc power as new evidence.

**EXT** Polo et al. [3, §§4–5] estimate full-benchmark scores using curated items, IRT, and combined estimators, evaluated on random/date-heldout models. Approximately 2-percentage-point average error with 100 items per scenario is not an 80%-power guarantee for small gaps. Rank correlation is evaluated, but close-gap testing guarantees are not established.

**EXT** Madaan et al. [2, §§3–5] analyze 13 benchmarks and over 280 models/checkpoints, separating seed variability, item-bootstrap intervals, and training monotonicity. Their tinyBenchmarks experiments preserve reasonably accurate means but increase seed variability and can reverse rankings. Continuous metrics/cloze prompts can reduce variability in their settings; changing either changes the estimand.

## 2. Direct overlap

**EXT** Kotawala [5, §§5–6, Appendix H] supplies paired MDE/required-item diagnostics, unpaired comparisons, exact/asymptotic McNemar and bootstrap checks, multiplicity, sequential sensitivity, and real subject clustering. Five models across four OLL tasks yield 40 comparisons. MMLU-Pro's nine top-ten adjacencies go from 4/9 unresolved to 6/9 with clustering. Validation resamples three real pairs for 1,000 trials. The opened implementation [5a] estimates full-matrix parameters, then samples those same rows **with replacement**, without a pilot/validation split. Adjacent versus all-pair multiplicity is already compared.

**EXT** Neuhof and Benjamini [6] construct Holm-adjusted rank intervals using paired tests on PromptEval/MMLU: 15 models, 57 subjects, 100 prompt variants. Subject heterogeneity dominates prompt variability in their comparison. Appendix A explicitly distinguishes marginal from simultaneous coverage and selection after ranking. Subject-aware, selection-valid ranking inference is therefore not new.

**EXT** Basile et al. [7] introduce sk-power, paired simulation-based planning, and an OLL audit exceeding 2,000 comparisons across five datasets. A larger panel alone is not methodological novelty. Blackwell et al. [8] study repeated-run score prediction intervals, a different uncertainty target.

## 3. Statistical recommendations

**INFERENCE**, following [1,4,5]: align item IDs; define $D_i=X_{Ai}-X_{Bi}$ and $\hat\delta=\bar D$. For independent items,

$$SE_{paired}^2=s_D^2/n,\qquad SE_{unpaired}^2=(s_A^2+s_B^2)/n.$$

Positive covariance explains pairing's advantage; marginal accuracies cannot identify it. Prespecify practical effect $\delta_0$, significance $\alpha$, and power $1-\beta$. Normal planning gives

$$n^*\approx (z_{1-\alpha/2}+z_{1-\beta})^2\sigma_D^2/\delta_0^2.$$

Here $z_p=\Phi^{-1}(p)$, reversing Miller's tail-index convention; this is not exact McNemar power.

**INFERENCE**, following [4,5]: binary discordance $q=P(D\ne0)$ gives $Var(D)=q-\delta^2$. With A-only successes $b$ and B-only successes $c$, exact conditional McNemar uses $b\mid(b+c)\sim Binomial(b+c,1/2)$ under the IID null. Exact power averages rejection over $M\sim Binomial(n,q)$ and $b\mid M\sim Binomial(M,(q+\delta)/(2q))$, for $q>0$ and $|\delta|\le q$. Dependent differences require cluster-aware tests.

**INFERENCE**, following [1,6]: declare micro-item versus equal-subject weighting and fixed-subject versus random-subject inference. For the latter, resample aligned whole-subject vectors; report cluster counts and leave-one-subject-out sensitivity. More questions in existing subjects are not interchangeable with more independent subjects.

**INFERENCE**, following [5,6]: Holm-correct a declared full family before displaying selected adjacencies. For sorted p-values, reject sequentially while $p_{(j)}\le\alpha/(m-j+1)$; stop at the first failure. Prespecifying an adjacency rule does not freeze score-selected pair identities. Bonferroni planning at $\alpha/m$ is conservative; Holm power requires the joint rejection pipeline, not retrospective thresholds.

## 4. Modest, testable extension

**INFERENCE** Proposed scope: all 395 models in the **2024 Open LLM Leaderboard source population**, across five binary-accuracy benchmarks—not today's frontier leaderboard. The extension is **out-of-pilot calibration**, not invention of paired power. Estimate variance/disagreement and choose planning pairs on pilot items; freeze choices; assess predicted versus heldout rejection curves, including close gaps. Report calibration error, sign errors, and Monte Carlo intervals. Separate observed-gap diagnostics from prespecified-effect planning. Unattainable sample sizes remain unattainable.

**INFERENCE** Compare IID/subject-sensitive designs where metadata exist and full-family Holm/descriptive adjacent-only results. Separately measure finite-benchmark recovery by without-replacement subsampling. For $N$ fixed observed differences, simple random sampling gives

$$Var(\bar D_n\mid D_{1:N})=(1-n/N)S_D^2/n,$$

with $S_D^2$ using denominator $N-1$. This concerns the fixed pool mean, **not** superpopulation uncertainty. Response matrices cannot establish future-subject power or decoding variability. Position this as a broader, independently calibrated replication with explicit estimands and selection handling, not a new statistical method.

## References (opened; EXT)

1. Evan Miller. 2024. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*. arXiv preprint 2411.00640. [Full text](https://arxiv.org/html/2411.00640).
2. Lovish Madaan, Aaditya K. Singh, Rylan Schaeffer, Andrew Poulton, Sanmi Koyejo, Pontus Stenetorp, Sharan Narang, Dieuwke Hupkes. 2024. *Quantifying Variance in Evaluation Benchmarks*. arXiv 2406.10229; NeurIPS 2024 **Regulatable ML workshop**, not verified as main-conference proceedings. [Full text](https://arxiv.org/html/2406.10229); [official venue record](https://neurips.cc/virtual/2024/106953).
3. Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, Mikhail Yurochkin. 2024. *tinyBenchmarks: evaluating LLMs with fewer examples*. ICML, PMLR 235:34303–34326. [Proceedings](https://proceedings.mlr.press/v235/maia-polo24a.html); [full text, arXiv 2402.14992](https://arxiv.org/html/2402.14992).
4. Dallas Card, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowald, Dan Jurafsky. 2020. *With Little Power Comes Great Responsibility*. EMNLP, pp. 9263–9274. [Publication record](https://aclanthology.org/2020.emnlp-main.745/); [read PDF](https://aclanthology.org/2020.emnlp-main.745.pdf).
5. Anany Kotawala. 2026. *Resolution Diagnostics for Paired LLM Evaluation*. [arXiv 2605.30315 full text](https://arxiv.org/html/2605.30315). [Companion repository](https://github.com/akotawala10/llm-power) identifies ICML 2026 Workshop on Hypothesis Testing.
   5a. [Opened prospective-analysis implementation](https://raw.githubusercontent.com/ananykotawala/llm-power/main/experiments/analysis_prospective.py).
6. Bitya Neuhof, Yuval Benjamini. 2026. *Quantifying Ranking Uncertainty in LLM Benchmarks*. [arXiv 2607.16259 full text](https://arxiv.org/html/2607.16259); conference venue unknown from opened metadata.
7. Angelo Basile, Areg Mikael Sarvazyan, José Ángel González. 2026. *How Many Samples Do We Need? A Toolkit for Power-Aware Evaluation Design*. LREC, pp. 4507–4513. [Publication record](https://aclanthology.org/2026.lrec-1.353/); [read PDF](https://aclanthology.org/2026.lrec-1.353.pdf).
8. Robert E. Blackwell, Jon Barry, Anthony G. Cohn. 2025 revision of 2024 preprint. *Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores*. [arXiv 2410.03492v2 full text](https://arxiv.org/html/2410.03492v2).

Written with AI coding assistance.
