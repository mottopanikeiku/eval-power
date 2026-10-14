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

**EXT** The pilot-effect pitfall itself predates LLM evaluation. Albers and Lakens [9, §§3–8] show how noisy pilot effect estimates and selective follow-up can produce underpowered main studies, and recommend a smallest effect size of interest fixed independently of the pilot. Their full publisher PDF was opened. Kraemer et al. [10] warn against using pilot studies to guide proposal power calculations; publisher and PubMed abstracts/metadata were opened, not the paywalled full text. **INFERENCE** This project measures the size of that known problem for paired LLM benchmarks. Using pilots to estimate variance while fixing a meaningful effect beforehand is the standard alternative, not a newly validated future-power guarantee.

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

**INFERENCE** Proposed scope: all 395 models in the **2024 Open LLM Leaderboard source population**, across five binary-accuracy benchmarks—not today's frontier leaderboard. The extension measures the known pilot-effect planning problem on disjoint paired benchmark items; it does not invent paired power or discover that pitfall [9,10]. Estimate variance/disagreement; fix the meaningful target effect before evaluation; assess predicted versus heldout rejection curves. Keep observed-gap diagnostics separate from prespecified-effect planning, and leave unattainable sample sizes unattainable.

**INFERENCE** Compare IID/subject-sensitive designs where metadata exist and full-family Holm/descriptive adjacent-only results. Separately measure finite-benchmark recovery by without-replacement subsampling. For $N$ fixed observed differences, simple random sampling gives

$$Var(\bar D_n\mid D_{1:N})=(1-n/N)S_D^2/n,$$

with $S_D^2$ using denominator $N-1$. This concerns the fixed pool mean, **not** superpopulation uncertainty. Response matrices cannot establish future-subject power or decoding variability. Position this as a broader, independently calibrated replication with explicit estimands and selection handling, not a new statistical method.

## 5. Repeated answers and preference votes

I extend the accuracy analysis with newly generated answers from pinned instruct models on identified, disjoint benchmark items. The [collection protocol](../results/prospective/protocol.json) separates repeated-decoding noise from item-selection variation using a standard within-item/between-item variance decomposition, not a new power formula. I retain negative component estimates before clipping them for planning. The observed-gap rule deliberately tests the pilot-planning concern in [9,10]; it is not my recommendation for choosing a meaningful effect.

I estimate

$$
\operatorname{Var}(\bar Y_A-\bar Y_B)=\sigma^2_{\mathrm{item}}+\sigma^2_{\mathrm{decode}}/k.
$$

I estimate decoding variance within questions and subtract it from the observed variance of item means. I retain the unclipped item estimate before applying a zero floor for planning.

I published the [sample-size plan](https://github.com/mottopanikeiku/eval-power/commit/bbbae30a37a3b80ca43af6e8e23444e3892bdb78) after the pilots and before confirmation. Each feasible pair received its planned 16–264 new questions and a paired t-test on five-answer item means. I reused answers across comparisons sharing a model and collected **9,222 confirmation answers**, separate from the pilot. Three pairs exceeded the 512-question collection limit and were not run: GSM8K Qwen3–Phi required 870; ARC Qwen1.5–Mistral required 1,503 and Qwen3–Phi required 881.

I use flexible final-number extraction inspired by lm-eval-harness [12] for GSM8K. My initial grader demanded `####`; relaxing numeric extraction changed Qwen1.5's score from 15.9% to 65.9% on the same original sampled texts. Phi still truncated 18.1% of unconstrained ARC samples at 1,024 tokens. I changed all six models to guided direct-choice generation using vLLM [13]: each can generate only `Answer: <one of the provided labels>`. This changes the quantity being measured and is not conventional option-likelihood scoring.

I found that vLLM's child-seed offset [14] caused neighboring questions to reuse 426 of 3,840 requested streams in the initial pilot. I reserved a separate block of child seeds for every item and recollected both primary pilots: **3,840 distinct stochastic seeds, zero reused**. I retained the earlier outputs in the [pilot revision analysis](../results/prospective/pilot_revision.json). Strict-format rescoring of the primary confirmation detects 22/27 differences; I did not plan the sample sizes for that secondary metric.

The [collector](../modal_app.py) records prompts' hashes, answers, seeds, revisions and generation settings. I estimate all cloud calls, including failed setup and diagnostic reruns, at **at most $3.02** from [booked resources](../results/prospective/costs.json). This is not an invoice or a local timing measurement.

I also analyze public human-preference votes using the Bradley–Terry approach used by Chatbot Arena [11]. I compare inverse-Hessian intervals with an individual-vote bootstrap, then test pilot plans by resampling disjoint heldout votes. This is historical resampling, not collection of new human votes. Ties, graph identification, failed fits, score-independent planning-pair selection, and the distinction between total network votes and direct pair votes are described in [the methods](../results/arena/methods.txt).

I retained only model names and outcome counts from the Apache-2.0 release, not conversations or user IDs. For the llama-2-7b-chat / llama2-70b-steerlm-chat comparison, I planned 142,611 network votes but detected a difference in only 18/200 held-out resamples.


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
9. Casper Albers, Daniël Lakens. 2018. *When power analyses based on pilot data are biased: Inaccurate effect size estimators and follow-up bias*. Journal of Experimental Social Psychology 74:187–195. [DOI](https://doi.org/10.1016/j.jesp.2017.09.004); [opened publisher PDF via Groningen](https://pure.rug.nl/ws/files/64849114/1_s2.0_S002210311630230X_main.pdf).
10. Helena Chmura Kraemer, Jim Mintz, Art Noda, Jared Tinklenberg, Jerome A. Yesavage. 2006. *Caution regarding the use of pilot studies to guide power calculations for study proposals*. Archives of General Psychiatry 63(5):484–489. DOI: 10.1001/archpsyc.63.5.484. [Opened PubMed record/abstract](https://pubmed.ncbi.nlm.nih.gov/16651505/); [opened publisher metadata/abstract](https://www.ovid.com/journals/aogp/abstract/00000756-200605000-00003~caution-regarding-the-use-of-pilot-studies-to-guide-power).
11. Wei-Lin Chiang et al. 2024. *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*. [Opened arXiv abstract and metadata](https://arxiv.org/abs/2403.04132). [Pinned public vote source and license](../data/arena_manifest.json).
12. EleutherAI lm-evaluation-harness. [Opened GSM8K strict-match and flexible-extract configuration](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml). My numeric normalization is inspired by, not byte-identical to, its regex pipeline.
13. vLLM 0.10.2. [Opened native guided-choice documentation](https://docs.vllm.ai/en/v0.10.2/serving/openai_compatible_server.html#extra-parameters); [guided-decoding parameters](https://docs.vllm.ai/en/v0.10.2/api/vllm/sampling_params.html).
14. vLLM 0.10.2. [Opened repeated-request seed implementation](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/engine/parallel_sampling.py#L45-L74): child seed is parent seed plus child index.

Written with AI coding assistance.
