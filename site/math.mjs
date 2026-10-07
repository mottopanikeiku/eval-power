// Two-sided Gaussian planning, matching src/eval_power/stats.py.
const finite = (value, name) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be a finite number.`);
  }
  return value;
};

export function normalCDF(x) {
  finite(x, 'Normal argument');
  const a = Math.abs(x);
  let tail;
  if (a < 2) {
    // Integral of the normal density as a positive, convergent series.
    let term = a;
    let sum = term;
    for (let k = 1; k < 500; k++) {
      term *= a * a / (2 * k + 1);
      sum += term;
      if (term <= sum * Number.EPSILON) break;
    }
    tail = 0.5 - Math.exp(-a * a / 2) / Math.sqrt(2 * Math.PI) * sum;
  } else {
    // Laplace's continued fraction for the normal tail (Mills ratio).
    let denominator = a;
    for (let k = 200; k >= 1; k--) denominator = a + k / denominator;
    tail = Math.exp(-a * a / 2) / Math.sqrt(2 * Math.PI) / denominator;
  }
  tail = Math.max(0, Math.min(0.5, tail));
  return x < 0 ? tail : 1 - tail;
}

export function normalQuantile(p) {
  finite(p, 'Probability');
  if (!(p > 0 && p < 1)) throw new RangeError('Probability must be between 0 and 1.');
  // Work in the lower tail, avoiding cancellation for small alpha.
  const lowerProbability = p > 0.5 ? 1 - p : p;
  let lower = -40;
  let upper = 0;
  for (let k = 0; k < 100; k++) {
    const middle = (lower + upper) / 2;
    if (normalCDF(middle) < lowerProbability) lower = middle;
    else upper = middle;
  }
  const result = (lower + upper) / 2;
  return p > 0.5 ? -result : result;
}

function validatePlan(delta, variance, alpha) {
  finite(delta, 'Gap');
  finite(variance, 'Variance');
  finite(alpha, 'Alpha');
  if (Math.abs(delta) > 1) throw new RangeError('The absolute gap cannot exceed 100 percentage points.');
  if (!(variance > 0)) throw new RangeError('Variance must be positive; a zero-variance pilot cannot support planning.');
  if (!(alpha > 0 && alpha < 1)) throw new RangeError('Alpha must be between 0 and 1.');
}

function normalPower(delta, variance, n, z) {
  const noncentrality = Math.abs(delta) * Math.sqrt(n) / Math.sqrt(variance);
  return normalCDF(-z - noncentrality) + normalCDF(noncentrality - z);
}

export function gaussianPower(delta, variance, n, alpha = 0.05) {
  validatePlan(delta, variance, alpha);
  if (!Number.isSafeInteger(n) || n <= 0) throw new RangeError('Item count must be a positive safe integer.');
  return normalPower(delta, variance, n, -normalQuantile(alpha / 2));
}

export function requiredItems(delta, variance, alpha = 0.05, power = 0.8) {
  validatePlan(delta, variance, alpha);
  finite(power, 'Power');
  if (delta === 0) throw new RangeError('Choose a nonzero meaningful gap; equal accuracies have no finite budget.');
  if (!(power > alpha && power < 1)) throw new RangeError('Power must exceed alpha and be less than 1.');
  const z = -normalQuantile(alpha / 2);
  // The same monotone integer correction used by Python; no rounded closed-form shortcut.
  let upper = 1;
  while (normalPower(delta, variance, upper, z) < power) {
    if (upper > Number.MAX_SAFE_INTEGER / 2) {
      upper = Number.MAX_SAFE_INTEGER;
      if (normalPower(delta, variance, upper, z) < power) {
        throw new RangeError('Item budget exceeds the safe integer range. Choose a larger gap.');
      }
      break;
    }
    upper *= 2;
  }
  let lower = 0;
  while (upper - lower > 1) {
    const middle = lower + Math.floor((upper - lower) / 2);
    if (normalPower(delta, variance, middle, z) >= power) upper = middle;
    else lower = middle;
  }
  return upper;
}

export function comparisonVariance({ pa, pb, paired, dependence = 'correlation', rho = 0, discordance }) {
  finite(pa, 'Model A accuracy');
  finite(pb, 'Model B accuracy');
  if (pa < 0 || pa > 1 || pb < 0 || pb > 1) throw new RangeError('Accuracies must be between 0% and 100%.');
  const independent = pa * (1 - pa) + pb * (1 - pb);
  const minDiscordance = Math.abs(pa - pb);
  const maxDiscordance = Math.min(pa + pb, 2 - pa - pb);
  if (!paired) return { variance: independent, discordance: null, minDiscordance, maxDiscordance };
  let q;
  if (dependence === 'discordance') {
    q = finite(discordance, 'Discordance');
  } else if (dependence === 'correlation') {
    finite(rho, 'Correlation');
    if (rho < -1 || rho > 1) throw new RangeError('Correlation must be between −1 and 1.');
    const scale = Math.sqrt(pa * (1 - pa) * pb * (1 - pb));
    if (scale === 0) throw new RangeError('Correlation is undefined for a 0% or 100% accuracy. Use discordance instead.');
    q = pa + pb - 2 * (pa * pb + rho * scale);
  } else {
    throw new RangeError('Choose correlation or discordance.');
  }
  const tolerance = 1e-12;
  if (q < minDiscordance - tolerance || q > maxDiscordance + tolerance) {
    throw new RangeError(`These accuracies require discordance between ${(100 * minDiscordance).toFixed(2)}% and ${(100 * maxDiscordance).toFixed(2)}%. Choose a feasible correlation or discordance.`);
  }
  q = Math.max(minDiscordance, Math.min(maxDiscordance, q));
  return { variance: q - (pa - pb) ** 2, discordance: q, minDiscordance, maxDiscordance };
}
