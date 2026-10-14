import test from 'node:test';
import assert from 'node:assert/strict';
import { formatDecimal, formatPercent, formatPointGap } from './format.mjs';
import { comparisonVariance } from './math.mjs';

test('Point gaps trim zeros and floating-point subtraction noise', () => {
  for (const delta of [0.02, 0.72 - 0.7, 0.7 - 0.72]) {
    assert.equal(formatPointGap(delta), '2');
    assert.equal(`For a ${formatPointGap(delta)}-point gap`, 'For a 2-point gap');
  }
  assert.equal(formatPointGap(0.025), '2.5');
  assert.equal(formatPointGap(0.00125), '0.125');
  assert.equal(formatPointGap(0.123456), '12.346');
  assert.equal(formatPointGap(0.1), '10');
  assert.equal(formatPointGap(1), '100');
});

test('Small nonzero gaps stay visible and zero has no decimal padding', () => {
  assert.equal(formatPointGap(0.000004), '0.0004');
  assert.equal(formatPointGap(0), '0');
  assert.equal(formatPointGap(-0), '0');
});

test('Percentages and variance use the same unpadded decimal style', () => {
  assert.equal(formatPercent(0.8), '80%');
  assert.equal(formatPercent(0.20601191), '20.6%');
  assert.equal(formatPercent(0.0234, 2), '2.34%');
  assert.equal(formatDecimal(0.20584311, 6), '0.205843');
  assert.equal(formatDecimal(0.21, 6), '0.21');
});

test('Feasible-discordance errors also trim trailing zeros', () => {
  assert.throws(
    () => comparisonVariance({ pa: 0.7, pb: 0.72, paired: true, dependence: 'discordance', discordance: 0.01 }),
    /discordance between 2% and 58%\./,
  );
});
