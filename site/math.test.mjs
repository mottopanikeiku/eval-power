import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { comparisonVariance, gaussianPower, normalCDF, normalQuantile, requiredItems } from './math.mjs';
import { parseCSV } from './csv.mjs';

const fixture = JSON.parse(await readFile(new URL('./python-fixture.json', import.meta.url), 'utf8'));
for (const [index, c] of fixture.cases.entries()) {
  test(`Python item budget and power ${index}: gap=${c.delta}, v=${c.variance}, alpha=${c.alpha}, power=${c.power}`, () => {
    const n = requiredItems(c.delta, c.variance, c.alpha, c.power);
    assert.ok(Math.abs(n - c.items) <= fixture.item_tolerance, `${n} vs Python ${c.items}`);
    assert.ok(Math.abs(gaussianPower(c.delta, c.variance, c.items, c.alpha) - c.power_at_items) <= fixture.power_absolute_tolerance);
    if (c.items > 1) assert.ok(Math.abs(gaussianPower(c.delta, c.variance, c.items - 1, c.alpha) - c.power_before_items) <= fixture.power_absolute_tolerance);
    assert.ok(gaussianPower(c.delta, c.variance, n, c.alpha) >= c.power);
    if (n > 1) assert.ok(gaussianPower(c.delta, c.variance, n - 1, c.alpha) < c.power);
  });
}
for (const [index, c] of fixture.comparisons.entries()) {
  test(`Python binary variance ${index}`, () => {
    const inputs = { ...c, paired: true };
    const qVariance = comparisonVariance({ ...inputs, dependence: 'discordance' }).variance;
    assert.ok(Math.abs(qVariance - c.paired_variance) <= fixture.variance_absolute_tolerance);
    if (c.rho !== null) assert.ok(Math.abs(comparisonVariance(inputs).variance - c.paired_variance) <= fixture.variance_absolute_tolerance);
    assert.ok(Math.abs(comparisonVariance({ ...inputs, paired: false }).variance - c.unpaired_variance) <= fixture.variance_absolute_tolerance);
  });
}
test('Normal CDF tails and inverse', () => {
  assert.equal(normalCDF(0), 0.5);
  for (const p of [1e-8, 1e-6, 0.025, 0.1, 0.5, 0.8, 0.99, 1 - 1e-8]) {
    assert.ok(Math.abs(normalCDF(normalQuantile(p)) - p) < 1e-14);
  }
  assert.ok(normalCDF(-8) > 0);
  assert.ok(Math.abs(normalQuantile(0.025) + 1.959963984540054) < 1e-12);
});
test('Invalid or impossible plans are refused', () => {
  for (const args of [[0, 0.2], [0.01, 0], [0.01, -1], [NaN, 0.2], [0.01, 0.2, 0], [0.01, 0.2, 1], [0.01, 0.2, 0.05, 0.05], [0.01, 0.2, 0.05, 1], [1.1, 0.2], [true, 0.2]]) {
    assert.throws(() => requiredItems(...args), RangeError);
  }
  assert.throws(() => requiredItems(1e-100, 0.2), /safe integer/);
  assert.throws(() => gaussianPower(0.1, 0.2, 1.5), /integer/);
  assert.throws(() => comparisonVariance({ pa: 0.1, pb: 0.9, paired: true, rho: 0.9 }), /require discordance/);
  assert.throws(() => comparisonVariance({ pa: 0.7, pb: 0.72, paired: true, dependence: 'discordance', discordance: 0.01 }), /require discordance/);
  assert.throws(() => comparisonVariance({ pa: 0, pb: 0.5, paired: true, rho: 0 }), /undefined/);
  assert.throws(() => comparisonVariance({ pa: 0.7, pb: 1.01, paired: false }), /Accuracies/);
});
test('CSV quoting, CRLF, and blank lines', () => {
  assert.deepEqual(parseCSV('a,b\r\n"x,y","quoted ""word"""\r\n\r\n'), [{ a: 'x,y', b: 'quoted "word"' }]);
  assert.throws(() => parseCSV('a,b\n1'), /header/);
});
