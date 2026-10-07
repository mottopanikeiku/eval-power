import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { parseCSV } from './csv.mjs';

// Build uses only the committed inputs; no network or hand-entered UI results.
execFileSync(process.execPath, [new URL('./build.mjs', import.meta.url).pathname]);
const counts = JSON.parse(await readFile(new URL('./dist/data/pilot-summary.json', import.meta.url), 'utf8'));
const calibration = parseCSV(await readFile(new URL('./dist/data/calibration.csv', import.meta.url), 'utf8'));
test('Calibration includes all committed plans and matches published denominators', () => {
  assert.equal(counts.rows.length, 10);
  assert.equal(counts.rows.reduce((sum, row) => sum + row.total, 0), 2000);
  assert.equal(counts.target_power, 0.8);
  assert.equal(counts.alpha, 0.05);
  for (const c of counts.rows) {
    const r = calibration.find((row) => row.benchmark === c.benchmark && Number(row.pilot_n) === c.pilot_n);
    assert.equal(c.total, Number(r.pair_split_plans));
    assert.equal(c.evaluated, Number(r.iid_evaluated));
    assert.ok(Number.isInteger(c.reached) && c.reached >= 0 && c.reached <= c.evaluated);
  }
});
test('Observed success is not the complement of definitely-below-target', () => {
  const arc = counts.rows.find((row) => row.benchmark === 'arc' && row.pilot_n === 128);
  assert.equal(arc.reached, 77);
  assert.equal(arc.evaluated, 191);
  const gsm = counts.rows.find((row) => row.benchmark === 'gsm8k' && row.pilot_n === 128);
  assert.equal(gsm.reached, 45);
  assert.equal(gsm.evaluated, 196);
  const summary = calibration.find((row) => row.benchmark === 'arc' && Number(row.pilot_n) === 128);
  assert.notEqual(arc.reached / arc.evaluated, 1 - Number(summary.iid_below_target_95mc_fraction));
});
