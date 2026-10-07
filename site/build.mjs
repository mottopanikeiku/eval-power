import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { parseCSV } from './csv.mjs';

const site = fileURLToPath(new URL('.', import.meta.url));
const root = resolve(site, '..');
const dist = resolve(site, 'dist');
await mkdir(resolve(dist, 'data'), { recursive: true });
for (const file of ['index.html', 'style.css', 'app.mjs', 'math.mjs', 'csv.mjs']) {
  await copyFile(resolve(site, file), resolve(dist, file));
}
await copyFile(resolve(root, 'results/calibration.csv'), resolve(dist, 'data/calibration.csv'));
const config = JSON.parse(await readFile(resolve(root, 'results/configuration.json'), 'utf8'));
const plans = parseCSV(gunzipSync(await readFile(resolve(root, 'results/pilot_plans.csv.gz'))).toString('utf8'));
const groups = new Map();
for (const plan of plans) {
  const key = `${plan.benchmark}/${plan.pilot_n}`;
  if (!groups.has(key)) groups.set(key, { benchmark: plan.benchmark, pilot_n: Number(plan.pilot_n), total: 0, evaluated: 0, reached: 0 });
  const row = groups.get(key);
  row.total++;
  if (plan.iid_evaluated === 'True') {
    const observed = Number(plan.observed_iid);
    const requested = Number(plan.requested_power);
    if (!plan.observed_iid || !Number.isFinite(observed) || requested !== config.target_power) {
      throw new Error(`Invalid evaluated plan in ${key}.`);
    }
    row.evaluated++;
    if (observed >= requested) row.reached++;
  }
}
const summary = parseCSV(await readFile(resolve(root, 'results/calibration.csv'), 'utf8'));
for (const row of summary) {
  const counts = groups.get(`${row.benchmark}/${row.pilot_n}`);
  if (!counts || counts.total !== Number(row.pair_split_plans) || counts.evaluated !== Number(row.iid_evaluated)) {
    throw new Error('Individual plan counts do not match the committed calibration summary.');
  }
}
await writeFile(resolve(dist, 'data/pilot-summary.json'), JSON.stringify({
  source: 'results/pilot_plans.csv.gz',
  configuration: 'results/configuration.json',
  target_power: config.target_power,
  alpha: config.alpha,
  definition: 'Fraction of evaluable plans with observed_iid >= requested_power; point estimates, not guaranteed true power.',
  rows: [...groups.values()],
}, null, 2) + '\n');
console.log(`Built site from ${plans.length} committed pilot plans (${summary.length} benchmark/pilot summaries).`);
