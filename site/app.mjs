import { comparisonVariance, gaussianPower, requiredItems } from './math.mjs';
import { parseCSV } from './csv.mjs';

const element = (id) => document.getElementById(id);
const number = (id) => Number(element(id).value);
const percent = (n) => `${(n * 100).toFixed(1)}%`;
const integer = new Intl.NumberFormat('en-US');

function showField(id, shown) {
  element(id).hidden = !shown;
  for (const input of element(id).querySelectorAll('input, select')) input.disabled = !shown;
}

function updateFields() {
  const useGap = element('input-mode').value === 'gap';
  showField('gap-field', useGap);
  showField('accuracy-b-field', !useGap);
  const paired = element('design').value === 'paired';
  showField('paired-fields', paired);
  showField('correlation-field', paired && element('dependence').value === 'correlation');
  showField('discordance-field', paired && element('dependence').value === 'discordance');
}
for (const id of ['input-mode', 'design', 'dependence']) element(id).addEventListener('change', updateFields);
updateFields();

element('theme').addEventListener('change', () => {
  const theme = element('theme').value;
  if (theme === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
});

function calculate() {
  try {
    const pa = number('accuracy-a') / 100;
    const pb = element('input-mode').value === 'gap' ? pa + number('gap') / 100 : number('accuracy-b') / 100;
    const paired = element('design').value === 'paired';
    const { variance, discordance } = comparisonVariance({
      pa, pb, paired, dependence: element('dependence').value,
      rho: number('correlation'), discordance: number('discordance') / 100,
    });
    const delta = pb - pa;
    const alpha = number('alpha');
    const power = number('power') / 100;
    const n = requiredItems(delta, variance, alpha, power);
    element('item-count').textContent = integer.format(n);
    element('item-unit').textContent = paired ? 'shared items · both models answer each item' : `items per model · ${integer.format(BigInt(n) * 2n)} responses in total`;
    element('plan-details').textContent = `For a ${Math.abs(delta * 100).toFixed(3)}-point gap, alpha ${alpha}, and ${percent(power)} target power. Variance: ${variance.toFixed(6)}.${paired ? ` Discordance: ${percent(discordance)}.` : ''} Gaussian power at this budget: ${percent(gaussianPower(delta, variance, n, alpha))}.`;
    element('plan-error').hidden = true;
  } catch (error) {
    element('item-count').textContent = '—';
    element('item-unit').textContent = 'No budget calculated';
    element('plan-details').textContent = '';
    element('plan-error').textContent = error.message;
    element('plan-error').hidden = false;
  }
}
element('plan-form').addEventListener('submit', (event) => {
  event.preventDefault();
  calculate();
});
calculate();

let calibration;
let pilotSummary;
function renderCalibration() {
  if (!calibration || !pilotSummary) return;
  const benchmark = element('benchmark').value;
  const pilot = number('pilot');
  const row = calibration.find((r) => r.benchmark === benchmark && Number(r.pilot_n) === pilot);
  const counts = pilotSummary.rows.find((r) => r.benchmark === benchmark && r.pilot_n === pilot);
  if (!row || !counts) throw new Error('No committed results for this benchmark and pilot.');
  const below = Number(row.iid_below_target_95mc_fraction);
  const above = Number(row.iid_above_target_95mc_fraction);
  const uncertain = Math.max(0, 1 - below - above);
  // All interpolated values are numbers from local committed data.
  element('calibration').innerHTML = `
    <div class="metrics">
      <div class="metric"><h3>Estimated power ≥ 80%</h3><strong>${percent(counts.reached / counts.evaluated)}</strong><p>${counts.reached} of ${counts.evaluated} evaluable plans reached the target in the simulation.</p></div>
      <div class="metric"><h3>Median detection rate</h3><strong>${percent(Number(row.iid_power_median))}</strong><p>Middle half: ${percent(Number(row.iid_power_q25))}–${percent(Number(row.iid_power_q75))}.</p></div>
      <div class="metric"><h3>Clearly below target</h3><strong>${percent(below)}</strong><p>The upper 95% Monte Carlo bound was below 80%.</p></div>
    </div>
    <p><strong>${counts.evaluated} of ${counts.total} plans were evaluable.</strong> ${counts.total - counts.evaluated} had zero pilot gap or variance and are excluded, not counted as successes.</p>
    <div class="bar" aria-hidden="true"><span class="below" style="width:${100 * below}%"></span><span class="uncertain" style="width:${100 * uncertain}%"></span><span class="above" style="width:${100 * above}%"></span></div>
    <p class="hint">95% Monte Carlo bounds: ${percent(below)} below target · ${percent(uncertain)} include target · ${percent(above)} above target. These categories are stricter than comparing the point estimate to 80%.</p>`;
}
for (const id of ['benchmark', 'pilot']) element(id).addEventListener('change', renderCalibration);

try {
  const responses = await Promise.all([fetch('data/calibration.csv'), fetch('data/pilot-summary.json')]);
  if (responses.some((response) => !response.ok)) throw new Error('Calibration files could not be loaded.');
  calibration = parseCSV(await responses[0].text());
  pilotSummary = await responses[1].json();
  renderCalibration();
} catch (error) {
  element('calibration').textContent = `Committed calibration results are unavailable: ${error.message} Use the source links below.`;
  element('calibration').classList.add('error');
}
