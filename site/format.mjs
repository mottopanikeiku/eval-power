export function formatDecimal(value, places) {
  const rounded = Number(value.toFixed(places));
  // Keep small nonzero values visible instead of describing them as zero.
  return String(rounded === 0 && value !== 0 ? Number(value.toPrecision(3)) : rounded);
}

export const formatPercent = (value, places = 1) => `${formatDecimal(value * 100, places)}%`;
export const formatPointGap = (delta) => formatDecimal(Math.abs(delta) * 100, 3);
