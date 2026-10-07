// RFC 4180-style quoted fields, including CRLF and embedded delimiters.
export function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (!quoted && (c === ',' || c === '\n' || c === '\r')) {
      row.push(field);
      field = '';
      if (c !== ',') {
        if (row.some((value) => value !== '')) rows.push(row);
        row = [];
        if (c === '\r' && text[i + 1] === '\n') i++;
      }
    } else field += c;
  }
  if (quoted) throw new Error('Unclosed quoted CSV field.');
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  const headers = rows.shift();
  if (!headers) throw new Error('CSV has no header.');
  return rows.map((values) => {
    if (values.length !== headers.length) throw new Error('CSV row does not match its header.');
    return Object.fromEntries(headers.map((header, i) => [header, values[i]]));
  });
}
