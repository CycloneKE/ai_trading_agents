// Build-time settings. NEXT_PUBLIC_SOURCE_HASH fingerprints the dashboard's
// own source so it can tell when the server was built from different code
// (src/utils/build_info.py computes the same fingerprint the same way).
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const FOLDERS = ['components', 'pages', 'utils'];

function jsFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) return jsFiles(full);
    return e.isFile() && e.name.endsWith('.js') ? [full] : [];
  });
}

function sourceHash(root) {
  const files = FOLDERS.flatMap((f) => jsFiles(path.join(root, f)))
    .map((f) => path.relative(root, f).split(path.sep).join('/'))
    .sort();
  if (!files.length) return '';
  const digest = crypto.createHash('sha256');
  for (const rel of files) {
    digest.update(Buffer.concat([Buffer.from(rel), Buffer.from([0]),
      fs.readFileSync(path.join(root, rel)), Buffer.from([0])]));
  }
  return digest.digest('hex').slice(0, 12);
}

module.exports = {
  env: { NEXT_PUBLIC_SOURCE_HASH: sourceHash(__dirname) },
};
