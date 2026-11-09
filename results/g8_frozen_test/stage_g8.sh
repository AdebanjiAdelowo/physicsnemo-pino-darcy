#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-$HOME/Documents/GitHub/physicsnemo-pino-darcy}"
ARCHIVE="${2:-$HOME/Downloads/G8_verified_results.zip}"
cd "$REPO"
test "$(git branch --show-current)" = 'research/regularization-controls' || { echo 'Wrong branch'; exit 1; }
test -z "$(git status --porcelain)" || { echo 'Working tree not clean'; exit 1; }
git fetch origin research/regularization-controls
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/research/regularization-controls)" || { echo 'Branch not synced'; exit 1; }
mkdir -p results/g8_frozen_test docs
python3 - "$ARCHIVE" <<'PYCODE'
import sys,zipfile,hashlib,pathlib
a=pathlib.Path(sys.argv[1]);dst=pathlib.Path('results/g8_frozen_test')
with zipfile.ZipFile(a) as z:
 names=z.namelist(); assert len(names)==12 and len(set(names))==12
 for name in names:
  assert '/' not in name and '..' not in name
  target=pathlib.Path('docs/G8_ARCHIVE_VERIFICATION.md') if name=='G8_ARCHIVE_VERIFICATION.md' else dst/name
  target.write_bytes(z.read(name))
 lines=(dst/'SHA256SUMS').read_text().splitlines()
 hashes={n:h for h,n in (line.split('  ',1) for line in lines if '  ' in line)}
 for name in names:
  if name.endswith('.json'):
   assert hashlib.sha256((dst/name).read_bytes()).hexdigest()==hashes[name],name
 print('Verified nine JSON SHA256 hashes')
PYCODE
git add docs/G8_ARCHIVE_VERIFICATION.md results/g8_frozen_test
git diff --cached --stat
echo 'Review staged changes. Commit with:'
echo 'git commit -m "results: archive verified G8 frozen test outputs"'
echo 'git push origin research/regularization-controls'
