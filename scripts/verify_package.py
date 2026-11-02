"""Verify the downloaded package before regenerating any output files."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    errors = []
    count = 0
    for line in (ROOT / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError('Invalid path in checksum manifest')
        count += 1
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(relative)
    if errors:
        raise SystemExit('Missing or modified files: ' + ', '.join(errors))
    print(f'Verified {count} files against SHA256SUMS.txt')

if __name__ == '__main__':
    main()
