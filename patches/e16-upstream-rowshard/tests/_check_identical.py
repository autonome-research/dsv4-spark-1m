"""Check that each added block in a PR diff hunk (contiguous '+' lines that
define a def/class/const) appears verbatim in the ported file."""
import sys, re
diff, target = sys.argv[1], sys.argv[2]
src = open(target).read()
blocks, cur = [], []
for line in open(diff):
    if line.startswith('+') and not line.startswith('+++'):
        cur.append(line[1:])
    else:
        if cur: blocks.append(''.join(cur)); cur = []
if cur: blocks.append(''.join(cur))
for b in blocks:
    if re.search(r'^(class |def |MIN_TP|    def |\S+ = )', b, re.M) and len(b.splitlines()) > 3:
        name = re.search(r'^(?:class|def)\s+(\w+)|^(\w+) =', b, re.M)
        print(('IDENTICAL ' if b in src else 'DIFFERS   ') + (name.group(1) or name.group(2)), len(b.splitlines()), 'lines')
