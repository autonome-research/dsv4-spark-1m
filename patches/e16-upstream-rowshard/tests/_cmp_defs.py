"""Compare source text of named top-level defs/classes between two files (ast, no exec)."""
import ast, sys
a, b, names = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
def get(p):
    src = open(p).read(); t = ast.parse(src); out = {}
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            s = n.lineno - 1 - len(n.decorator_list)
            out[n.name] = '\n'.join(src.splitlines()[s:n.end_lineno])
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            tg = n.targets[0] if isinstance(n, ast.Assign) else n.target
            if isinstance(tg, ast.Name): out[tg.id] = ast.get_source_segment(src, n)
        if isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name):
                    out[n.name + '.' + m.target.id] = ast.get_source_segment(src, m)
    return out
A, B = get(a), get(b)
for nm in names:
    print(('SAME   ' if A.get(nm) == B.get(nm) and nm in A else 'DIFFER ') + nm, 'ours' if nm in A else 'MISSING-ours', 'up' if nm in B else 'MISSING-up')
