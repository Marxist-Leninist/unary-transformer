T="|"
NEG="-"
ZERO=""
def flip(x):
    return ZERO if not x else x.removeprefix(NEG) if x.startswith(NEG) else NEG+x
def add(a,b):
    pa=not a.startswith(NEG);pb=not b.startswith(NEG)
    a=a.removeprefix(NEG);b=b.removeprefix(NEG)
    if pa==pb:return (ZERO if pa else NEG)+a+b if a or b else ZERO
    while a and b:
        a=a.removesuffix(T);b=b.removesuffix(T)
    if a:return a if pa else NEG+a
    return (b if pb else NEG+b) if b else ZERO
def mul(a,b):
    out=ZERO
    repeats=a.removeprefix(NEG)
    for mark in repeats:out=add(out,b)
    return flip(out) if a.startswith(NEG) else out
def gt(a,b):
    return add(a,flip(b)).startswith(T)
