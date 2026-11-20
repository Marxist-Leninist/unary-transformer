"""Sparse decoder-only Transformer: two unary causal attention + FFN residual blocks."""
from tally import T,ZERO,NEG,add,mul,gt
KEYS=('a','b','c','d')
OUT=('R','S','EOS')
FEATURES=('req','edgeK','edgeV','labelK','labelV','f1','scratch','f2','ans','done')
PARAMS=('embed','Q','K','V','O','F','G','read','stop')
UNIT=(ZERO,T)
WATCHDOG='||||'
def zeros():
    return {name:dict.fromkeys(KEYS,ZERO) for name in FEATURES}
def basis(key,v):
    return {a:v if a==key else ZERO for a in KEYS}
def scalar(v):
    return basis('a',v)
def embed(tok,w):
    h=zeros()
    # This is a structured symbolic vocabulary, not natural-language BPE.
    role,*args=tok.split(':')
    if role=='ask':
        key,=args;h['req']=basis(key,w)
    elif role=='edge':
        key,value=args;h['edgeK']=basis(key,w);h['edgeV']=basis(value,w)
    elif role=='label':
        key,value=args;h['labelK']=basis(key,w);h['labelV']=basis('a' if value=='m' else 'b',w)
    elif tok in ('R','S'):
        h['done']=scalar(w)
    elif tok!='.':
        raise ValueError('unknown token')
    return h
def vector_add(a,b):
    return {k:add(a[k],b[k]) for k in KEYS}
def vector_scale(x,w):
    return {k:mul(x[k],w) for k in KEYS}
def dot(x,y):
    ret=ZERO
    for k in KEYS:ret=add(ret,mul(x[k],y[k]))
    return ret
def relu(x):
    return {k:v if v.startswith(T) else ZERO for k,v in x.items()}
def value_project(x,w):
    base=vector_scale(x,w['V'])
    if 'X' not in w:return base
    swap=x.copy();swap['a'],swap['b']=x['b'],x['a']
    return vector_add(base,vector_scale(swap,w['X']))
def apply_layer(states,h,w,lanes):
    qkey,kkey,vkey,fkey,okey=lanes
    q=vector_scale(h[qkey],w['Q'])
    best=ZERO; chosen=None
    if any(q.values()):
        for prev in states:
            score=dot(q,vector_scale(prev[kkey],w['K']))
            if gt(score,best):
                chosen,best=prev,score
    r=h.copy()
    if chosen is not None:
        r[fkey]=vector_add(r[fkey],vector_scale(value_project(chosen[vkey],w),w['O']))
    r[okey]=vector_add(r[okey],vector_scale(relu(vector_scale(r[fkey],w['F'])),w['G']))
    return r
L1=('req','edgeK','edgeV','f1','scratch')
L2=('scratch','labelK','labelV','f2','ans')

def last_hidden(tokens,w,disable=()):
    before=[];out1=[]
    for t in tokens:
        x=embed(t,w['embed'])
        out1.append(x if 'layer1' in disable else apply_layer(before,x,w['one'],L1))
        before.append(x)
    last=out1.pop()
    return last if 'layer2' in disable else apply_layer(out1,last,w['two'],L2)

def logits(prefix,w,disable=()):
    h=last_hidden(prefix,w,disable)
    raw=mul(h['ans']['a'],w['read'])
    # Read from the two learned categorical logits.
    counter=mul(h['ans']['b'],w['read'])
    stop=mul(h['done']['a'],w['stop'])
    return {'R':raw,'S':counter,'EOS':stop}

def predict(prefix,w,disable=()):
    scores=logits(prefix,w,disable)
    best=ZERO; chosen='.'
    for t,s in scores.items():
        if gt(s,best):chosen,best=t,s
    return chosen

def generate(prefix,w,disable=()):
    seq=list(prefix);new=[]
    for mark in WATCHDOG:
        token=predict(seq,w,disable)
        new.append(token)
        if token=='EOS':return tuple(new),True
        seq.append(token)
    return tuple(new),False
