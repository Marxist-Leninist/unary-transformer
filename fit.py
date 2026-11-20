"""Exact supervised discrete search over BOTH blocks on next-token labels."""
from decoder import *
from pathlib import Path
from itertools import product
import json
P=Path(__file__).parent
# 4 hidden link root/middle structures; never gives rationale labels.
FIT=(
 (('edge:b:c','label:a:m','label:b:m','edge:a:b','label:c:n','ask:a'),('S','EOS')),
 (('label:a:m','edge:c:d','label:d:n','edge:a:b','label:b:n','ask:a'),('R','EOS')),
 (('edge:b:c','label:a:n','label:b:m','edge:c:d','label:c:n','ask:b'),('R','EOS')),
 (('label:d:m','edge:c:d','edge:d:a','edge:b:c','label:c:m','ask:b'),('S','EOS')),
 (('label:b:n','edge:c:d','label:d:m','edge:d:a','label:a:n','ask:c'),('S','EOS')),
 (('edge:b:c','label:d:m','label:a:n','edge:d:a','label:b:m','ask:d'),('R','EOS')),
)
SIGNS=('-|','|')
PALETTES=((T,ZERO),(ZERO,T),(T,T))
def arm(q,k,v,x=None):
    w={'Q':q,'K':k,'V':v,'O':T,'F':T,'G':T}
    if x is not None:w['X']=x
    return w
def initial():
    w={'Q':ZERO,'K':ZERO,'V':ZERO,'O':ZERO,'F':ZERO,'G':ZERO}
    return {'embed':ZERO,'read':ZERO,'stop':ZERO,'one':w.copy(),'two':{**w,'X':ZERO}}
def fit():
    # Fixed sparse graph and declared domains: Q/K∈{0,±1}; all gain lanes∈{0,+1}.
    # Zero-severing skip: any zero mandatory link kills BOTH answer logits for every
    # labeled answer (except parallel V2 paths). Either zero Q/K gives no attention.
    # Zero stop/embed kills EOS. Such candidates cannot fit exact next-token labels.
    # Remaining configurations are searched jointly as full model checkpoints.
    zero=initial()
    assert predict(FIT[0][0],zero)!='S'
    cases=[]
    for pre,target in FIT:
        for tok in target:
            cases.append((pre,tok))
            pre=pre+(tok,)
    trials=ZERO; checks=ZERO
    for q,k,q2,k2,palette in product(SIGNS,SIGNS,SIGNS,SIGNS,PALETTES):
        w={'embed':T,'read':T,'stop':T,'one':arm(q,k,T),'two':arm(q2,k2,*palette)}
        trials=add(trials,T)
        for prefix,tok in cases:
            checks=add(checks,T)
            if predict(prefix,w)!=tok:break
        else:
            return {'schema':'two-block-causal-unary-v1','weights':w,'initial_next_token_loss':initial_objective(zero,cases),
                    'terminal_next_token_loss':initial_objective(w,cases),'tested_models_tally':trials,'tested_rows_tally':checks}
    raise RuntimeError('no fit')
def initial_objective(w,cases):
    misses=ZERO
    for prefix,tok in cases:
        if predict(prefix,w)!=tok:misses=add(misses,T)
    return misses
if __name__=='__main__':
    (P/'checkpoint.json').write_text(json.dumps(fit(),indent=2)+'\n')
