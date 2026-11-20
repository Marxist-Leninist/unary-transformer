import ast,hashlib,json,operator,random
from pathlib import Path
from decoder import *
from fit import FIT
P=Path(__file__).parent
checkpoint=(P/'checkpoint.json').read_bytes()
w=json.loads(checkpoint)['weights']
def rngcase(r,noise,offsets):
    root_i=r.randrange(len(KEYS))
    middle_i=(root_i+r.choice(offsets))%len(KEYS)
    root,middle=KEYS[root_i],KEYS[middle_i]
    edge={k:(middle if k==root else r.choice(KEYS)) for k in KEYS}
    lab={k:r.choice('mn') for k in KEYS}
    records=[f'edge:{k}:{v}' for k,v in edge.items()]+[f'label:{k}:{v}' for k,v in lab.items()]
    records+=['.']*noise
    r.shuffle(records)
    return tuple(records+[f'ask:{root}']),('S' if lab[middle]=='m' else 'R','EOS'),(root,middle)
def fixture(seed,noise,offsets):
    r=random.Random(seed);rows={}
    while len(rows)<96:
        x,y,links=rngcase(r,noise,offsets);rows[x]=(y,links)
    return tuple(rows.items())
cohorts={'new_links_only':fixture(7119,0,(0,2,3)),'length17_and_new_links':fixture(4721,8,(0,2,3)),'length65_and_new_links':fixture(15671,56,(0,2,3)),
         'length129_same_link_types':fixture(7407,120,(1,))}
assert all(x not in {p for p,t in FIT} for rows in cohorts.values() for x,_ in rows)
old_links={('a','b'),('b','c'),('c','d'),('d','a')}
assert all(pair not in old_links for k,rows in cohorts.items() if 'new_links' in k for x,(y,pair) in rows)
assert sum(map(len,cohorts.values()))==len({x for rows in cohorts.values() for x,_ in rows})
def edit(w,where,name,value):
    return {**w,where:{**w[where],name:value}}
arms={'full':(w,()), 'no_block1':(w,('layer1',)), 'no_block2':(w,('layer2',)),
      'no_ff1':(edit(w,'one','F',''),()),'no_ff2':(edit(w,'two','F',''),()),
      'identity_only_value':(edit(edit(w,'two','X',''),'two','V','|'),()),
      'always_R_then_EOS':None}
stats={}
for name,pack in arms.items():
    stats[name]={}
    for k,rows in cohorts.items():
        good=end=next_tokens=0
        for prefix,(expected,_) in rows:
            if pack is None:generated,terminal=('R','EOS'),True; predicted=('R','EOS')
            else:
                wa,disabled=pack
                generated,terminal=generate(prefix,wa,disabled)
                predicted=(predict(prefix,wa,disabled),predict(prefix+expected[:1],wa,disabled))
            good+=terminal and generated==expected
            end+=terminal
            next_tokens+=sum(a==b for a,b in zip(predicted,expected))
        stats[name][k]={'n':len(rows),'free_gen_exact_with_eos':good,'emitted_eos':end,'teacher_forced_token_exact':next_tokens}
assert all(s['free_gen_exact_with_eos']==s['n'] and s['teacher_forced_token_exact']==2*s['n'] for s in stats['full'].values())
counterfactuals=0; rewired=0
for k,rows in cohorts.items():
    for pre,(expected,(root,middle)) in rows[:12]:
        swap='m' if expected[0]=='R' else 'n'
        x=tuple(f'label:{middle}:{swap}' if t.startswith(f'label:{middle}:') else t for t in pre)
        assert generate(x,w)==((('R' if expected[0]=='S' else 'S'),'EOS'),True)
        counterfactuals+=1
        alternatives=[t.split(':')[1] for t in pre if t.startswith('label:') and t.endswith(':'+swap)]
        if alternatives:
            alt=alternatives[0]
            proposal=tuple(f'edge:{root}:{alt}' if t.startswith(f'edge:{root}:') else t for t in pre)
            assert generate(proposal,w)==(('R' if expected[0]=='S' else 'S','EOS'),True)
            rewired+=1
# Causal prefix invariance: the common prefix logits must be exactly equal when a
# new future record is appended. Run the actual first-layer function for its earlier anchor.
causality=0
for pre,_ in cohorts['length17_and_new_links'][:16]:
    states=[];original=[]
    for t in pre:
        h=embed(t,w['embed']);original.append(apply_layer(states,h,w['one'],L1));states.append(h)
    x=embed('label:b:m',w['embed'])
    extended=tuple(original)+(apply_layer(states,x,w['one'],L1),)
    assert apply_layer(original[:-1],original[-1],w['two'],L2)==apply_layer(extended[:-2],extended[-2],w['two'],L2)
    causality+=1
for file in ('decoder.py','tally.py','fit.py'):
    tree=ast.parse((P/file).read_text())
    assert not any(isinstance(n,ast.Constant) and isinstance(n.value,float) for n in ast.walk(tree))
index={arm:round(100*sum(d['free_gen_exact_with_eos']/d['n'] for d in vals.values())/len(vals),8) for arm,vals in stats.items()}
print(json.dumps({'checkpoint_sha256':hashlib.sha256(checkpoint).hexdigest(),'fixtures_sha256':hashlib.sha256(repr(cohorts).encode()).hexdigest(),'macro_exact_generation_index':index,'tests':stats,'counterfactual_payloads_flipped':counterfactuals,'counterfactual_pointers_rewired':rewired,'causality_checks':causality,'training_2_tokens_each':len(FIT)},indent=2,sort_keys=True))
