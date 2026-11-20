import json,statistics as s
from pathlib import Path
p=Path(__file__).parent
samples=[json.loads(x.read_text()) for x in sorted(p.glob('bench.*.json'))]
assert len(samples)==2 and all(x['status']=='PASS' for x in samples)
rows=[r for x in samples for r in x['pareto']]
names=samples[0]['fields']
med={n:s.median(v) for n,v in zip(names,zip(*rows))}
baseline=med['cpu_pulse_ms']
print(json.dumps({'median_ms':med,'same_quality_speed_index':{k:baseline/v*100 for k,v in med.items()},'slowdown_vs_cpu':{k:v/baseline for k,v in med.items()},'paired_rounds':len(rows),'seed_count':len(samples),'cpu_rows_per_sec':4096*1000/baseline,'measured_regime':'same host, 4096x8 score layouts, controls are exact literal pulse arrays, preallocated buffers; resident CUDA event vs host wall'},sort_keys=True,indent=2))
