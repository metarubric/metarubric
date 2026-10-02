"""Case-balanced group errors and bounded log-weight updates."""
from __future__ import annotations

import importlib.util
import math
from collections import defaultdict
from pathlib import Path

V3_PHI=Path(__file__).resolve().parent/'legacy/vendor/control/phi.py'
from .profiles import PROFILES, profile, tau_cap

PROJECTORS = {}
for dataset in PROFILES:
    spec = importlib.util.spec_from_file_location("metarubrics_phi_" + dataset, V3_PHI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ANCHOR = dict(profile(dataset)["anchors"])
    module.TAU_CAP = tau_cap(dataset)
    PROJECTORS[dataset] = module


def group_errors(observations):
    """observations: independent per-item grades with train provenance."""
    by=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    seen=set()
    for o in observations:
        if o.get('split')!='train' or o.get('source')!='independent_full_rubric':
            raise ValueError('only independent training-side observations are allowed')
        if type(o['met']) is not bool or not math.isfinite(o['points']) or o['points']==0:
            raise ValueError('invalid signed criterion grade')
        key=(o['record_id'],o['criterion_id'])
        if key in seen:raise ValueError('duplicate judgment')
        seen.add(key)
        e=1-float(o['met']) if o['points']>0 else float(o['met'])
        by[o['cell']][o['pair_id']][o['record_id']].append(e)
    stats={}
    for cell,cases in by.items():
        case_losses=[]
        for records in cases.values():
            response_losses=[sum(v)/len(v) for v in records.values()]
            case_losses.append(sum(response_losses)/len(response_losses))
        stats[cell]={'error':sum(case_losses)/len(case_losses),'cases':len(cases),
                     'responses':sum(len(v) for v in cases.values())}
    return stats


def update_tau(old,stats,eta=.1,*,dataset="healthbench"):
    profile(dataset)
    if not math.isfinite(eta) or eta<0:raise ValueError('invalid step size')
    if set(stats)-set(old):raise ValueError('all tau cells must be initialized before updates')
    if any(not math.isfinite(v) for v in old.values()):raise ValueError('nonfinite tau')
    errors={k:s['error'] for k,s in stats.items() if s['cases']>0}
    if any(not math.isfinite(e) or not 0<=e<=1 for e in errors.values()):
        raise ValueError('invalid group error')
    if not errors or eta==0:return dict(old)
    mean=sum(errors.values())/len(errors)
    proposed={k:v+eta*(errors[k]-mean) if k in errors else v for k,v in old.items()}
    phi=PROJECTORS[dataset].Phi({tuple(k.split('|')):v for k,v in proposed.items()}).project()
    return {'|'.join(k):v for k,v in phi.tau.items()}
