"""MetaRubrics verl hook: frozen content and weight adaptation per segment."""
import contextvars
import hashlib
import importlib.util
import json
import math
import os
import uuid
import time
from pathlib import Path

from metarubrics.revisions import apply_revisions, digest, validate_snapshot
from metarubrics.profiles import profile

ROOT=Path(__file__).resolve().parent / "legacy"
spec=importlib.util.spec_from_file_location('metarubrics_base_reward',ROOT/'vendor/verl_reward_v3.py')
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
CTX=contextvars.ContextVar('metarubrics_request')


def apply_phi(contract):
    if CTX.get()['split'] != 'train' or not contract.get('rubric_meta'):return
    if len(contract['rubric_meta'])!=len(contract['rubrics']):
        raise ValueError('rubric metadata mismatch')
    ctx=CTX.get()
    anchors = profile(ctx['phi'].get('dataset', 'healthbench'))['anchors']
    contract['rubrics']=[{**r,'points':round(math.copysign(1.,m.get('sign',r['points']))
                    *anchors[m['severity']]*math.exp(ctx['phi']['tau'].get(m['severity']+'|'+m['edit_label'],0.)),6)}
                    for r,m in zip(contract['rubrics'],contract['rubric_meta'])]


def capture(contract,result,response,reference_label):
    if not contract.get('rubric_meta'):return
    CTX.get()['captured']=(contract,result,response)


_decode = base._decode_contract


def decode_revised(ground_truth, extra_info):
    contract = _decode(ground_truth, extra_info)
    ctx = CTX.get()
    ctx['sample_id'] = contract.get('sample_id')
    ctx['base_contract_sha256'] = digest(contract)
    result = apply_revisions(contract, ctx['phi'], ctx['split'])
    ctx['revised_contract_sha256'] = digest(result)
    return result


base._decode_contract = decode_revised
base._apply_live_tau=apply_phi
base._dump_online_scored=capture
_parse_exam=base._exam_index


def strict_exam_index(text,n):
    value=_parse_exam(text,n)
    if value is None:
        raise ValueError('extractor did not return a valid option index')
    return value


base._exam_index=strict_exam_index
retry_spec=importlib.util.spec_from_file_location('metarubrics_exam_reader',ROOT/'resilient_exam.py')
retry_module=importlib.util.module_from_spec(retry_spec)
retry_spec.loader.exec_module(retry_module)


def _record_exam_retry(event):
    ctx=CTX.get()
    ctx.setdefault('exam_retry_events',[]).append(event)
    if event.get('reason') in ('invalid_option_index','truncated_invalid_index'):
        folder=Path(__file__).resolve().parents[2]/'exam_format_events'
        folder.mkdir(exist_ok=True)
        record={'time':time.time(),'sample_id':ctx.get('sample_id'),'snapshot_sha256':digest(ctx['phi']),**event}
        with (folder/(uuid.uuid4().hex+'.json')).open('x') as f:json.dump(record,f,ensure_ascii=False)



base._exam_score=retry_module.make_exam_score(base,_record_exam_retry)


async def compute_score(*args,phi_snapshot=None,trace_dir=None,**kwargs):
    if phi_snapshot is None or trace_dir is None:
        raise ValueError('explicit phi_snapshot and trace_dir are required')
    raw=Path(phi_snapshot).read_bytes()
    phi=json.loads(raw)
    validate_snapshot(phi)
    dataset = phi.get('dataset', 'healthbench')
    source = kwargs.get('data_source', args[0] if args else None)
    if source not in profile(dataset)['data_sources']:
        raise ValueError('snapshot dataset does not match data_source')
    extra = kwargs.get('extra_info', args[3] if len(args) > 3 else None)
    reader_model=kwargs.get('exam_extractor_model') or os.environ.get('ACRE_EXAM_EXTRACTOR_MODEL')
    reader_url=kwargs.get('exam_extractor_url') or os.environ.get('ACRE_EXAM_EXTRACTOR_URL')
    expected_model=os.environ.get('METARUBRIC_EXAM_MODEL','exam-qwen3-1.7b')
    expected_url=os.environ.get('METARUBRIC_EXAM_URL','http://127.0.0.1:19331')
    if reader_model != expected_model or reader_url != expected_url:
        raise RuntimeError('wrong exam reader configuration; refusing policy update')
    ctx={'phi':phi,'captured':None,'split':(extra or {}).get('split')}
    token=CTX.set(ctx)
    rid=uuid.uuid4().hex
    try:
        result=await base.compute_score(*args,**kwargs)
        result['method'] = 'MetaRubrics'
        result['dataset'] = dataset
        result['metarubrics_version'] = phi['version']
        result['metarubrics_snapshot_sha256'] = digest(phi)
        result['split'] = ctx['split']
        result['base_contract_sha256'] = ctx.get('base_contract_sha256')
        result['revised_contract_sha256'] = ctx.get('revised_contract_sha256')
        result['exam_retry_count']=len(ctx.get('exam_retry_events',[]))
        result['exam_reader_model']=reader_model
        result['exam_reader_url']=reader_url
        if result.get('exam_failed',0):
            result['judge_valid']=False
            result['rubric_valid']=False
            result['judge_failure']='exam_extraction_failed'
        result['record_id']=rid  # present on success AND failure, including validation
        captured=ctx['captured']
        if captured is not None:
            c,judged,response=captured
            record={'record_id':rid,'sample_id':c['sample_id'],'pair_id':c['pair_id'],
                    'side':c['side'],'response':response,'phi_version':phi['version'],
                    'phi_sha256':hashlib.sha256(raw).hexdigest(),
                    'configuration_version':judged.get('configuration_version'),
                    'rubric_hash':judged.get('rubric_hash'),
                    'reward_mode':os.environ.get('ACRE_REWARD_MODE', 'legacy_scalar'),
                    'criteria':judged['criteria'],
                    'rubric_meta':[{**m,'points':r['points'],'tags':r.get('tags',[])}
                                   for r,m in zip(c['rubrics'],c['rubric_meta'])],
                    'global_claim_support':judged['global_claim_support'],
                    'holistic_score':judged['holistic_score'],
                    'exam_retry_events':ctx.get('exam_retry_events',[]),**result}
            d=Path(trace_dir);d.mkdir(parents=True,exist_ok=True)
            temp=d/(rid+'.tmp');dest=d/(rid+'.json')
            with temp.open('x') as f:
                json.dump(record,f,ensure_ascii=False)
                f.flush();os.fsync(f.fileno())
            os.replace(temp,dest)
        if not result.get('judge_valid') or result.get('exam_failed', 0):
            raise RuntimeError('Invalid MetaRubrics grade; refusing policy update: ' + str(result.get('judge_failure')))
        return result
    finally:
        CTX.reset(token)
