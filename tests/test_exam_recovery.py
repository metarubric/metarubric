import asyncio,json,types,unittest
import httpx
from openai import AsyncOpenAI
from unittest.mock import patch,AsyncMock
from metarubrics.legacy import resilient_exam as m

class Reply:
 def __init__(self,value,status=200):self.value=value;self.status=status
 @property
 def model(self):return self.value.get('model','exam-qwen3-1.7b')
 def model_dump(self,**kw):return self.value

def answer(value='0',finish='stop'):
 return {'choices':[{'message':{'content':value},'finish_reason':finish}]}
class Session:
 def __init__(self,replies):
  self.replies=list(replies);self.calls=[]
  self.chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
 async def create(self,**body):
  self.calls.append(body);reply=self.replies.pop(0)
  if isinstance(reply.value,Exception):raise reply.value
  if reply.status in (408,429,500,502,503,504):raise m.TransientExtractorError(f'extractor HTTP {reply.status}')
  if reply.status!=200:raise RuntimeError(f'extractor HTTP {reply.status}')
  return reply
base=types.SimpleNamespace(EXAM_EXTRACT_SYSTEM='fixture',_exam_render=lambda i,r:r,_exam_index=lambda t,n:int(t))
item={'options':['a','b'],'points':1,'key':0}
class Recovery(unittest.IsolatedAsyncioTestCase):
 async def test_openai_sdk_serializes_local_reader_request(self):
  captured={}
  async def handler(request):
   captured.update(json.loads(request.content))
   return httpx.Response(200,json={'id':'chatcmpl-test','object':'chat.completion','created':0,'model':'exam-qwen3-1.7b','choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'0'}}]})
  sdk=AsyncOpenAI(api_key='local-vllm',base_url='http://127.0.0.1:19331/v1',max_retries=0,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
  result=await m.make_exam_score(base,lambda e:None)(sdk,'http://127.0.0.1:19331','exam-qwen3-1.7b','fixture',[item])
  await sdk.close()
  self.assertEqual(result,(1.,0));self.assertEqual(captured['max_tokens'],512);self.assertEqual(captured['chat_template_kwargs'],{'enable_thinking':False})
 async def test_local_reader_rejects_wrong_served_model(self):
  session=Session([Reply({**answer(), 'model':'exam-qwen3-4b'})])
  with self.assertRaisesRegex(RuntimeError,'exam extraction failed'):
   await m.make_exam_score(base,lambda e:None)(session,'http://fixture','exam-qwen3-1.7b','fixture',[item])
 async def run_score(self,replies):
  events=[];session=Session(replies)
  with patch.object(m.asyncio,'sleep',new=AsyncMock()):
   result=await m.make_exam_score(base,events.append)(session,'http://fixture','gpt-5.4-mini','fixture',[item])
  return result,session,events
 async def test_timeout_is_retried_before_scoring(self):
  r,s,e=await self.run_score([Reply(asyncio.TimeoutError()),Reply(answer())]);self.assertEqual(r,(1.,0));self.assertEqual(len(s.calls),2)
 async def test_503_is_retried(self):
  r,s,e=await self.run_score([Reply({},503),Reply(answer())]);self.assertEqual(r,(1.,0))
 async def test_permanent_http_error_never_becomes_reward(self):
  with self.assertRaises(RuntimeError):await self.run_score([Reply({},401)])
 async def test_malformed_answer_never_becomes_reward(self):
  with self.assertRaises(RuntimeError):await self.run_score([Reply(answer('bad'))])
 async def test_valid_answer_is_not_replaced(self):
  r,s,e=await self.run_score([Reply(answer('1'))]);self.assertEqual(r,(0.,0));self.assertEqual(len(s.calls),1)
 async def test_truncation_retry_preserved(self):
  r,s,e=await self.run_score([Reply(answer('bad','length')),Reply(answer())]);self.assertEqual(r,(1.,0));self.assertEqual(s.calls[1]['max_tokens'],2048)

class RewardGuard(unittest.IsolatedAsyncioTestCase):
 async def test_exam_client_closes_when_extraction_raises(self):
  from metarubrics import reward
  client=types.SimpleNamespace(close=AsyncMock())
  class Response:
   status=200
   async def __aenter__(self):return self
   async def __aexit__(self,*args):pass
   async def json(self,**kwargs):return {}
  class HttpSession:
   async def __aenter__(self):return self
   async def __aexit__(self,*args):pass
   def post(self,*args,**kwargs):return Response()
  contract={'sample_id':'sample','prompt':'question',
            'rubrics':[{'criterion':'criterion','points':1}],
            'exam_items':[{'stem':'question','options':['a','b'],'key':0,'points':1}]}
  from metarubrics.revisions import initial_snapshot
  token=reward.CTX.set({'phi':initial_snapshot(dataset='healthbench'),'split':'train'})
  try:
   with patch.object(reward.base,'AsyncOpenAI',return_value=client), \
        patch.object(reward.base.aiohttp,'ClientSession',return_value=HttpSession()), \
        patch.object(reward.base,'_exam_score',new=AsyncMock(side_effect=RuntimeError('invalid extraction'))):
    with self.assertRaisesRegex(RuntimeError,'invalid extraction'):
     await reward.base.compute_score('acre_healthbench_twin','answer',contract,
                                     evaluator_url='http://127.0.0.1:19790',
                                     exam_extractor_url='http://127.0.0.1:19331',
                                     exam_extractor_model='exam-qwen3-1.7b',max_retries=1)
  finally:
   reward.CTX.reset(token)
  client.close.assert_awaited_once()

 async def test_legacy_4b_reader_alias_is_rejected_by_default(self):
  import tempfile,json
  from pathlib import Path
  from metarubrics import reward
  from metarubrics.revisions import initial_snapshot
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'phi.json';p.write_text(json.dumps(initial_snapshot(dataset='healthbench')))
   with self.assertRaisesRegex(RuntimeError,'wrong exam reader configuration'):
    await reward.compute_score(data_source='acre_healthbench_twin',extra_info={'split':'train'},phi_snapshot=str(p),trace_dir=td,exam_extractor_model='exam-qwen3-4b',exam_extractor_url='http://127.0.0.1:19331')
 async def test_invalid_grade_cannot_return_to_trainer(self):
  import tempfile,json
  from pathlib import Path
  from metarubrics import reward
  from metarubrics.revisions import initial_snapshot
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'phi.json';p.write_text(json.dumps(initial_snapshot(dataset='healthbench')))
   with patch.object(reward.base,'compute_score',new=AsyncMock(return_value={'judge_valid':False,'exam_failed':0,'judge_failure':'TimeoutError'})):
    with self.assertRaisesRegex(RuntimeError,'refusing policy update'):
     await reward.compute_score(data_source='acre_healthbench_twin',extra_info={'split':'train'},phi_snapshot=str(p),trace_dir=td,exam_extractor_model='exam-qwen3-1.7b',exam_extractor_url='http://127.0.0.1:19331')

if __name__=='__main__':unittest.main()
