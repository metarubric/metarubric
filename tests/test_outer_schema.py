import json,tempfile,types,unittest
import httpx
from openai import AsyncOpenAI
from pathlib import Path
from unittest.mock import AsyncMock,patch
from outer_boundary import Client,OPENAI_API_BASE_URL,parse_grade
class Reply:
 def __init__(self,parsed):self.parsed=parsed
 model='gpt-5.4-mini'
 choices=None
 def model_dump(self,**kw):return {'model':self.model,'choices':[{'finish_reason':'stop','message':{'content':json.dumps(self.parsed)}}]}
 def __post_init__(self):pass
class Session:
 def __init__(self,items):
  self.items=list(items);self.calls=0
  self.chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
 async def create(self,**kwargs):
  self.calls+=1;reply=Reply(self.items.pop(0))
  reply.choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=json.dumps(reply.parsed)))]
  return reply
class Schema(unittest.IsolatedAsyncioTestCase):
 def test_remote_endpoint_is_official_openai(self):
  self.assertEqual(OPENAI_API_BASE_URL,'https://api.openai.com/v1')
 async def test_openai_sdk_serializes_outer_request(self):
  captured={}
  async def handler(request):
   captured.update(json.loads(request.content))
   return httpx.Response(200,json={'id':'chatcmpl-test','object':'chat.completion','created':0,'model':'gpt-5.4-mini','choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'judgments':[{'id':0,'met':True}]})}}]})
  with tempfile.TemporaryDirectory() as td:
   sdk=AsyncOpenAI(api_key='test-key',base_url='https://api.openai.com/v1',max_retries=0,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
   result=await Client(sdk,Path(td)).ask('panel','fixture',{},validator=lambda d:parse_grade(d,1))
   await sdk.close()
   self.assertTrue(result['judgments'][0]['met']);self.assertIn('max_completion_tokens',captured);self.assertEqual(captured['reasoning_effort'],'none')
 async def test_invalid_cached_panel_is_not_reused(self):
  bad={'judgments':[{'id':0,'met':True}]};good={'judgments':[{'id':0,'met':True},{'id':1,'met':False}]}
  with tempfile.TemporaryDirectory() as td:
   session=Session([bad,good]);client=Client(session,Path(td));await client.ask('fixed-reference-panel','fixture',{})
   result=await client.ask('fixed-reference-panel','fixture',{},validator=lambda d:parse_grade(d,2))
   self.assertEqual(result,good);self.assertEqual(session.calls,2);self.assertTrue(list((Path(td)/'invalid').glob('*.json')))
 async def test_new_invalid_panel_retried_before_cache(self):
  bad={'judgments':[]};good={'judgments':[{'id':0,'met':False}]}
  with tempfile.TemporaryDirectory() as td,patch('outer_boundary.asyncio.sleep',new=AsyncMock()):
   s=Session([bad,good]);c=Client(s,Path(td));result=await c.ask('panel','fixture',{},validator=lambda d:parse_grade(d,1));self.assertEqual(result,good);self.assertEqual(s.calls,2)
 async def test_valid_cached_label_never_resampled(self):
  good={'judgments':[{'id':0,'met':False}]}
  with tempfile.TemporaryDirectory() as td:
   s=Session([good]);c=Client(s,Path(td))
   for _ in range(2):self.assertEqual(await c.ask('panel','fixture',{},validator=lambda d:parse_grade(d,1)),good)
   self.assertEqual(s.calls,1)
if __name__=='__main__':unittest.main()
