import unittest
from metarubrics import reward
from test_exam_recovery import Session,Reply,answer,base,item
class FormatRetry(unittest.IsolatedAsyncioTestCase):
 async def test_out_of_range_reply_gets_format_retry(self):
  # Exercise strict real parser: 2 is outside two 0-based options.
  fake=type('Base',(),{'EXAM_EXTRACT_SYSTEM':'fixture','_exam_render':staticmethod(lambda i,r:r),'_exam_index':staticmethod(reward.strict_exam_index)})
  s=Session([Reply(answer('2')),Reply(answer('1'))]);events=[]
  result=await reward.retry_module.make_exam_score(fake,events.append)(s,'http://fixture','gpt-5.4-mini','fixture',[item])
  self.assertEqual(result,(0.,0));self.assertEqual(len(s.calls),2)
  self.assertIn('0, 1',s.calls[1]['messages'][-1]['content'])
  self.assertTrue(events[0]['recovered'])
 async def test_persistent_invalid_reply_is_not_a_score(self):
  s=Session([Reply(answer('bad')) for _ in range(3)])
  with self.assertRaises(RuntimeError):await reward.retry_module.make_exam_score(base,lambda e:None)(s,'http://fixture','gpt-5.4-mini','fixture',[item])
  self.assertEqual(len(s.calls),3)
if __name__=='__main__':unittest.main()
