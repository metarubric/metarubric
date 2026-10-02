"""Retry transient transport failures and one truncation; never score failed extraction."""
import asyncio
import openai

class TransientExtractorError(RuntimeError):
    pass


def make_exam_score(base, on_retry, retry_max_tokens=2048):
    async def score(client, url, model, response, items):
        async def request(item, limit, previous_reply=None):
            body = {'model': model, 'max_tokens': limit, 'temperature': 0.,
                    'reasoning_effort': 'low',
                    'messages': [{'role': 'system', 'content': base.EXAM_EXTRACT_SYSTEM},
                                 {'role': 'user', 'content': base._exam_render(item, response)}]}
            if previous_reply is not None:
                allowed = ', '.join(str(i) for i in range(len(item['options'])))
                body['messages'].extend([
                    {'role':'assistant','content':previous_reply[:2048]},
                    {'role':'user','content':'Your previous output was not a valid option number. Using the SAME candidate answer and options above, return exactly one of these 0-based integers: '+allowed+'. Return only the integer, without explanation. Do not infer unstated facts.'}])
            extra_body = None
            if model == 'exam-qwen3-1.7b':
                body.pop('reasoning_effort', None)
                extra_body = {
                    'chat_template_kwargs': {'enable_thinking': False},
                }
            attempt = 0
            while True:
                try:
                    completion = await client.chat.completions.create(
                        **body, extra_body=extra_body)
                    if model == 'exam-qwen3-1.7b' and completion.model != model:
                        raise RuntimeError('unexpected exam reader model')
                    return completion.model_dump(mode='json')
                except openai.APIStatusError as error:
                    if error.status_code not in (408, 429, 500, 502, 503, 504):
                        raise
                    delay = min(2 ** min(attempt, 6), 60)
                    attempt += 1
                    on_retry({'reason': 'transient_transport', 'attempt': attempt,
                              'error_type': type(error).__name__, 'delay_seconds': delay,
                              'max_tokens': limit})
                    await asyncio.sleep(delay)
                except (TransientExtractorError, asyncio.TimeoutError,
                        openai.APIConnectionError, openai.APITimeoutError,
                        openai.RateLimitError) as error:
                    delay = min(2 ** min(attempt, 6), 60)
                    attempt += 1
                    on_retry({'reason': 'transient_transport', 'attempt': attempt,
                              'error_type': type(error).__name__, 'delay_seconds': delay,
                              'max_tokens': limit})
                    await asyncio.sleep(delay)

        def parse(data, item):
            value = base._exam_index(data['choices'][0]['message']['content'], len(item['options']))
            if value is None:
                raise ValueError('invalid extractor option index')
            return value

        async def one(index, item):
            limit = 512
            previous_reply = None
            pending = []
            for attempt in range(3):
                data = await request(item, limit, previous_reply)
                try:
                    value = parse(data, item)
                    for event in pending:event['recovered'] = True
                    return value
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    choices = data.get('choices') or [{}]
                    choice = choices[0]
                    previous_reply = str((choice.get('message') or {}).get('content') or '')
                    truncated = choice.get('finish_reason') == 'length'
                    event = {'item_index':index, 'reason':'truncated_invalid_index' if truncated else 'invalid_option_index',
                             'attempt':attempt+1, 'initial_max_tokens':limit,
                             'retry_max_tokens':retry_max_tokens if truncated else limit,
                             'finish_reason':choice.get('finish_reason'),
                             'reader_model':model, 'returned_content':previous_reply,
                             'candidate_answer':response, 'question':item.get('stem',''),
                             'options':item['options'], 'recovered':False,
                             'retry_scheduled':attempt<2}
                    pending.append(event)
                    on_retry(event)
                    if attempt == 2:
                        raise RuntimeError('exam reader repeatedly returned an invalid option index') from error
                    if truncated:limit = retry_max_tokens

        answers = await asyncio.gather(*(one(i, item) for i, item in enumerate(items)), return_exceptions=True)
        errors = [answer for answer in answers if isinstance(answer, Exception)]
        if errors:
            raise RuntimeError(f'exam extraction failed: {type(errors[0]).__name__}') from errors[0]
        positive_mass = sum(abs(float(item['points'])) for item in items if float(item['points']) > 0)
        if positive_mass <= 0:
            return None, 0
        total, failed = 0., 0
        for answer, item in zip(answers, items):
            if isinstance(answer, Exception):
                failed += 1
                continue
            weight = abs(float(item['points']))
            correct = answer == int(item['key'])
            if float(item['points']) > 0:
                total += weight if correct else 0.
            else:
                total += 0. if correct else -weight
        return min(1., total / positive_mass), failed

    return score
