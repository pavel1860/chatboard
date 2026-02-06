from chatboard.llms.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ActionMessage

def load_messages(raw_messages):
  messages = []
  for raw_msg in raw_messages:
    if raw_msg['role'] == 'assistant':
      messages.append(AIMessage(**raw_msg))
    elif raw_msg['role'] == 'user':
      messages.append(HumanMessage(**raw_msg))
    elif raw_msg['role'] == 'tool':
      messages.append(ActionMessage(**raw_msg))
    elif raw_msg['role'] == 'system':
      messages.append(SystemMessage(**raw_msg))
    else:
      raise ValueError(f"Unsupported role: {raw_msg['role']}")
  return messages
  

test_pirate_messages = [{'id': None,
  'content': 'Sir yes Sir!',
  'content_blocks': None,
  'name': None,
  'is_example': False,
  'is_history': False,
  'is_output': False,
  'model': None,
  'did_finish': True,
  'role': 'assistant',
  'run_id': None,
  'tool_calls': None,
  'raw': None},
 {'id': None,
  'content': 'Hi there matey, give me a quest',
  'content_blocks': None,
  'name': None,
  'is_example': False,
  'is_history': False,
  'is_output': False,
  'role': 'user'},
 {'id': 'chatcmpl-A8Q6GJHwUoVd7hnXkkeobLpejCgMx',
  'content': None,
  'content_blocks': None,
  'name': None,
  'is_example': False,
  'is_history': False,
  'is_output': False,
  'model': 'gpt-4o-2024-05-13',
  'did_finish': True,
  'role': 'assistant',
  'run_id': None,
  'tool_calls': None,
  'raw': {'id': 'chatcmpl-A8Q6GJHwUoVd7hnXkkeobLpejCgMx',
   'choices': [{'finish_reason': 'stop',
     'index': 0,
     'logprobs': None,
     'message': {'content': None,
      'refusal': None,
      'role': 'assistant',
      'function_call': None,
      'tool_calls': [{'id': 'call_arbaVaHBI5U8GRyREPcsZQLb',
        'function': {'arguments': '{"description":"Retrieve the lost treasure from Skeleton Island.","task":"Find the hidden map in the abandoned fort and use it to navigate to Skeleton Island. Once there, defeat the undead guardians and bring back the treasure.","reward":200}',
         'name': 'give_quest_action'},
        'type': 'function'}]}}],
   'created': 1726570316,
   'model': 'gpt-4o-2024-05-13',
   'object': 'chat.completion',
   'service_tier': None,
   'system_fingerprint': 'fp_25624ae3a5',
   'usage': {'completion_tokens': 57,
    'prompt_tokens': 716,
    'total_tokens': 773,
    'completion_tokens_details': {'reasoning_tokens': 0}}}},
 {'id': 'call_arbaVaHBI5U8GRyREPcsZQLb',
  'content': '{"name":"Jack Sparrow","health":100,"mana":30,"strength":8,"reputation":50}',
  'content_blocks': None,
  'name': None,
  'is_example': False,
  'is_history': False,
  'is_output': False,
  'role': 'tool'}]