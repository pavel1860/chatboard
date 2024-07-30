from typing import Dict, List
from pydantic import BaseModel
from chatboard.text.llms.conversation import validate_msgs, AIMessage, HumanMessage, ActionMessage, SystemMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from chatboard.text.llms.prompt_tracer import PromptTracer


def get_message(run_data):
    print(run_data)
    if run_data['role'] == 'user':
        return HumanMessage(content=run_data['content'])
    if run_data['role'] == 'assistant':
        tool_calls = None
        if run_data['additional_kwargs'].get('tool_calls'):
            tool_calls = [ChatCompletionMessageToolCall(**tc) for tc in run_data['additional_kwargs']['tool_calls']]
            # tool_calls = run_data['additional_kwargs']['tool_calls']
        return AIMessage(content=run_data['content'], tool_calls=tool_calls)
    if run_data['role'] == 'system':
        return SystemMessage(content=run_data['content'])
    else:
        raise ValueError(f"Unknown role: {run_data['role']}")
    
    
class LsLlmRun(BaseModel):
    name: str
    run_type: str
    messages: list = []
    
    def __getitem__(self, key):
        return self.messages[key]
    
    def __repr__(self) -> str:
        return self.show()
    
    def show(self, idx=0, tabs=0):
        return f"{'\t' * tabs}{idx}. {self.name} - {self.run_type}\n" + "\n".join([f"{'\t' * (tabs + 2)}{i}. {m.role}" for i, m in enumerate(self.messages)])
        


class LsRun(BaseModel):
    name: str
    run_type: str
    children: List['LsRun'] = []
    
    def __getitem__(self, key):
        return self.children[key]
        
    def get_names(self):
        return [c.name for c in self.children]
    
    def __repr__(self) -> str:
        return self.show()
    
    def show(self, idx=None, tabs=0 ):
        idx_str = f"{idx}. " if idx is not None else ""
        return f"{'\t' * tabs}{idx_str}{self.name} - {self.run_type}\n" + "\n".join([f"{c.show(i, tabs+1)}" for i, c in enumerate(self.children)])

    

def get_run_messages(run):
    try:        
        if run.run_type == "llm":
            print(run.name)
            ls_llm_run = LsLlmRun(name=run.name, run_type=run.run_type)
            messages = [get_message(m['data']) for m in run.inputs['messages']]
            print("----output----")
            messages.append(get_message(run.outputs['generations'][0]['message']['data']))            
            ls_llm_run.messages = messages
            return ls_llm_run    
        ls_run = LsRun(name=run.name, run_type=run.run_type)
        if run.child_runs:
            for child_run in run.child_runs:
                ls_child = get_run_messages(child_run)
                ls_run.children.append(ls_child)
        return ls_run
    except Exception as e:
        print (e)
        


async def get_run(name: str, error: bool=True):
    trancer = PromptTracer()
    runs = await trancer.aget_runs(name=name, error=error)
    run = await trancer.aget_run(runs[0].id)
    ls_run = get_run_messages(run)    
    return ls_run