import asyncio
import copy
from enum import Enum
import json
from typing import Any, List, Optional
from langchain_openai import ChatOpenAI
import pydantic as pyd
from langsmith import RunTree
import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, validator
from .chat_prompt import ChatPrompt, ChatResponse, ChatChunk, validate_input_variables
from .completion_parsing import num_split_field, split_field
# from components.etl.system_conversation import AIMessage, Conversation, ConversationRag
from .conversation import HumanMessage, SystemConversation, AIMessage, Conversation, ConversationRag, from_langchain_message
from .rag_manager import RagVectorSpace
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

from .tracer import Tracer



class AgentChunkTypes(Enum):
    AGENT_START = "agent_start"
    AGENT_UPDATE = "agent_update"
    AGENT_FINISH = "agent_finish"
    AGENT_ERROR = "agent_error"



class AgentChunk(BaseModel):
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    msg_type: AgentChunkTypes
    func: Optional[str]
    iteration: Optional[int]
    prompt_chunk: Optional[ChatChunk]
    agent_response: Optional[ChatResponse]

    class Config:
        arbitrary_types_allowed = True


    def to_dict(self):
        return {
            "msg_type": self.msg_type.value,
            "func": self.func,
            "iteration": self.iteration,
            "prompt_chunk": self.prompt_chunk.to_dict(),
            "agent_response": self.agent_response.to_dict() if self.agent_response is not None else None,
        }


class ViewAgentResponse(BaseModel):
    run_id: str
    state: Any

# class AgentResponse(BaseModel):
#     value: Any
#     run_id: str
#     # costs: Optional[Dict[str, Any]] = None
#     model: Optional[str] = None

#     def to_dict(self):
#         if hasattr(self.value, 'to_dict') and callable(getattr(self.value, 'to_dict')):
#             value = self.value.to_dict()
#         else:
#             value = self.value
#         return {
#             "value": value,
#             "run_id": self.run_id,
#             "costs": self.costs,
#             "model": self.model
#         }


class FinishAction(BaseModel):
    """a finish action to indicate you are satisfied with the result that there is nothing else to do and all the frames are ready."""
    # to_finish: bool = Field(..., description="a boolean to indicate that the action is finished.")
    reason: str = Field(..., description="the reason for the finish action.")


class FinishActionException(Exception):
    pass



class ShortTermMessageMemory:

    def __init__(self, memory_length=5):
        self.conversation = []
        self.memory_length = memory_length

    def add_message(self, message):
        self.memory.append(message)

    def get_memory(self):
        return self.memory[-self.memory_length:]



class LongTermRagMessageMemory:
    pass



def get_tool_scheduler_prompt(tool_dict):
    tool_function = tool_dict['function']
    prompt = f"""{tool_function["name"]}: {tool_function["description"]}\n\tparameters:"""
    for prop, value in tool_function["parameters"]['properties'].items():
        prompt += f"\n\t\t{prop}: {value['description']}"
    return prompt



class AgentStep(BaseModel):
    state: Any
    action: Any






class AgentAction(BaseModel):
    _output: Optional[str] = PrivateAttr("")
    _success: Optional[bool | None] = PrivateAttr(None)

    def success(self, msg):
        self._success = True
        self._output = msg
        return self
    
    def failure(self, msg):
        self._success = False
        self._output = msg
        return self
    

def conversation_factory() -> Conversation:
    return Conversation()


class AgentState(BaseModel):
    _actions: List[AgentAction] = PrivateAttr([])
    _conversation: Conversation = PrivateAttr(default_factory=conversation_factory)
    
    def _last_success(self):
        return self._actions[-1]._success
    
    def message(self):
        if len(self._conversation) == 0:
            return None
        return self._conversation[-1]
    
    def action(self):
        if len(self._actions) == 0:
            return None
        return self._actions[-1]
    
    def _did_finish(self):
        return type(self._actions[-1]) == FinishAction
    
    def to_dict(self):
        d = self.dict()
        d['_actions'] = [a.dict() for a in self._actions]
        d['_conversation'] = self._conversation.to_dict()
        return d
# class AgentRunContext:
    
#     def __init__(self, state):
#         self.state = state
#         self.conversation = Conversation()
#         self.actions = []
    

# https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models

class ViewAgent:

    def __init__(
            self, 
            promptpath,
            context,
            reducer,
            # state,
            tools, 
            views,            
            name="ChatAgent",
            system_filename=None, 
            user_filename=None, 
            save_rag=False, 
            rag_index=None,
            rag_length=0,
            model=None,
            stop_sequences=None,
            logit_bias=None,
            max_memory_length=4,
            max_iterations=10,
            tracer_run=None, 
            max_actions=1,
            rag_vectorizer=None,
            rag_model=None,
            rag_diverse=None,
            seed=None,
            **kwargs           
        ):
        self.context = context
        self.tools = tools + [FinishAction]
        # self.tools = tools 
        if not name:
            raise ValueError("ChatAgent must have a name")
        
        self.views = views
        self.name = name
        self.reducer = reducer
        # self.state_history = [AgentStep(state=state, action=None)]
        
        self.system_filename = system_filename
        self.user_filename = user_filename
        self.model = model
        self.prompt = ChatPrompt(
            promptpath=promptpath,
            name=name,
            system_filename=system_filename,
            user_filename=user_filename,
            model=self.model,
            stop_sequences=stop_sequences,
            logit_bias=logit_bias,
            seed=seed,
        )
        self.max_iterations = max_iterations
        self.save_rag = save_rag  
        # self.conversation = Conversation()
        self.rag_index = rag_index
        self.rag_length = rag_length
        self.rag_diverse = rag_diverse
        self.rag_space = RagVectorSpace(
            rag_index or name,
            rag_vectorizer,
            rag_model
        )
        self.max_memory_length = max_memory_length
        self.max_actions = max_actions

        self.tracer_run = tracer_run
        self.kwargs = kwargs


    @property
    def state(self):
        return self.state_history[-1].state
    
    
    def _render_view(self, state, **kwargs):
        render_view_kwargs = {}
        for view in self.views:
            render_view_kwargs[view.__name__] = view(state)
        render_view_kwargs.update(self.kwargs)
        render_view_kwargs.update(kwargs)
        return render_view_kwargs
    
    def _render_view_prompt(self, state, **kwargs):
        render_view_kwargs = self._render_view(state, **kwargs)
        prompt_user_template, prompt_metadata = self.prompt.prompt_manager.get_template(
            is_chat_template=True, 
            template_type="user", 
            filename=kwargs.get("user_filename", self.user_filename)
        )
        validate_input_variables(render_view_kwargs, prompt_metadata['input_variables'])
        langchain_user_message = prompt_user_template.format(**render_view_kwargs)
        user_message = from_langchain_message(langchain_user_message)
        return user_message
    
    async def next_step(self, state, prompt=None, step_index=None, input_postfix=None, system_postfix=None, tracer_run=None, retry=3, **kwargs): 
        for i in range(retry):
            try:
                tool_descriptions = [get_tool_scheduler_prompt(convert_to_openai_tool(t)) for t in self.tools]

                render_view_kwargs = self._render_view(state, **kwargs)

                conversation = state._conversation

                system_conversation = await self.prompt.build_conversation(
                    prompt=prompt,
                    conversation=conversation,            
                    tool_names=[tool.__name__ for tool in self.tools],
                    tools=tool_descriptions,
                    input_postfix=input_postfix,
                    system_postfix=system_postfix,
                    **render_view_kwargs
                )

                examples = None
                if self.rag_space and self.rag_length:
                    examples = await self.rag_space.similarity(state, self.rag_length, alpha=0.5, use_diverse=self.rag_diverse)
                    example_messages = []
                    for i, e in enumerate(examples):
                        user_message = self._render_view_prompt(e.metadata.state, **kwargs)
                        ai_message = AIMessage(**e.metadata.message.dict())                
                        example_messages += self.render_example(user_message, ai_message, i)

                    system_conversation.add_examples(example_messages)

                openai_messages = system_conversation.to_openai()
                
                llm_response = await self.prompt.llm.send(
                    openai_messages=openai_messages,
                    tracer_run=tracer_run,
                    **kwargs,
                )

                llm_output = llm_response.choices[0].message

                parsed_response = await self.prompt.parse_completion(
                    output=llm_output,
                    **kwargs
                )        

                llm_tool_completion, tool_calls = await self.parse_tool(llm_output.content, tracer_run=tracer_run)

                ai_message = AIMessage(content=llm_output.content, tool_calls = tool_calls)

                system_conversation.append(ai_message)        

                llm_response.choices[0].message
                state._conversation = system_conversation.conversation.copy()
                return state, tool_calls
            except Exception as e:
                print("Step Error:", e)
                if i < retry:
                    print("retrying...")
                    continue
                else:
                    raise e
                # return ChatResponse(
                #     value=parsed_response,
                #     run_id=str(step_run.id),
                #     conversation=conversation,
                #     tools=tool_calls
                # )
    
    def render_example(self, human_message: HumanMessage, ai_message: AIMessage, idx):
        human_message.content = f"""
EXAMPLE {idx}:
{human_message.content}
"""
        ai_message.content = f"""
EXAMPLE {idx}:
{ai_message.content}
"""
        return [human_message, ai_message]
        

    async def handle_action(self, state, tools, tracer_run, **kwargs):
        # check_for_finish_action(agent_output)
        need_to_finish = False
        for action in tools:
            print("Tool:", action)
            if type(action) == FinishAction:
                need_to_finish = True
                # raise FinishActionException()
        # if "FinishAction" in agent_output.value:
            # raise FinishActionException()
        is_success = None
        for action in tools[:self.max_actions]:            
            with tracer_run.create_child(
                name=type(tools[0]).__name__,
                run_type="tool",
                inputs={
                    "tool": tools,
                }
            ) as tool_run:
                
                if type(action) == FinishAction:
                    tool_run.end(outputs={
                        "state": state,
                        "action_output": "finished",
                    })
                    state._actions.append(action)
                    return state
                state, action = await self.reducer(self.context, state, action, tracer_run=tool_run)
                state._actions.append(action)
                if action._success is None:
                    raise ValueError("Action must have a success value")
                if is_success is None or is_success is True:
                    is_success = action._success
                # self.append_action_output(action_output)
                state._conversation[-1].content = state._conversation[-1].content + "Action_Output: " + action._output
                # print("State:", state)
                tool_run.end(outputs={
                    # "state": state.to_dict(),
                    "action_output": action._output,
                })
        return state
        
    
    async def run(self, state, tracer_run=None, **kwargs):
        async for state in self.step_generator(state=state, tracer_run=tracer_run, **kwargs):
            pass
        return state
    

    async def step_generator(self, state, tracer_run=None, **kwargs):
        with Tracer(
            tracer_run=tracer_run,
            name=self.name,
            run_type="chain",
            inputs={
                "state": state.to_dict(),
                "kwargs": kwargs,
            },
            extra={
                "rag_index": self.rag_index,
                "system_filename": self.system_filename,
                "user_filename": self.user_filename,
            }
        ) as agent_run:
            state.run_id = str(agent_run.id)
            state._conversation = Conversation()
            did_stop=False
            
            for i in range(self.max_iterations):
                with Tracer(
                    tracer_run=agent_run,
                    name=f"{self.name}Step{str(i)}",
                    run_type="prompt",
                    inputs={
                        "state": state.to_dict(),
                    },
                    extra={
                        "system_filename": self.system_filename, 
                        "user_filename": self.user_filename,
                        "rag_index": self.rag_index,
                    }
                ) as step_run:
                    state, tools = await self.next_step(state, tracer_run=step_run, **kwargs)
                    state = await self.handle_action(state, tools, tracer_run=step_run, **kwargs)

                    step_run.end(outputs={
                        "state": state.to_dict(),
                    })
                    if state._did_finish():
                        did_stop = True
                        break
                    yield state
            
            agent_run.end(outputs={
                "state": state.to_dict(),
                "did_stop": did_stop,
            })
            yield state


    async def runner_generator2(self, **kwargs):
        print("#####", self.name)
        with Tracer(
            tracer_run=self.tracer_run,
            name=self.name,
            run_type="chain",
            inputs={
                "state": self.state.to_dict(),
                "kwargs": kwargs,
            },
            extra={
                "rag_index": self.rag_index,
                "system_filename": self.system_filename,
                "user_filename": self.user_filename,
            }
        ) as agent_run:
            
            # state = context.store.get_init_state(str(agent_run.id), **kwargs)        
            # state_history = [state.copy()]
            state = copy.deepcopy(self.state)

            try:
                for i in range(self.max_iterations):
                    print(f"iteration: {i}")            
                    agent_output = await self.next_step(
                        state=state, 
                        step_index=i,
                        tracer_run=agent_run,
                        **kwargs
                    )
                    # context.store.push(str(agent_run.id), state, agent_output.value, agent_output.tools)
                    check_for_finish_action(agent_output)
                    for action in agent_output.tools:
                        if type(action) == FinishAction:
                            raise FinishActionException()
                        with agent_run.create_child(
                            name=type(agent_output.tools[0]).__name__,
                            run_type="tool",
                            inputs={
                                "tool": agent_output.tools,
                            }
                        ) as tool_run:
                            state, action_output = await self.reducer(self.context, state, action, tracer_run=tool_run)
                            self.append_action_output(action_output)
                            # print("State:", state)
                            tool_run.end(outputs={
                                "state": state,
                                "action_output": action_output,
                            })
                            state.step = i
                            self.state_history.append(AgentStep(state=state, action=action))
                            yield state
                            # yield ViewAgentResponse(
                            #     run_id=str(agent_run.id),
                            #     state=state,
                            # )
                else:
                    did_stop = False
            except FinishActionException:
                did_stop = True

            self.run_id = str(agent_run.id)
            agent_run.end(outputs={
                "state": state.to_dict(),
                "did_stop": did_stop,
                # "evaluation": evaluation,
            })
            self.state_history.append(AgentStep(state=state, action=action))
            yield state


    async def get_examples(self, conversation, rag_length, state):
        return await self.rag_space.similarity(conversation, rag_length)
        

    async def parse_tool(self, content, tracer_run=None):        
        # split_res = num_split_field("Action", content, maxsplit=1)
        split_res = split_field("Action", content)
        if split_res is not None and len(split_res) > 1:
            content = "Action:" + split_res[1]

        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        llm_with_tools = llm.bind_tools(self.tools)
        
        tool_chain = llm_with_tools | PydanticToolsParser(tools=self.tools)


        for _ in range(3):
            try:
                tool_calls = list(await tool_chain.ainvoke(content))
                if len(tool_calls) == 1:
                    return None, tool_calls
            except Exception as e:
                print("tool parsing error:", e)
                continue
        else:
            raise Exception("Failed to parse the action")

        print("tool_calls", tool_calls)

        return None, tool_calls

    # async def parse_tool(self, content, tracer_run=None):
    #     return await self.call_parse_tool(content, self.tools, tracer_run=tracer_run)
    

    async def call_parse_tool_llm(self, content, tools, tool_choice=None, tracer_run=None):
        content = await self.parcer(content)
        # tool_conv = Conversation([conversation[0]])
        tool_conv = Conversation()
        tool_conv.append(
            AIMessage(
                content=content
            )
        )
        # tool_choice = None
        # for tool in self.tools:
        #     if tool.__name__ in tool_conv.messages[-1].content:
        #         tool_choice = tool.__name__
        #         break

        llm_tool_completion, tool_calls = await self.prompt.call_llm_with_tools(
            # conversation=conversation,
            conversation=tool_conv,
            tools=tools,
            tool_choice=tool_choice,
            tracer_run=tracer_run
        )
        return llm_tool_completion, tool_calls


    async def parcer(self, completion, context = None):
        return completion

    # async def parcer(self, completion, context = None):
    #     split_res = num_split_field("Action", completion, maxsplit=1)
    #     if split_res is not None and len(split_res) > 1:
    #         completion = "Action:" + split_res[1]

    #     return completion
    
    
    async def reducer(self, context, state, action: BaseModel = None, run_manager=None):
        # if action.type == "StockMedia":
        return state
    

    def append_action_output(self, action_output: str):        
        self.conversation[-1].content = self.conversation[-1].content + "Action_Output: " + action_output


    








# class HistoryStep:

#     def __init__(self, state, aciton) -> None:
#         self.states = []
#         self.actions
        







def check_for_finish_action(agent_output):
    for action in agent_output.tools:
        print("Tool:", action)
        if type(action) == FinishAction:
            raise FinishActionException()
    if "FinishAction" in agent_output.value:
        raise FinishActionException()





class AgentRunner:


    def __init__(self, context, agent, state, reducer, name="Agent", max_iterations=10, rag_index=None, tracer_run=None, **kwargs):
        self.agent = agent
        # self.state = state
        self.reducer = reducer
        self.name = name
        self.max_iterations = max_iterations
        self.tracer_run = tracer_run
        self.run_states = [state]
        self.context = context
        self.kwargs = kwargs
        self.rag_index = rag_index
        self.run_id = None


    @property
    def state(self):
        return self.run_states[-1]
    

    def history(self, index):
        return self.run_states[index]


    async def run_loop(self):
        async for state in self.runner_generator():
            self.run_states.append(copy.deepcopy(state))
        return state    
        

    async def run(self):
        async for state in self.runner_generator():
            pass
        return self
        # return await self.post_run(self.run_states[-1])

    async def runner_generator(self):
        print("#####", self.name)
        with Tracer(
            tracer_run=self.tracer_run,
            name=self.name,
            run_type="chain",
            inputs={
                "state": self.state,
                "kwargs": self.kwargs,
            },
            extra={
                "rag_index": self.rag_index,
                "system_filename": self.kwargs.get("system_filename", None),
                "user_filename": self.kwargs.get("user_filename", None),
            }
        ) as agent_run:
            
            # state = context.store.get_init_state(str(agent_run.id), **kwargs)        
            # state_history = [state.copy()]
            state = copy.deepcopy(self.state)

            try:
                for i in range(self.max_iterations):
                    print(f"iteration: {i}")            
                    agent_output = await self.agent.next_step(
                        state=state, 
                        step_index=i,
                        tracer_run=agent_run,
                        **self.kwargs
                    )
                    # context.store.push(str(agent_run.id), state, agent_output.value, agent_output.tools)
                    check_for_finish_action(agent_output)
                    for action in agent_output.tools:
                        if type(action) == FinishAction:
                            raise FinishActionException()
                        with agent_run.create_child(
                            name=type(agent_output.tools[0]).__name__,
                            run_type="tool",
                            inputs={
                                "tool": agent_output.tools,
                            }
                        ) as tool_run:
                            state, action_output = await self.reducer(self.context, state, action, tracer_run=tool_run)
                            self.agent.append_action_output(action_output)
                            # print("State:", state)
                            tool_run.end(outputs={
                                "state": state,
                                "action_output": action_output,
                            })
                            state.step = i
                            self.run_states.append(AgentStep(state, action))
                            yield state
                            # yield ViewAgentResponse(
                            #     run_id=str(agent_run.id),
                            #     state=state,
                            # )
                else:
                    did_stop = False
            except FinishActionException:
                did_stop = True

            evaluation = await self.evaluate(state)
            self.run_id = str(agent_run.id)
            agent_run.end(outputs={
                "state": state,
                "did_stop": did_stop,
                "evaluation": evaluation,
            })
            self.run_states.append(AgentStep(state, action))
            yield state
            # yield ViewAgentResponse(
            #     run_id=str(agent_run.id),
            #     state=state,
            # )
            

    async def evaluate(self, state):
        return state

    async def post_run(self, state):
        return state
