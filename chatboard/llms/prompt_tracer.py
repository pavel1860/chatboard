import asyncio
import json
import os
from typing import Any, Dict, List, Union

from langsmith import Client
from langsmith.schemas import Feedback, Run
from ..llms.messages import AIMessage, HumanMessage, SystemMessage

LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "default")




class PromptRun:

    def __init__(self, run: Run, is_prompt=False):
        self.run = run
        self.children: Any = []
        self.llm_run: Any = None
        self.prompt_run: Any = None
        self.sequence_run: Any = None
        self.is_prompt: Any = is_prompt


    def __getitem__(self, index):
        return self.children[index]
    

    def to_string(self):
        def tree_walk(run, depth):
            print(f'{" " * (depth * 2 )}{run.name}')
            for child in run.children:
                tree_walk(child, depth + 1)
        tree_walk(self, 0)


    def html_reper(self):
        return self.get_html()
    
    def show_html(self, system=False):
        from IPython.display import HTML, display
        display(HTML(self.get_html(system=system)))


    @property
    def user_message(self):
        if self.llm_run is not None:
            for msg in self.llm_run.inputs['messages']:
                if 'HumanMessage' in msg['id']:
                    return msg['kwargs']['content']
        return None
    
    @property
    def system_message(self):
        if self.llm_run is not None:
            for msg in self.llm_run.inputs['messages']:
                if 'SystemMessage' in msg['id']:
                    return msg['kwargs']['content']
        return None
    
    @property
    def generation_message(self):        
        if self.llm_run is not None:
            for msg in self.llm_run.outputs['generations']:
                return msg["message"]["kwargs"]["content"]
        return None
    

    def get_messages(self):
        messages = []
        for msg in self.llm_run.inputs['messages']:
            if 'HumanMessage' in msg['id']:
                messages.append(HumanMessage(content=msg['kwargs']['content']))
            elif 'SystemMessage' in msg['id']:
                messages.append(SystemMessage(content=msg['kwargs']['content']))
           
        for msg in self.llm_run.outputs['generations']:
            messages.append(AIMessage(content=msg["message"]["kwargs"]["content"]))
        return messages

    def get_html(self, system=False, font_size=8):
    # def html_show(self):
        def tree_walk(run, html_text, depth, idx):
            html_text += f'<div style="margin-left: {20*depth}px; font-size: {font_size}px;">'
            html_text +=f'<h3>{idx} {run.name}</h3>'
            if run.llm_run is not None:
                if depth == 0:
                    model_name = run.llm_run.outputs["llm_output"]["model_name"]
                    # html_text += f'<div style="margin-left: {20*(depth + 1)}px">{model_name} llm output</div>'
                    # input_txt = self.llm_input.replace("\n", "<br>")
                    # html_text += f'<p style="margin-left: {20*(depth + 1)}px">{input_txt}</p>'
                    # output_txt = self.llm_output.replace("\n", "<br>")
                    # html_text += f'<p style="margin-left: {20*(depth + 1)}px">{output_txt}</p>'
                    for msg in run.llm_run.inputs['messages']:
                        message_tag= ""
                        if 'HumanMessage' in msg['id']:
                            message_tag = """<span style="background-color: blue; color: white; padding: 5px; border-radius: 5px; margin-top: 10px">Human</span>"""
                        elif 'SystemMessage' in msg['id']:
                            if not system:
                                continue
                            message_tag = """<span style="background-color: purple; color: white; padding: 5px; border-radius: 5px; margin-top: 10px">System</span>"""

                        msg_text = msg['kwargs']['content'].replace("\n", "<br>")
                        html_text += f"""
                        <div style="margin-left: {20*(depth + 1)}px; border: solid 1px; border-radius: 10px; margin-top: 10px; padding: 5px">
                            {message_tag}
                            <p >{msg_text}</p>
                        </div>
                        """
                        # print(msg)
                    html_text += f'<div style="margin-left: {20*(depth + 1)}px"><h3>llm output</h3> {model_name} </div>'
                    for msg in run.llm_run.outputs['generations']:
                        output_text = msg["message"]["kwargs"]["content"].replace("\n", "<br>")
                        html_text += f"""
                        <div style="margin-left: {20*(depth + 1)}px; border: solid 1px; border-radius: 10px; margin-top: 10px; padding: 5px">
                            <span style="background-color: red; color: white; padding: 5px; border-radius: 5px; margin-top: 10px">AI</span>
                            <p>{output_text}</p>
                        </div>
                        """
                else:
                    html_text += f'<div style="margin-left: {20*(depth + 1)}px">llm output</div>'
                # for gen in self.llm_run.outputs['generations']:
                    # text = gen["text"].replace("\n", "<br>")
                    # html_text += f'<p style="margin-left: {20*(depth + 1)}px">{text}</p>'
            html_text+="</div>"
            for idx, child in enumerate(run.children):
                html_text = tree_walk(child, html_text, depth + 1, idx)
            return html_text
        return tree_walk(self, "", 0, 0)

        # return f'<div>{self.run.name}</div>'
    
    # def append_child(self, child: "PromptRun"):
    #     if child.run_type == "llm":
    #         self.llm_run = child
    #     elif child.run_type == "prompt":
    #         self.prompt_run = child
    #     self.children.append(child)

    def append(self, child: "PromptRun"):
        self.children.append(child)
    
    def _repr_html_(self):
        return self.html_reper()
    
    @property
    def start_time(self):
        return self.run.start_time
    
    @property
    def end_time(self):
        return self.run.end_time
    
    @property
    def start_time_str(self):
        return self.run.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    @property
    def id(self):
        return self.run.id
    
    @property
    def name(self):
        return self.run.name
    
    @property
    def run_type(self):
        if self.run.name == "RunnableSequence":
            return "prompt"
        return self.run.run_type
    
    @property
    def duration(self):
        if self.run.end_time is None:
            return None
        return self.run.end_time - self.run.start_time
    
    @property
    def total_tokens(self):
        return self.run.data.metrics.total_tokens
    
    def get_feedback_stats(self):
        return self.run.feedback_stats
    
    def get_parent_Ids(self):
        return [str(p) for p in self.run.parent_run_ids]#type: ignore
    
    @property
    def metadata(self):
        
        if 'metadata' in self.run.extra:#type: ignore
            return self.run.extra['metadata']#type: ignore
        return {}
    
    @property
    def llm_output(self):
        if self.llm_run is not None:
            output = ""
            for gen in self.llm_run.outputs['generations']:
                output += gen['text']
            return output
    
    @property
    def llm_input(self):
        if self.llm_run is not None:
            text = ''
            for msg in self.llm_run.inputs['messages']:
                text += msg['kwargs']['content']
            return text

    
    @property
    def output(self):        
        return self.run.outputs['output']['content'] #type: ignore
    
    @property
    def prompt_name(self):
        prompt_name = self.metadata.get('prompt', None)
        prompt_commit = self.metadata.get('commit', None)
        return f'{prompt_name}:{prompt_commit}'
    
    def to_json(self):
        # run_dict = self.run.dict()
        # run_dict['id'] = str(run_dict['id'])
        # return run_dict
        run_json = json.loads(self.run.json())
        run_json['children'] = [c.to_json() for c in self.children]            
        return run_json
    


class PromptRunTree:

    def __init__(self, parent: Run, children: List[Run]):
        self.parent = parent
        self.children_lookup = {c.id: c for c in children}
        for run in children:
            if run.run.child_run_ids:
                for c_id in run.run.child_run_ids:
                    run.append_child(self.children_lookup[c_id])
        self.children = [c for c in children if parent.id == c.run.parent_run_id]

    

    def html_reper(self):
        return f'''
<div>{self.parent.name} {self.parent.run_type}</div>
{"".join([f"<div>{c.name} {c.run_type}</div>" for c in self.children])}
'''
    @property
    def duration(self):
        return self.parent.duration
    
    @property
    def id(self):
        return self.parent.id
    
    def _repr_html_(self):
        return self.html_reper()
    
    @property
    def inputs(self):
        return self.parent.run.inputs
    
    @property
    def outputs(self):
        return self.parent.run.outputs
    
    # def get_top_level(self):
    #     return [c for c in self.children if self.parent.id == c.run.parent_run_id]
    

    def to_json(self):
        return {
            'run': self.parent.to_json(),
            'states': [c.to_json() for c in self.children]
        }


def add_filter(filters, new_filter):
    if filters is None:
        return new_filter
    return f'and({filters}, {new_filter})'


class PromptTracer:

    def __init__(self) -> None:
        self.client = Client()
        


    def get_run_list(
            self, 
            name: List[str] | str | None = None,             
            execution_order=None, 
            filters=None, 
            project_name=LANGCHAIN_PROJECT, 
            session_id=None,
            limit=10,
            error: bool=None,
            is_root=True
        ):
        runs = []        
        active_filter = None
        if name is not None:
            if type(name) != list:
                name = [name] #type: ignore
            name_filters = []
            for n in name: #type: ignore
                name_filters.append(f'eq(name, "{n}")')
            active_filter = ",".join(name_filters)
            if len(name_filters) > 1:
                active_filter = f'or({active_filter})'
        # if session_id is not None:
        
            
        if filters is not None:
            if active_filter is not None:
                active_filter = f'and({active_filter}, {filters})'
            else:
                active_filter = filters
                
        if session_id:
            active_filter = add_filter(active_filter, f"and(eq(metadata_key, 'session_id'), eq(metadata_value, '{session_id}'))")                  
            
        for run in self.client.list_runs(
            project_name=project_name,
            filter=active_filter,
            execution_order=execution_order,
            error=error,
            limit=limit,
            is_root=is_root
            # filter='or(eq(name, "AgentExecutor"), eq(name, "RouterChain"), eq(name, "AvoConversationalRetrievalChain"))',                    
        ):
            runs.append(PromptRun(run))
            if len(runs) > limit:
                break
        return runs
    
    async def aget_runs(
            self, 
            name: List[str] | str | None = None, 
            execution_order=None, 
            session_id=None,
            filter=None, 
            project_name=LANGCHAIN_PROJECT, 
            limit=10,
            error: bool=None,
            is_root=True
        ):
        return await asyncio.get_running_loop().run_in_executor(
            None, self.get_run_list, name, execution_order, filter, project_name, session_id, limit, error, is_root
        )


    async def aget_run(self, run_id: str):
        return await asyncio.get_running_loop().run_in_executor(
            None, self.get_run, run_id, True
        )

    def get_run(self, run_id: str, output_raw=False):
        # run = self.client.read_run(run_id)
        # child_runs = list(self.client.list_runs(
        #     project_name="default",
        #     run_ids=run.child_run_ids
        # ))
        lc_run = self.client.read_run(run_id, load_child_runs=True)
        if output_raw:
            return lc_run
        
        # def tree_walk(parent_run):
        #     if parent_run.run.name == "RunnableSequence":
        #         for child in parent_run.run.child_runs:
        #             if child.run_type == "prompt":
        #                 parent_run.prompt_run = child
        #             if child.run_type == "llm":
        #                 parent_run.llm_run = child
        #         return False
                
        #     for child in parent_run.run.child_runs:
        #         chiled_run = PromptRun(child)
        #         tree_walk(chiled_run)
        #         parent_run.append(chiled_run)
        def tree_walk(parent_run):
            if parent_run.run.name == "RunnableSequence":                
                return False
            if parent_run.run.child_runs is None:
                return False
                # return False

            for child in parent_run.run.child_runs:
                # if child.name == "RunnableSequence":
                #     parent_run.is_prompt = True
                #     parent_run.sequence_run = child
                #     for leaf in child.child_runs:
                #         if leaf.run_type == "prompt":
                #             parent_run.prompt_run = leaf
                #         if leaf.run_type == "llm":
                #             parent_run.llm_run = leaf
                #     return
                # print(child.run_type)
                if child.run_type == "llm":
                    parent_run.llm_run = child
                elif child.run_type == "chain":                    
                    chiled_run = PromptRun(child)
                    tree_walk(chiled_run)
                    parent_run.append(chiled_run)
        # child_runs.sort(key=lambda x: x.execution_order, reverse=False)
        # child_runs.sort(key=lambda x: x.start_time)
        run = PromptRun(lc_run)
        tree_walk(run)
        return run
        # return PromptRunTree(PromptRun(run), [PromptRun(r) for r in  child_runs])
    
    async def aget_run_states(self, run_id: str):
        return await asyncio.get_running_loop().run_in_executor(
            None, self.get_run_states, run_id#type: ignore
        )
    
    def delete_run(self, run_id: str):
        self.client.unshare_run(run_id)

    def adelete_run(self, run_id: str):
        return asyncio.get_running_loop().run_in_executor(
            None, self.delete_run, run_id
        )

    def list_feedback(self, run_ids: List[str] | str | None=None) -> List[Feedback]:
        if type(run_ids) != list:
            run_ids = [run_ids]#type: ignore
        
        feedback_list = []
        for f in  self.client.list_feedback(run_ids=run_ids):
            feedback_list.append(f)
        return feedback_list
    
    async def alist_feedback(self, run_ids: List[str] | str | None=None) -> List[Feedback]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.list_feedback, run_ids
        )

    
    def create_feedback(
            self, 
            run_id: str, 
            key: str, 
            score: Union[float , int , bool , None] = None, 
            value: Union[float , int , bool , str , dict , None] = None,
            correction: Union[str, dict, None] = None,
            comment: Union[str, None] = None,
            source_info: Union[Dict[str, Any] , None] = None,
            # feedback_source_type: str = "API",

        ):
        """
        Parameters
        ----------
        score: float , int , bool , None 
            The score to rate this run on the metric or aspect.
        value: float , int , bool , str , dict , None
            The display value or non-numeric value for this feedback.
        correction: 
            The correct ground truth for this run.
        comment: str, None
            A comment about this feedback, or additional reasoning about why it received this score.
        source_info: Dict[str, Any] , None
            Information about the source of this feedback (e.g., a user ID, model type, tags, etc.)
        feedback_source_type: "API", "MODEL" 
            The type of feedback source, either API or MODEL.
        """
        return self.client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            value= value,
            correction=correction,#type: ignore
            comment= comment,
            source_info= source_info,
            # feedback_source_type= feedback_source_type,
        )
    
    async def acreate_feedback(
            self, 
            run_id: str, 
            key: str, 
            score: Union[float , int , bool , None] = None, 
            value: Union[float , int , bool , str , dict , None] = None,
            correction: Union[str, dict, None] = None,
            comment: Union[str, None] = None,
            source_info: Union[Dict[str, Any] , None] = None,
            # feedback_source_type: str = "API",

        ):
        return await asyncio.get_running_loop().run_in_executor(
            None, self.create_feedback, run_id, key, score, value, correction, comment, source_info
        )
    

    def update_feedback(
        self, 
        feedback_id: str, 
        key: str, 
        score: Union[float , int , bool , None] = None, 
        value: Union[float , int , bool , str , dict , None] = None,
        correction: dict[Any, Any] | None = None,
        comment: Union[str, None] = None,
        source_info: Union[Dict[str, Any] , None] = None,
        feedback_source_type: str = "API",
    ):       
        return self.client.update_feedback(
            feedback_id=feedback_id,
            # key=key,
            score=score,
            value= value,
            correction= correction,
            comment= comment,
            # source_info= source_info,
            # feedback_source_type= feedback_source_type,
        )
    
    async def aupdate_feedback(
        self, 
        feedback_id: str, 
        # key: str, 
        score: Union[float , int , bool , None] = None, 
        value: Union[float , int , bool , str , dict , None] = None,
        correction: Union[str, dict, None] = None,
        comment: Union[str, None] = None,
        # source_info: Union[Dict[str, Any] , None] = None,
        # feedback_source_type: str = "API",
    ):       
        return await asyncio.get_running_loop().run_in_executor(
            # None, 
            self.update_feedback, 
            feedback_id, #type: ignore
            # key, 
            score, 
            value, 
            correction, 
            comment, 
            # source_info, 
            # feedback_source_type
        )
    
    def delete_feedback(
        self, 
        feedback_id: str
    ):
        return self.client.delete_feedback(
            feedback_id=feedback_id
        )
    

    def delete_all_feedback(self, run_id: str):
        feedback_list = self.list_feedback(run_id)
        for f in feedback_list:
            self.delete_feedback(f.id)#type: ignore

    def tag(self, run_id: str, tags: List[str] | str):
        if type(tags) == str:
            tags = [tags]
        return self.client.update_run(
            run_id=run_id,
            tags=tags #type: ignore
        )

    def delete_project(self, project_name: str):
        self.client.delete_project(project_name=project_name)



    def get_run_step(self, run_id: str, step_idx: int, pydantic_model):
        run = self.get_run(run_id)        
        step_run = run.run.child_runs[step_idx] #type: ignore
        return {}
        