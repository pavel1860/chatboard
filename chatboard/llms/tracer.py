





import os
import traceback
from typing import Any, Dict, List, Literal, Optional

# from langsmith import RunTree
from langsmith.run_trees import RunTree

from ..llms.messages import BaseMessage, AIMessage, HumanMessage

# class RunTree(ls_schemas.RunBase):
# outputs: Optional[Dict] = None
# error: Optional[str] = None,
# end_time: Optional[datetime] = None,
# events: Optional[Sequence[ls_schemas.RunEvent]] = None,

RunTypes = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]

class Tracer:
    """
    Tracer class to trace the execution of the code.

    Args:
        name (str): Name of the tracer.
        inputs (Dict): Inputs to the tracer.
        run_type (str): Type of the run. "tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"
        session_id (str): Session id of the tracer.
        extra (Dict): Extra information to be logged.
        tracer_run (RunTree): Parent tracer run.
        is_traceable (bool): If the tracer is traceable or not
    """

    def __init__(
        self, 
        name: str, 
        inputs: Any = {}, 
        run_type: RunTypes="chain",
        session_id: str=None,
        extra: Dict[str, Any]={},
        metadata: Dict[str, Any]=None, 
        tracer_run=None, 
        is_traceable: bool | None=True, 
        tags: List[str] | str | None=None
    ):
        self.is_traceable = is_traceable
        self.tracer_run = None
        self.session_id = session_id
        self.outputs = {}
        if not self.is_traceable:
            return
        if os.environ.get("LANGCHAIN_API_KEY") is None:
            self.is_traceable = False
            return
        if type(tracer_run) == Tracer:
            tracer_run = tracer_run.tracer_run
        
        if metadata is not None:
            extra["metadata"] = metadata
        # if session_id is not None:
        #     extra["session_id"] = session_id

        if tracer_run is not None:
            session_id = tracer_run.metadata.get("session_id")
            self.tracer_run = tracer_run.create_child(
                name=name,
                run_type=run_type,
                inputs=inputs,
                extra=extra
                # extra=prompt_metadata,
            )
        else:
            self.tracer_run = RunTree(
                name=name,
                run_type=run_type,
                # inputs=log_kwargs,
                inputs=inputs,
                extra=extra
                # extra=prompt_metadata,
            )
        if session_id is not None:
            self.tracer_run.add_metadata({"session_id": session_id})
        
        if tags is not None:
            self.tracer_run.add_tags(tags)
            
        self.did_end = False

    @property
    def id(self):
        if self.is_traceable and self.tracer_run is not None:            
            return self.tracer_run.id

    def __enter__(self):
        return self

    def end(self, outputs: Any | None = None, errors: Optional[str]=None):
        if self.is_traceable and self.tracer_run is not None:
            self.tracer_run.end(outputs=outputs, error=errors)
            self.did_end = True
    
            
    def add_outputs(self, output: Any):
        if self.is_traceable and self.tracer_run is not None:
            if isinstance(output, AIMessage):                
                # output_dict = {}
                # messages = self.outputs.get("messages", [])
                # messages += [output]
                # self.outputs["messages"] = messages                
                # self.outputs["response"] = output.content
                if output.content:                    
                    self.outputs["response"] = output.content
                # if output.tool_calls:
                #     output_dict["tool_calls"] = output.tool_calls
                #     messages += [output_dict]
                    
                self.tracer_run.add_outputs(self.outputs)
            else:
                self.outputs.update(output)
                self.tracer_run.add_outputs(self.outputs)
        
        


    def end_documents(self, documents, errors: Optional[str]=None):
        if self.is_traceable and self.tracer_run is not None:
            self.tracer_run.end(outputs={"documents": documents}, error=errors)


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.is_traceable and self.tracer_run is not None:
            if exc_type is not None:
                traceback_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                traceback_string = "".join(traceback_lines)
                self.tracer_run.end(
                    error= f"Error: {str(exc_value)}\n   Traceback:\n{traceback_string}",
                )
            elif not self.did_end:
                self.tracer_run.end()
            self.tracer_run.post()
            return False
        else:
            return False
    
    def create_child(self, name, inputs: Any, run_type: RunTypes="chain", extra: Dict[str, Any]={}):
        if not self.is_traceable:
            return None
        session_id = self.tracer_run.metadata.get("session_id")
        return Tracer(
            name=name,
            run_type=run_type,
            inputs=inputs,
            extra=extra,
            session_id=session_id,
            tracer_run=self
        )
        
        
    def add_tags(self, tags):
        if not self.is_traceable or self.tracer_run is None:
            return
        self.tracer_run.add_tags(tags)