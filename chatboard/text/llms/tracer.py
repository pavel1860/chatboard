





import os
import traceback
from typing import Dict, Optional

from langsmith import RunTree


# class RunTree(ls_schemas.RunBase):
# outputs: Optional[Dict] = None
# error: Optional[str] = None,
# end_time: Optional[datetime] = None,
# events: Optional[Sequence[ls_schemas.RunEvent]] = None,
class Tracer:

    def __init__(self, name, inputs, run_type="chain", extra={}, tracer_run=None, is_traceable=True):
        self.is_traceable = is_traceable
        self.tracer_run = None
        if not self.is_traceable:
            return
        if os.environ.get("LANGCHAIN_API_KEY") is None:
            self.is_traceable = False
            return
        if type(tracer_run) == Tracer:
            tracer_run = tracer_run.tracer_run

        if tracer_run is not None:
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

    @property
    def id(self):
        if not self.is_traceable:
            return
        return self.tracer_run.id

    def __enter__(self):
        return self

    def end(self, outputs: Optional[Dict]=None, errors: Optional[str]=None):
        if not self.is_traceable:
            return
        self.tracer_run.end(outputs=outputs, error=errors)


    def end_documents(self, documents, errors: Optional[str]=None):
        if not self.is_traceable:
            return
        self.tracer_run.end(outputs={"documents": documents}, error=errors)


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.is_traceable:
            return False
        if exc_type is not None:
            traceback_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            traceback_string = "".join(traceback_lines)
            self.tracer_run.end(
                error= f"Error: {str(exc_value)}\n   Traceback:\n{traceback_string}",
            )
        self.tracer_run.post()
        return False
    
    def create_child(self, name, inputs, run_type="chain", extra={}):
        if not self.is_traceable:
            return None
        return Tracer(
            name=name,
            run_type=run_type,
            inputs=inputs,
            extra=extra,
            tracer_run=self
        )