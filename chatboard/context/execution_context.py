

import contextvars
from typing import Any
from ..tracer.langsmith_tracer import RunTypes
from ..tracer import Tracer


CURR_CONTEXT = contextvars.ContextVar("curr_context")



class ExecutionContext:
    
    
    def __init__(
        self, 
        span_name: str | None = None, 
        request_id: str | None = None,
        manager_phone_number: str | None = None,
    ):
        self.tracer_run = None
        self.parent_ctx = None
        self.span_name = span_name
        self.request_id = request_id
        self.manager_phone_number = manager_phone_number
    
    @property
    def tracer(self):
        if self.tracer_run is None:
            raise ValueError("Tracer not set")
        return self.tracer_run
    
    @property
    def trace_id(self):
        return str(self.tracer.id)
    
    @classmethod
    def current(cls)-> "ExecutionContext":
        try:            
            ctx = CURR_CONTEXT.get()
            return ctx
        except LookupError:            
            raise ValueError("Context not set")               
        
    
    @classmethod
    def current_or_none(cls)-> "ExecutionContext | None":
        try:            
            ctx = CURR_CONTEXT.get()
            return ctx
        except LookupError:
            return None        
    
    def _set_context(self):
        self._ctx_token = CURR_CONTEXT.set(self)
        
        
    def _reset_context(self):
        if self._ctx_token is not None:
            CURR_CONTEXT.reset(self._ctx_token)
        self._ctx_token = None

    def build_child(self, span_name: str | None = None):
        child = ExecutionContext(span_name=span_name, request_id=self.request_id)
        child.parent_ctx = self
        return child
        
    def start_tracer(self, name: str, run_type: RunTypes = "prompt", inputs: dict[str, Any] | None = None):
        self.tracer_run = Tracer(
            name=name,
            run_type=run_type,
            inputs=inputs,
            is_traceable=True,
            tracer_run=self.parent_ctx._tracer_run if self.parent_ctx is not None else None,
        )
        return self
    
    
    async def __aenter__(self):    
        # if self._branch is None or self._turn is None:
            # raise ValueError("Branch or turn not set")
        self._set_context()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        self._reset_context()
        if self.tracer_run is not None:
            self.tracer_run.__exit__(exc_type, exc_value, traceback)
        if exc_type is not None:
            return False        
        return True
