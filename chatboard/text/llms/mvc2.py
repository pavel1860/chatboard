
from typing import Any, Literal, Tuple, Union, List
from functools import wraps
# from chatboard.text.llms.mvc import BaseModel, Field
from pydantic import BaseModel, Field
from uuid import uuid4

class ViewNode(BaseModel):
    vn_id: str = Field(default_factory=lambda: str(uuid4()), description="id of the view node")
    name: str = Field(None, description="name of the view function")
    title: str | None = None
    role: Literal["assistant", "user", "system"] | None = None
    # views: List[Union["ViewNode", BaseModel, str]] | Tuple[Union["ViewNode", BaseModel, str]] | "ViewNode" | BaseModel | str 
    views: Any
    actions: List[BaseModel] | BaseModel | None = None
    
    
    
    

    
def view(container=None, title=None, actions=None, role=None):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):                            
            outputs = func(*args, **kwargs)
            if container is not None:
                outputs = container(*outputs if isinstance(outputs, tuple) else (outputs,))
            view_instance = ViewNode(
                name=func.__name__,
                title=title,
                views=outputs,
                actions=actions,
                role=role,
            )
            return view_instance            
            # outputs = func(*args, **kwargs)
            # view_instance = ViewNode(
            #     name=func.__name__,
            #     title=title,
            #     views=outputs,
            #     actions=actions
            # )
            # if container is not None:
            #     view_instance = container(view_instance)
            # return view_instance
        return wrapper
    
    return decorator


# def view(container=None, title=None, actions=None):

#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):            
#             outputs = func(*args, **kwargs)
#             view_instance = ViewNode(
#                 name=func.__name__,
#                 title=title,
#                 views=outputs,
#                 actions=actions
#             )            
#             return view_instance
#         return wrapper
    
#     if container is not None:
#         return decorator(container)
    
#     return decorator



def render_view(
    view, 
    **kwargs):


    def _iterate_views(node, is_reversed=False):
        if isinstance(node.views, list) or isinstance(node.views, tuple):
            if is_reversed:
                for view in reversed(node.views):
                    yield view
            else:
                for view in node.views:
                    yield view
        else:
            yield node.views

    
    # node = view(**kwargs)
    node = view

    stack1 = [node]
    stack2 = []

    base_models = {}

    rendered_outputs = {}
    for i in range(10):
        if not stack1:
            break
        
        node = stack1.pop()
        stack2.append(node)
        for view in _iterate_views(node, is_reversed=True):
            if isinstance(view, ViewNode):
                stack1.append(view)
            elif isinstance(view, BaseModel):
                base_models[view.__class__.__name__] = view
            if isinstance(view, list) or isinstance(view, tuple):
                for v in view:
                    if isinstance(v, ViewNode):
                        stack1.append(v)

    def tabs(num: int):
        return "  " * num
        

    while stack2:
        node = stack2.pop()
        prompt = ""
        if node.title:
            prompt += tabs(len(stack2)) + f"### {node.title}\n"
        for view in _iterate_views(node):
            if isinstance(view, str):
                prompt += tabs(len(stack2)) + f"{view}\n"
            elif isinstance(view, ViewNode):
                prompt += tabs(len(stack2)) + rendered_outputs[view.vn_id]
            elif isinstance(view, BaseModel):
                prompt += tabs(len(stack2)) + str(view.model_dump()) + "\n"
            else:
                raise ValueError(f"view type not supported: {type(view)}")
        rendered_outputs[node.vn_id] = prompt
    return prompt, rendered_outputs, base_models