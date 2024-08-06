
import json
from functools import wraps
from typing import Any, List, Literal, Tuple, Union
from uuid import uuid4

# from chatboard.text.llms.mvc import BaseModel, Field
from pydantic import BaseModel, Field

ViewWrapperType = Literal["xml", "markdown", None]
BaseModelRenderType =  Literal['model_dump', 'json']
class ViewNode(BaseModel):
    vn_id: str = Field(default_factory=lambda: str(uuid4()), description="id of the view node")
    name: str = Field(None, description="name of the view function")
    title: str | None = None
    numerate: bool = False
    base_model: BaseModelRenderType = 'json'
    wrap: ViewWrapperType = None
    role: Literal["assistant", "user", "system"] | None = None
    # views: List[Union["ViewNode", BaseModel, str]] | Tuple[Union["ViewNode", BaseModel, str]] | "ViewNode" | BaseModel | str 
    views: Any
    actions: List[BaseModel] | BaseModel | None = None
    
    
    
    

    
def view(
    container=None, 
    title=None, 
    actions=None, 
    role=None,
    numerate=False,
    base_model: BaseModelRenderType = 'json',
    wrap: ViewWrapperType = None
    ):

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
                base_model=base_model,
                numerate=numerate,
                wrap=wrap,
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

def list_view(rules: list[str], numbered: bool = True):
    if numbered:
        return "\n".join([f"{i}. {r}" for i, r in enumerate(rules)])
    else:
        return "\n".join(rules)



def render_tabs(num: int):
    return "ֿ\t" * num

def render_model(model: BaseModel, node: ViewNode, i: int, tabs: int = 0):
    prompt = f"{render_tabs(tabs)}"
    if node.numerate:
        prompt += f"{i + 1}. "
        
    if node.base_model == 'json':
        return prompt + json.dumps(model.model_dump(), indent=2)
    elif node.base_model == 'model_dump':
        return prompt + str(model.model_dump()) + "\n"
    else:
        raise ValueError(f"base_model type not supported: {node.base_model}")


def render_string(string: str, node: ViewNode, i:int, tabs: int = 0):
    prompt = f"{render_tabs(tabs)}"
    if node.numerate:
        prompt += f"{i + 1}. "
    prompt += string + "\n"
    return prompt


def render_title(title: str, node: ViewNode, tabs: int = 0):
    if node.wrap == "xml":
        return f"{render_tabs(tabs)}<{title}>\n"
    if node.wrap == "markdown":
        return f"{render_tabs(tabs)}## {title}\n"
    return f"{render_tabs(tabs)}{title}:\n"

def render_ending(title: str, node: ViewNode, tabs: int = 0):
    if node.wrap == "xml":
        return f"{render_tabs(tabs)}</{title}>\n"
    return ''



#? in render view we are using 2 stacks so that we can render the views in the correct order
# ?is a view is between 2 strings, we want to render the view between the strings
def render_view(
    node: ViewNode | Tuple[ViewNode], 
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
    if type(node) == tuple:
        stack1 = [*node]    
    else:
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


        

    while stack2:
        node = stack2.pop()
        prompt = ""
        if node.title:
            # prompt += render_tabs(len(stack2)) + f"### {node.title}\n"
            prompt += render_title(node.title, node, tabs=len(stack2))
        for i, view in enumerate(_iterate_views(node)):
            if isinstance(view, str):
                # prompt += tabs(len(stack2)) + f"{view}\n"
                prompt += render_string(view, node, i, tabs=len(stack2))
            elif isinstance(view, ViewNode):
                prompt += render_tabs(len(stack2)) + rendered_outputs[view.vn_id]
            elif isinstance(view, BaseModel):
                # prompt += render_tabs(len(stack2)) + str(view.model_dump(mode="json")) + "\n"
                prompt += render_model(view, node, i, tabs=len(stack2))
            else:
                raise ValueError(f"view type not supported: {type(view)}")
        if node.title:
            prompt += render_ending(node.title, node, tabs=len(stack2))
        rendered_outputs[node.vn_id] = prompt
        
    # final_prompt = "\n".join(rendered_outputs.values())
    final_prompt = prompt
    return final_prompt, rendered_outputs, base_models





