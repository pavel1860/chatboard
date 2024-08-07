
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
    index: int | None = None
    actions: List[BaseModel] | BaseModel | None = None
    depth: int = 0
    
    def get_type(self):
        return type(self.views)
    
    def has_wrap(self):
        return self.wrap is not None or self.title is not None
    
    def is_leaf(self):
        return self.get_type() == str or issubclass(self.get_type(), BaseModel)
    
    def __hash__(self):
        return self.vn_id.__hash__()
    
    
    
    

    
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
                
            sub_views = []
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                for i, o in enumerate(outputs):
                    if isinstance(o, str):
                        sub_views.append(
                            ViewNode(
                                name=f"{func.__name__}_str_{i}",
                                views=o,
                                numerate=numerate,
                                index=i,
                                role=role
                            )   
                        )
                    elif isinstance(o, ViewNode):
                        sub_views.append(o)
                    elif isinstance(o, BaseModel):
                        sub_views.append(
                            ViewNode(
                                name=f"{func.__name__}_model_{i}",
                                views=o,
                                numerate=numerate,
                                base_model=base_model,
                                index=i,
                                role=role
                            )
                        )
                    else:
                        raise ValueError(f"view type not supported: {type(o)}")
            else:
                sub_views = outputs
            view_instance = ViewNode(
                name=func.__name__,
                title=title,
                views=sub_views,
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

def add_tabs(content: str, tabs: int):
    return "\n".join([render_tabs(tabs) + c for c in content.split("\n")])
    # return content.replace("\n", f"\n{render_tabs(tabs)}")


def render_model(node: ViewNode):
    model = node.views
    prompt = ""
    if node.numerate:
        prompt += f"{node.index + 1}. "
        
    if node.base_model == 'json':
        return add_tabs(prompt + json.dumps(model.model_dump(), indent=2), node.depth)
    elif node.base_model == 'model_dump':
        return add_tabs(prompt + str(model.model_dump()) + "\n", node.depth)
    else:
        raise ValueError(f"base_model type not supported: {node.base_model}")


def render_string(node: ViewNode):
    prompt = ''
    depth = node.depth + 1 if node.has_wrap() else node.depth
    if node.numerate:
        prompt += f"{node.index + 1}. "
    prompt += node.views
    return add_tabs(prompt, depth)



def add_wrapper(content: str, node: ViewNode):
    title = node.title if node.title is not None else ''
    if node.wrap == "xml":
        return add_tabs((
            f"<{title}>\n"
            f"\n{content}"
            f"</{title}>\n"   
        ), node.depth)
    
    if node.wrap == "markdown":
        return add_tabs((
            f"## {title}\n"
            f"\t{content}\n"
        ), node.depth)
    return add_tabs((
        f"{title}:"
        f"\t{content}"
        , node.depth)
    )


    
def render_wrapper_starting(node: ViewNode):
    title = node.title if node.title is not None else ''
    if node.wrap == "xml":
        return add_tabs(f"<{title}>", node.depth)
    elif node.wrap == "markdown":
        return add_tabs(f"## {title}", node.depth)
    return add_tabs(f'{title}:', node.depth)

def render_wrapper_ending(node: ViewNode):
    title = node.title if node.title is not None else ''
    if node.wrap == "xml":
        return add_tabs(f"</{title}>", node.depth)
    return ''



#? in render view we are using 2 stacks so that we can render the views in the correct order
# ?is a view is between 2 strings, we want to render the view between the strings
def render_view(node: ViewNode, **kwargs):

    if type(node) == tuple:
        stack = [*node]    
    else:
        stack = [node]

    base_models = {}
    visited = set()
    result = []
    while stack:
        peek_node = stack[-1]
                            
        if peek_node not in visited:
            visited.add(peek_node)
            if peek_node.has_wrap():
                result.append(render_wrapper_starting(peek_node))
            if peek_node.get_type() == str:
                result.append(render_string(peek_node))
            elif peek_node.get_type() == list or peek_node.get_type() == tuple:
                for view in reversed(peek_node.views):
                    # if peek_node.has_wrap():
                    #     view.depth = peek_node.depth + 1
                    # if view.is_leaf():
                    #     if peek_node.has_wrap():
                    #         view.depth = peek_node.depth + 2
                    #     else:
                    #         view.depth = peek_node.depth + 1
                    # else:
                    #     if peek_node.has_wrap():
                    #         view.depth = peek_node.depth + 1
                    if peek_node.has_wrap():
                        view.depth = peek_node.depth + 1
                    else:
                        view.depth = peek_node.depth
                    
                    stack.append(view)
            elif issubclass(peek_node.get_type(), BaseModel):
                base_models[peek_node.views.__class__.__name__] = peek_node.views
                result.append(render_model(peek_node))
            else:
                raise ValueError(f"view type not supported: {type(view)}")
        else:
            if peek_node.has_wrap():
                result.append(render_wrapper_ending(peek_node))
            stack.pop(-1)
    prompt = "\n".join(result)
    return prompt, result, base_models