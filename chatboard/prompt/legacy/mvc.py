from __future__ import annotations

import json
import string
import textwrap
from functools import wraps
from typing import Any, Callable, Generator, List, Literal, Tuple, Type, Union
from uuid import uuid4

from chatboard.llms.messages import AIMessage, ActionCall, ActionMessage, BaseMessage, ContentType
from chatboard.llms.utils.action_manager import Actions
from chatboard.utils.string_utils import camel_to_snake
from pydantic import BaseModel, Field

ViewWrapperType = Literal["xml", "markdown", None]
BaseModelRenderType =  Literal['model_dump', 'json']
ListModelRender = Literal['list', 'view_node']

# class ContentBlock(BaseModel):
#     vn_id: str = Field(default_factory=lambda: str(uuid4()), description="id of the view node")
#     name: str = Field(None, description="name of the view function")
#     title: str | None = None
#     numerate: bool = False
#     base_model: BaseModelRenderType = 'json'
#     wrap: ViewWrapperType = None
#     role: Literal["assistant", "user", "system"] | None = "user"
#     role_name: str | None = None
#     # views: List[Union[ViewNode, BaseModel, str]] | Tuple[Union[ViewNode, BaseModel, str]] | ViewNode | BaseModel | str 
#     content_blocks: Any
#     index: int | None = None
#     actions: List[BaseModel] | BaseModel | None = None
#     depth: int = 0
#     indent: int | None = None
#     list_model: ListModelRender = 'view_node'
    
#     def get_type(self):
#         return type(self.content_blocks)
    
#     def has_wrap(self):
#         return self.wrap is not None or self.title is not None
    
#     def is_leaf(self):
#         return self.get_type() == str or issubclass(self.get_type(), BaseModel)
    
#     def __hash__(self):
#         return self.vn_id.__hash__()
RoleType = Literal["assistant", "user", "system", "tool"]
BulletType = Literal["number" , "astrix" , "dash" , "none", None] | str
StripType = Literal["left", "right"] | bool | None

def filter_by_tag(block: ViewBlock, tag: str | None) -> bool:
    return tag is None or block.tag == tag

def filter_by_role(block: ViewBlock, role: str | None) -> bool:
    return role is None or block.role == role

def filter_by_view_name(block: ViewBlock, view_name: str | None) -> bool:
    return view_name is None or block.view_name == view_name

def filter_by_class(block: ViewBlock, class_: str | None) -> bool:
    return class_ is None or block.class_ == class_

def filter_by_depth(depth: int, min_depth: int, max_depth: int) -> Callable[[ViewBlock, int], bool]:
    def inner(block: ViewBlock, current_depth: int) -> bool:
        if depth is not None:
            return current_depth == depth
        return min_depth <= current_depth <= max_depth
    return inner

def fitler_by_action_calls(block: ViewBlock, action_calls: bool | None) -> bool:
    return action_calls is None or bool(block.action_calls) == action_calls


def filter_by_ids(block: ViewBlock, uuids: list[str] | None) -> bool:
    return uuids is None or block.uuid in uuids

def combine_filters(
    tag: str | None = None, 
    role: str | None = None,
    view_name: str | None = None,
    class_: str | None = None,
    depth: int | None = None,
    min_depth: int = 0,
    max_depth: int = 100,
    action_calls: bool | None=None,
    uuids: list[str] | None = None
) -> Callable[[ViewBlock, int], bool]:
    
    def combined(block: ViewBlock, current_depth: int) -> bool:
        return (
            filter_by_tag(block, tag) and
            filter_by_role(block, role) and
            filter_by_view_name(block, view_name) and
            filter_by_class(block, class_) and
            filter_by_depth(depth, min_depth, max_depth)(block, current_depth) and 
            fitler_by_action_calls(block, action_calls) and 
            filter_by_ids(block, uuids)
        )
    
    return combined



class ViewBlock(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()), description="id of the view node")
    name: str | None = Field(None, description="name of the person who created the view")
    view_name: str
    title: str | None = None
    content: str | dict | BaseModel | None = None
    view_blocks: list[ViewBlock] = Field(default_factory=list, description="list of view blocks that are children of this view block")
    role: RoleType | None = None
    actions: list[Type[BaseModel]] | Type[BaseModel] | None = None
    
    tag: str | None = None
    class_: str | None = None
    parent_role: RoleType | None = None
    
    bullet: BulletType = "none"
    strip: StripType = Field(default=None, description="if the content should be stripped")
    base_model: BaseModelRenderType = 'json'
    base_model_indent: int = 2
    wrap: ViewWrapperType = None
    content_type: ContentType = 'text'
    role_name: str | None = None    
    index: int | None = None
    
    depth: int = 0
    indent: int = 0
    list_model: ListModelRender = 'view_node'
    action_calls: List[ActionCall] | None = None
    
    visited: bool = False
    
    def get_type(self):
        return type(self.content)
        return type(self.view_blocks)
    
    def has_wrap(self):
        return self.wrap is not None or self.title is not None
    
    def is_leaf(self):
        return self.get_type() == str or issubclass(self.get_type(), BaseModel)
    
    def __hash__(self):
        return self.uuid.__hash__()
    
    @property
    def action_call_uuids(self):
        if not self.action_calls:
            return []
        return [a.id for a in self.action_calls]
    
    def find_actions(self) -> Actions:
        actions = Actions()
        for block in self.pre_order_traversal():            
            if block.actions:
                actions.extend(block.actions)
        return actions
    
    def push(self, block: ViewBlock):
        assert isinstance(block, ViewBlock)
        self.view_blocks.append(block)
        
    def extend(self, blocks: list[ViewBlock]):
        assert all(isinstance(b, ViewBlock) for b in blocks)
        self.view_blocks.extend(blocks)
        
        
    def find(
        self,
        tag: str=None, 
        role: str=None, 
        view_name: str=None, 
        class_: str=None, 
        depth: int | None = None,
        min_depth: int=0, 
        max_depth: int=100,
        action_calls: bool | None=None,
        replace: bool=True,
        uuids: list[str] | None = None,
        enumerated: bool=False
        
    ) -> Generator[ViewBlock, None, None]:
                
        filter_func = combine_filters(tag, role, view_name, class_, depth, min_depth, max_depth, action_calls, uuids)
        for (current_depth, index), block in self.pre_order_traversal(enumerated=True):
            if filter_func(block, current_depth):
                if block.visited == False:
                    if not replace:
                        block.visited = True
                    if enumerated:
                        yield (current_depth, index), block
                    else:
                        yield block

            
    def first(
        self,        
        tag: str=None, 
        role: str=None, 
        view_name: str=None, 
        class_: str=None,
        depth: int | None = None,
        min_depth: int=0, 
        max_depth: int=100,
        action_calls: bool | None=None,
        uuids: list[str] | None = None,
        skip: int=0,
        enumerated: bool=False
    ) -> ViewBlock:
        """return the first block that matches the filter"""
        for (depth, index), block in self.find(
            tag, 
            role, 
            view_name, 
            class_, 
            depth, 
            min_depth, 
            max_depth, 
            action_calls=action_calls, 
            uuids=uuids,
            enumerated=True
        ):
            if skip == 0:
                if enumerated:
                    return (depth, index), block
                return block
            skip -= 1
    
    def count(
        self, 
        tag: str=None, 
        role: str=None, 
        view_name: str=None, 
        class_: str=None,
        depth: int | None = None,
        min_depth: int=0, 
        max_depth: int=100,
        action_calls: bool | None=None,
        uuids: list[str] | None = None
    ):
        """counting the number of blocks that match the filter under the current block"""
        count = 0
        for block in self.find(
            tag=tag,
            role=role,
            view_name=view_name,
            class_=class_,
            depth=depth,
            min_depth=min_depth,
            max_depth=max_depth,
            action_calls=action_calls,
            uuids=uuids
        ):
            count += 1
        return count
    
        
    def pre_order_traversal(self, enumerated=False) -> Generator[ViewBlock, None, None]:
        """
        Perform pre-order traversal of the tree without recursion.
        This yields each ContentBlock and its children in pre-order.
        """
        stack = [(self, 0, 0)]  # Initialize stack with the root node (self)
        
        while stack:
            current_block, depth, child_index = stack.pop()
            if current_block is not self:
                if enumerated:
                    yield (depth, child_index), current_block
                else:
                    yield current_block
            
            # Add children to the stack in reverse order so they are processed in the correct order
            for i, child in enumerate(reversed(current_block.view_blocks)):
                stack.append((child, depth + 1, len(current_block.view_blocks) - 1 - i))

                
    def post_order_traversal(self) -> Generator[ViewBlock, None, None]:
        """
        Perform post-order traversal of the tree without recursion.
        This yields each ContentBlock and its children in post-order.
        """
        stack1 = [self]  # Stack to store the nodes for traversal
        stack2 = []  # Stack to reverse the order of processing
        
        # First step: Visit nodes and push them onto stack2 in reverse order
        while stack1:
            current_block = stack1.pop()
            stack2.append(current_block)
            
            # Add children to stack1 (normal order, they'll be reversed in stack2)
            if current_block.view_blocks:
                stack1.extend(current_block.view_blocks)
        
        # Second step: Pop from stack2 and yield, which ensures post-order traversal
        while stack2:
            item = stack2.pop()
            # if item is not self:
            yield item

    def replace_all(self):
        for block in self.post_order_traversal():
            block.visited = False


def transform_list_to_view_blocks(        
        items: List[Union[ViewBlock, BaseModel, str]],
        view_name: str,
        role: Literal["assistant", "user", "system"] | None = None,
        bullet: Literal["number" , "astrix" , "dash" , "none", None] | str = None,
        base_model: BaseModelRenderType = 'json',
        base_model_indent: int = 2,
        indent: int = 0,
    ):
    """
    ensure that all items in the list are converted to ContentBlock
    """
    sub_views = []
    for i, o in enumerate(items):
        if isinstance(o, str):
            sub_views.append(
                ViewBlock(
                    view_name=f"{view_name}_str_{i}",
                    content=o,
                    bullet=bullet,
                    index=i,
                    role=role,
                    indent=indent,
                    base_model_indent=base_model_indent,
                    tag=o.tag if hasattr(o, "tag") else None,
                    class_=o.class_ if hasattr(o, "class_") else None
                )   
            )
        elif isinstance(o, dict):
            # raise ValueError("dict type not supported")
            sub_views.append(
                ViewBlock(
                    view_name=f"{view_name}_model_{i}",
                    content=o,
                    bullet=bullet,
                    base_model=base_model,
                    index=i,
                    role=role,
                    indent=indent
                )
            )
        elif isinstance(o, ViewBlock):
            sub_views.append(o)
        elif isinstance(o, BaseModel):
            # raise ValueError("dict type not supported")
            sub_views.append(
                ViewBlock(
                    view_name=f"{view_name}_model_{i}",
                    content=o,
                    bullet=bullet,
                    base_model=base_model,
                    index=i,
                    role=role,
                    indent=indent
                )
            )
        else:
            raise ValueError(f"view type not supported: {type(o)}")
    return sub_views


def create_view_block(
    views,
    view_name: str,
    name: str | None=None,
    title: str | None = None,
    wrap: ViewWrapperType = None,
    content_type: ContentType = 'text',
    actions: List[BaseModel] | BaseModel | None = None,
    role: Literal["assistant", "user", "system"] | None = None,
    bullet: Literal["number" , "astrix" , "dash" , "none", None] | str = None,
    strip: StripType = None,
    base_model: BaseModelRenderType = 'json',
    base_model_indent: int = 2,
    indent: int = 0,
    tag: str | None = None,
    class_: str | None = None,
):
    view_id = str(uuid4())
    content = None
    view_blocks = []
    action_calls = None
    if type(views) == list or type(views) == tuple:
        view_blocks = transform_list_to_view_blocks(
            items=views,
            view_name=view_name,
            role=role,
            bullet=bullet,
            base_model=base_model ,
            base_model_indent=base_model_indent,
            # indent=indent,
        )
    elif isinstance(views, str):
        content = views
    elif isinstance(views, dict):
        content = views
    elif isinstance(views, ViewBlock):
        # if view_name != "root":
            # raise ValueError("ViewBlock type not supported. pass a list of ContentBlock or a 'root' view_name")
        view_blocks = [views]
    elif isinstance(views, BaseMessage):
        content = views.content
        role = views.role
        view_id = views.id if views.id is not None else view_id
        if isinstance(views, AIMessage):
            action_calls = views.action_calls
        elif isinstance(views, ActionMessage):
            if not view_id.startswith("tool"):
                print("tool message id not starts with tool")
            print(views)
    elif isinstance(views, BaseModel):
        content = views
    return ViewBlock(
        uuid=view_id,
        view_name=view_name,
        name=name,
        title=title,
        view_blocks=view_blocks,
        content=content,
        actions=actions,
        base_model=base_model,
        base_model_indent=base_model_indent,
        bullet=bullet,
        strip=strip,
        wrap=wrap,
        content_type=content_type,
        role=role,
        indent=indent,        
        tag=tag,
        class_=class_,
        action_calls=action_calls
    )

    
def view(
    container=None, 
    title=None,
    actions=None, 
    role="user",
    name=None,
    bullet: Literal["number" , "astrix" , "dash" , "none", None] | str=None,
    strip: StripType = None,
    base_model: BaseModelRenderType = 'json',
    base_model_indent: int = 2,
    wrap: ViewWrapperType = None,
    content_type: ContentType = 'text',
    indent: int = 0,
    class_: str | None = None,
    tag: str | None = None
    ):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):                            
            outputs = func(*args, **kwargs)
            if container is not None:
                outputs = container(*outputs if isinstance(outputs, tuple) else (outputs,))
            block_instance = create_view_block(
                views=outputs,
                view_name=func.__name__, 
                name=name, 
                title=title, 
                wrap=wrap,
                content_type=content_type,
                actions=actions, 
                role=role, 
                bullet=bullet,
                strip=strip, 
                base_model=base_model,
                base_model_indent=base_model_indent,
                indent=indent, 
                tag=tag,
                class_=class_
            )   
            # sub_blocks = []
            # if isinstance(outputs, list) or isinstance(outputs, tuple):
            #     sub_blocks = transform_list_to_content_blocks(
            #         outputs, 
            #         name=func.__name__, 
            #         role=role, 
            #         numerate=numerate, 
            #         base_model=base_model,
            #         indent=indent
            #     )
            # else:
            #     sub_blocks = outputs
            # block_instance = ContentBlock(
            #     name=func.__name__,
            #     title=title,
            #     content_blocks=sub_blocks,
            #     actions=actions,
            #     base_model=base_model,
            #     numerate=numerate,
            #     wrap=wrap,
            #     role=role,
            #     role_name=name,
            #     indent=indent,
            # )
            return block_instance            
        return wrapper    
    return decorator

def list_view(rules: list[str], numbered: bool = True):
    if numbered:
        return "\n".join([f"{i}. {r}" for i, r in enumerate(rules)])
    else:
        return "\n".join(rules)



class SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            if key not in kwargs:
                raise KeyError(f"Missing value for key: '{key}'")
            return kwargs[key]
        else:
            return super().get_value(key, args, kwargs)
        
def replace_placeholders(template: str, **kwargs) -> str:
    formatter = SafeFormatter()
    formatted_string = formatter.format(template, **kwargs)
    return formatted_string


def render_tabs(num: int):
    # return "\t" * num
    return "  " * num

def add_tabs(content: str, tabs: int):
    return "\n".join([render_tabs(tabs) + c for c in content.split("\n")])
    # return content.replace("\n", f"\n{render_tabs(tabs)}")


def render_model(block: ViewBlock):
    model = block.view_blocks
    prompt = ""
    if block.bullet and block.index:
        prompt += f"{block.index + 1}. "
        
    if block.base_model == 'json':
        return add_tabs(prompt + json.dumps(model.model_dump(), indent=block.base_model_indent), block.depth)
    elif block.base_model == 'model_dump':
        return add_tabs(prompt + str(model.model_dump()) + "\n", block.depth)
    else:
        raise ValueError(f"base_model type not supported: {block.base_model}")


def render_string(block: ViewBlock, **kwargs):
    prompt = ''
    depth = block.depth + 1 if block.has_wrap() else block.depth
    if block.bullet and block.index:
        prompt += f"{block.index + 1}. "    
    prompt += textwrap.dedent(block.view_blocks).strip()
    prompt = add_tabs(prompt, depth)
    return replace_placeholders(prompt, **kwargs)

def render_dict(block: ViewBlock):
    prompt = ''
    depth = block.depth + 1 if block.has_wrap() else block.depth
    if block.bullet and block.index:
        prompt += f"{block.index + 1}. "
    prompt += json.dumps(block.view_blocks, indent=block.base_model_indent)
    return add_tabs(prompt, depth)

def add_wrapper(content: str, block: ViewBlock):
    title = block.title if block.title is not None else ''
    if block.wrap == "xml":
        return add_tabs((
            f"<{title}>\n"
            f"\n{content}"
            f"</{title}>\n"   
        ), block.depth)
    
    if block.wrap == "markdown":
        return add_tabs((
            f"## {title}\n"
            f"\t{content}\n"
        ), block.depth)
    return add_tabs((
        f"{title}:"
        f"\t{content}"
        ), block.depth)


    
def render_wrapper_starting(block: ViewBlock):
    title = block.title if block.title is not None else ''
    if block.wrap == "xml":
        return add_tabs(f"<{title}>", block.depth)
    elif block.wrap == "markdown":
        return add_tabs(f"## {title}", block.depth)
    return add_tabs(f'{title}:', block.depth)

def render_wrapper_ending(block: ViewBlock):
    title = block.title if block.title is not None else ''
    if block.wrap == "xml":
        return add_tabs(f"</{title}>", block.depth)
    return ''



def validate_node(block: any):
    if type(block) == str:
        return ViewBlock(view_blocks=block)    
    return block



def get_action_name(action_class: Type[BaseModel]):
    if hasattr(action_class, "_title"):
        return action_class._title.default
    return camel_to_snake(action_class.__name__)


def find_action(action_name, actions):
    for action in actions:
        if get_action_name(action) == action_name:
            return action
    return None





#? in render view we are using 2 stacks so that we can render the views in the correct order
# ?is a view is between 2 strings, we want to render the view between the strings
def render_block(block: ViewBlock | tuple, **kwargs):

    if type(block) == tuple:
        stack = [*reversed(block)]    
    else:
        stack = [block]

    base_models = {}
    visited = set()
    result = []
    while stack:
        # peek_node = validate_node(stack[-1])
        peek_block = stack[-1]
                            
        if peek_block not in visited:
            visited.add(peek_block)
            if peek_block.has_wrap():
                result.append(render_wrapper_starting(peek_block))
            if peek_block.get_type() == str:
                result.append(render_string(peek_block, **kwargs))
            elif peek_block.get_type() == dict:
                result.append(render_dict(peek_block))
            elif peek_block.get_type() == list or peek_block.get_type() == tuple:
                for block in reversed(peek_block.view_blocks):
                    if peek_block.has_wrap():
                        block.depth = peek_block.depth + 1
                    else:
                        block.depth = peek_block.depth
                    
                    stack.append(block)
            elif issubclass(peek_block.get_type(), ViewBlock):
                if peek_block.has_wrap():
                    peek_block.view_blocks.depth = peek_block.depth + 1
                stack.append(peek_block.view_blocks)
            elif issubclass(peek_block.get_type(), BaseModel):
                base_models[peek_block.view_blocks.__class__.__name__] = peek_block.view_blocks
                result.append(render_model(peek_block))
            else:
                raise ValueError(f"block type not supported: {type(peek_block)}")
        else:
            if peek_block.has_wrap():
                result.append(render_wrapper_ending(peek_block))
            stack.pop(-1)
    prompt = "\n".join(result)
    return prompt, result, base_models