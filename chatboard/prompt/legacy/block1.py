from abc import abstractmethod
from functools import singledispatchmethod, wraps
import re
from uuid import uuid4
from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, Generic, List, Literal, ParamSpec, TypeVar, Optional, Union
import json
from typing import Protocol, runtime_checkable
import textwrap
from ...conversation.models import Message
from ...utils.string_utils import int_to_roman
from ...llms.messages import ActionCall, LlmUsage
import datetime as dt

BulletType = Literal["number", "alpha", "roman", "roman_lower", "*", "-"]
TitleType = Literal["xml", "md", "html"]

class Style(BaseModel):
    bclass: str | None = Field(default=None, description="The class for the block")
    ttype: TitleType  | None = Field(default=None, description="The style of the title")
    bullet: BulletType | None = Field(default=None, description="The bullet style for the list")
    prefix: str | None = Field(default=None, description="prefix to add to the block title")
    postfix: str | None = Field(default=None, description="postfix to add to the block title")
    tabs: int = Field(default=0, description="number of tabs to add to the block content")
    indent: int = Field(default=0, description="number of spaces to indent the block content")
    dropout: float = Field(default=0.0, description="dropout probability for the block")
    
    def copy_tabs(self, eq: int | None = None, inc: int | None = None, dec: int | None = None):
        if eq is not None:
            return self.model_copy(update={"tabs": eq})
        elif inc is not None:
            return self.model_copy(update={"tabs": self.tabs + inc})
        elif dec is not None:
            return self.model_copy(update={"tabs": self.tabs - dec})
        else:
            raise ValueError("Must provide one of eq, inc, or dec")
        
    def copy_prefix(self, prefix: str):
        return self.model_copy(update={"prefix": prefix})
    
    def copy_postfix(self, postfix: str):
        return self.model_copy(update={"postfix": postfix})

    def merge(self, other: "Style"):
        other_style = other.model_dump(exclude_none=True, exclude_defaults=True)
        return self.model_copy(update=other_style)

class StyleManager():
    
    def __init__(self, styles: dict):
        self.styles = styles
        
    def get(self, block_id, style: Style | None = None):
        style_dict = self.styles.get(block_id, None)
        if style_dict is None:
            return style
        if style is None:
            return Style(**style_dict)
        return Style(**(style_dict | style.model_dump()))
    
    def add(self, style_dict: dict):
        self.styles.update(style_dict)
    
    def get_style(self, block: "BaseBlock", style: Style | None = None):
        style = block.get_style(style=self.get(block.id, style))
        return style
    
    
style_manager = StyleManager({})



@runtime_checkable
class Renderable(Protocol):
    
    def render(self, style: Style)->str:
        """
        Classes conforming to the Renderable Protocol must implement
        a `render` method with the specified arguments.
        """
        ...
        

def tabs_str(style: Style):
    return "  " * style.tabs
       

class strblk(str, Renderable):
    
    def render(self, style: Style | None = None):
        if style is None:
            style = Style()
        content = self
        if style.prefix:
            content = style.prefix + content
        if style.postfix:
            content = content + style.postfix
        if style.tabs or style.indent:
            content = "  " * (style.tabs or style.indent) + content
        return content
    
    
class dictblk(dict, Renderable):
    
    def render(self, style: Style | None = None):
        if style is None:
            style = Style()
        content = json.dumps(self, indent=style.indent)
        return content


def to_renderable(item):
    if isinstance(item, Renderable):
        return item
    elif isinstance(item, str):
        return strblk(item)
    elif isinstance(item, dict):
        return dictblk(item)
    else:
        raise ValueError(f"Invalid item type: {type(item)}. Must be str, dict, or Renderable")
 
# class listblk(list):
    
#     def render(self, indent=4):
#         return json.dumps(self, indent=indent)

BlockRole = Literal["assistant", "user", "system", "tool"]


class BaseBlock(BaseModel):
    db_msg_id: str | int = Field(default_factory=lambda: str(uuid4()), description="The uuid for the block. links to database.")
    platform_id: str | None = Field(default=None, description="The platform uuid for the block. links to platform.")
    role: str = Field(default="user", description="The role of the block")
    name: str | None = Field(default=None, description="The name of the block")
    id: int | str | None = Field(default=None, description="The id for the block")
    tag: str | None = Field(default=None, description="The tag for the block")
    bclass: str = Field(default="block", description="The class for the block")
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(), description="The creation time for the block")
    # message: Message | None = Field(default=None, description="The message for the block")
    _message: Message | None = None
    # style: Style = Field(default_factory=lambda: Style())
    _items: list[Renderable] = []
    
    
    # class Config:
    #     arbitrary_types_allowed = True
    def __init__(
        self,
        content: str | list[str] | list["BaseBlock"] | None = None,
        role: BlockRole = "user", 
        name: str | None = None,
        id: int | str | None = None,
        db_msg_id: str | None = None,                
        tag: str | None = None,
        platform_id: str | None = None,
        bclass: str = "block",
        created_at: dt.datetime | None = None,
    ):
        super().__init__()
        self.db_msg_id = db_msg_id or str(uuid4())
        self.role = role
        self.name = name
        self.id = id
        self.tag = tag
        self.bclass = bclass
        self.platform_id = platform_id
        self.created_at = created_at or dt.datetime.now()
        self._items = []
        self.set_content(content)
    
    def set_content(self, content: str | list[str] | list["BaseBlock"] | None):
        if content is not None:            
            if type(content) == list:
                for item in content:
                    self.append(item)
            else:
                self.append(content)
    
    
    @property
    def content(self):
        return "\n".join([item.render(style=style_manager.get_style(self)) for item in self._items])
    
    @property
    def message(self):
        return self._message
    
    @message.setter
    def message(self, value: Message):
        self._message = value
                
    def append(self, item):        
        if not isinstance(item, Renderable):
            if isinstance(item, str):
                item = strblk(textwrap.dedent(item).strip())
            elif isinstance(item, dict):
                item = dictblk(item)
            elif isinstance(item, BaseBlock):
                item = item
            else:
                raise ValueError(f"Invalid item type: {type(item)}. Must be str, dict, or Renderable")
        self._items.append(item)
        return item
    
    def extend(self, items):
        if not isinstance(items, list) and not isinstance(items, tuple):
            raise ValueError(f"Invalid items type: {type(items)}. Must be list or tuple")
        if isinstance(items, tuple):
            items = list(items)
        self._items.extend(items)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def __call__(self, *args):
        for arg in args:
            self.append(arg)
        return self
    
    def __add__(self, other):
        self.append(other)
        return self
    
    @abstractmethod
    def get_style(self, style: Style | None = None) -> Style:
        """
        generates a style object for the block
        """
        ...
        # if style is not None:
        #     return style.model_copy(update=self.style.model_dump())
        # return self.style
    
    def render(self, style: Style | None = None):        
        # new_style = self.get_style(style)
        # return "\n".join([item.render(style=new_style) for item in self._items])
        return "\n".join([item.render() for item in self._items])
    
    def __repr__(self) -> str:
        repr_str = super().__repr__()
        return repr_str + "\n" + self.render()
    
    
    @singledispatchmethod
    def get(self, tag: Union[str, List[str]]) -> Optional[Union[str, dict]]:
        """
        Generic get method. This will raise NotImplementedError
        if the type of tag is not supported.
        """
        raise NotImplementedError(f"Unsupported type: {type(tag)}")
    

    @get.register(str)
    def _(self, tag: str) -> Optional[str]:
        """
        Get the content for a single tag.
        """
        pattern = f'<{re.escape(tag)}>(.*?)</{re.escape(tag)}>'
        match = re.search(pattern, self.content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @get.register(list)
    def _(self, tags: List[str]) -> dict:
        """
        Get the content for a list of tags.
        Returns a dictionary with tags as keys and their corresponding content as values.
        """
        results = {}
        for tag in tags:
            content = self.get(tag)
            if content is not None:
                results[tag] = content
        return results
    
        
    
class TitleBlock(BaseBlock):
    ttype: TitleType = Field(default="md", description="The style of the title")
    title: str | None = Field(default=None, description="The title of the block")
    properties: Dict[str, str] | None = Field(default_factory=dict, description="Additional properties for the block")
    
    def __init__(
        self,
        title: str | None = None,
        content: str | list[str] | list["BaseBlock"] | None = None,
        ttype: TitleType = "md",    
        role: BlockRole = "user", 
        name: str | None = None,
        db_msg_id: str | None = None,  
        platform_id: str | None = None,
        id: int | str | None = None,
        tag: str | None = None,
        bclass: str = "block",
        created_at: dt.datetime | None = None,
    ):
        super().__init__(content=content, role=role, name=name, db_msg_id=db_msg_id, platform_id=platform_id, id=id, tag=tag, bclass=bclass, created_at=created_at)
        self.title = title
        self.ttype = ttype
        # self._items = []
        

    
    def _render_md(self):
        content = super().render()
        return f"## {self.title}\n{content}"
    
    def _render_html(self):
        content = super().render()
        return f"<h2>{self.title}</h2>\n{content}"
    
    def _render_xml(self):
        content = super().render()
        return f"<{self.title}>\n{content}\n</{self.title}>"    
    
    def get_style(self, style: Style | None = None) -> Style:        
        curr_style = Style(ttype=self.ttype)
        if style is not None:
            curr_style = curr_style.merge(style)
        return curr_style
    
    def _render_title(self, content: str, style: Style):
        content = textwrap.indent(content, "  ")
        if style.ttype == "md":
            content = f"{tabs_str(style)}## {self.title}\n{content}"
        elif style.ttype == "html":
            content = f"{tabs_str(style)}<h2>{self.title}</h2>\n{content}"
        elif style.ttype == "xml":
            content = f"{tabs_str(style)}<{self.title}>\n{content}\n{tabs_str(style)}</{self.title}>"
        else:
            raise ValueError(f"Invalid title style: {style.ttype}")
        return content

    # def render(self, style: Style | None = None):
    #     style = style_manager.get_style(self, style)        
    #     if self.title:
    #         content = super().render(style=style.copy_tabs(inc=1))
    #         # print(self.__class__.__name__, self.uuid, content)
    #         content = self._render_title(content, style)        
    #     else:
    #         content = super().render(style=style)
    #     return content
    
    def render(self, style: Style | None = None):
        style = style_manager.get_style(self, style)        
        if self.title:
            content = super().render(style=Style(tabs=style.tabs+1))
            # print(self.__class__.__name__, self.uuid, content)
            content = self._render_title(content, style)        
        else:
            content = super().render(style=style)
        return content
    
    
    def list(self, title=None, ttype: TitleType="md", bullet: BulletType = "-", role: BlockRole = "user", name: str | None = None, id: str | None = None):        
        lb = ListBlock(title=title, ttype=ttype, bullet=bullet, id=id, role=role, name=name)
        self.append(lb)
        return lb
    
    def block(self, title=None, ttype: TitleType="md", role: BlockRole = "user", name: str | None = None, id: str | None = None):        
        lb = TitleBlock(title=title, ttype=ttype, id=id, role=role, name=name)
        self.append(lb)
        return lb
    
    
# class Block(TitleBlock):
#     ...
    
    
    
class ListBlock(TitleBlock):
    bullet: BulletType = Field(default="-", description="The bullet style for the list")
    
    def __init__(
        self,
        title: str | None = None,
        ttype: TitleType = "md",
        bullet: BulletType = "-",
        id: str | None = None,
        role: BlockRole = "user",
        name: str | None = None,
        platform_id: str | None = None,
        db_msg_id: str | None = None,
        bclass: str = "block",
        created_at: dt.datetime | None = None,  
    ):
        super().__init__(title=title, ttype=ttype, id=id, role=role, name=name, platform_id=platform_id, db_msg_id=db_msg_id, bclass=bclass, created_at=created_at)
        self.bullet = bullet    

    
    def _get_prefix(self, idx: int):
        if self.bullet == "number":
            return f"{idx+1}. "
        elif self.bullet == "alpha":
            return f"{chr(97+idx)}. "
        elif self.bullet == "roman":
            return int_to_roman(idx+1) + ". "
        elif self.bullet == "roman_lower":
            return int_to_roman(idx+1, upper=False) + ". "
        return f"{self.bullet} "
    
    def _get_list_item(self, idx, item):
        bullet = self._get_prefix(idx)
        return f"{bullet} {item}"
    
    def get_style(self, style: Style | None = None) -> Style:        
        curr_style = Style(bullet=self.bullet, ttype=self.ttype)
        if style is not None:
            curr_style = curr_style.merge(style)
        return curr_style
    
    def render(self, style: Style | None = None):
        new_style = style_manager.get_style(self, style)
        new_style.tabs += 1
        content = "\n".join([item.render(style=new_style.model_copy(update={"prefix": self._get_prefix(idx)})) for idx, item in enumerate(self._items)])
        if self.title:
            content = self._render_title(content, new_style)                    
        return content
    







P = ParamSpec("P")
R = TypeVar("R")






def block_decorator(title=None, ttype: TitleType="md", id: str | None = None):        
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):                
            block = TitleBlock(title=title, ttype=ttype, id=id)
            output = func(*args, **kwargs)
            block.append(output)
            return block
        return wrapper
    return decorator
  

class block:
        
    
    def __init__(self, title: str | None = None, ttype: TitleType="md", role: BlockRole = "user", name: str | None = None, id: str | None = None):
        self.block_args = {"title": title, "ttype": ttype, "id": id, "role": role, "name": name}
    
    def __enter__(self):
        return TitleBlock(**self.block_args)
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass   
    
    @staticmethod
    def list(title: str | None = None, ttype: TitleType="md", bullet: BulletType = "-", role: BlockRole = "user", name: str | None = None, id: str | None = None):        
        return ListBlock(title=title, ttype=ttype, bullet=bullet, id=id, role=role, name=name)

    def _process_output(self, output: str | BaseBlock | List[str] | List[BaseBlock] | None):
        if output is None:
            return None
        if isinstance(output, str):
            return strblk(output)
        elif isinstance(output, dict):
            return dictblk(output)
        elif isinstance(output, list) or isinstance(output, tuple):
            return [to_renderable(item) for item in output]
        elif isinstance(output, BaseBlock):
            return output
        else:
            raise ValueError(f"Invalid output type: {type(output)}. Must be str, list, tuple, or TitleBlock")
        
        return output
    
    def __call__(self, func: Callable[P, TitleBlock]) -> Callable[P, TitleBlock]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> TitleBlock:
            output = func(*args, **kwargs)
            output = self._process_output(output)            
            block = TitleBlock(**self.block_args)
            if isinstance(output, list) or isinstance(output, tuple):
                block.extend(output)
            else:
                block.append(output)
            return block            
        return wrapper
    
  
        
    # def __call__(self, func: Callable[P, R]):
    #     # return func
    #     return block_decorator(title=self.block.title, ttype=self.block.ttype, id=self.block.id)
    
    # @staticmethod
    # def decorator(cls, title=None, ttype: TitleType="md", id: str | None = None):        
    #     def subdecorator(func: Callable[P, BaseBlock]) -> Callable[P, TitleBlock]:
    #         @wraps(func)
    #         def wrapper(*args, **kwargs):                
    #             block = TitleBlock(title=title, ttype=ttype, id=id)
    #             output = func(*args, **kwargs)
    #             block.append(output)
    #             return block
    #         return wrapper
    #     return subdecorator
    
    # @staticmethod
    # def wrap(title=None, ttype: TitleType="md", id: str | None = None):        
    #     return decorator(title=title, ttype=ttype, id=id)
    
    # def __call__(self, title=None, ttype: TitleType="md", id: str | None = None):
    #     return block_decorator(title=title, ttype=ttype, id=id)
    
    


# def block(
#     content: str | None = None,
#     title: str | None = None, 
#     ttype: TitleType = "md", 
#     role: BlockRole = "user", 
#     name: str | None = None, 
#     id: str | None = None,
#     func: Callable | None = None
# ):
#     """
#     Multi-purpose block factory that can be used as:
#     1. A factory function: block(title="My Title")
#     2. A decorator: @block(title="My Title")
#     3. A context manager: with block(title="My Title") as b:
#     """
#     block_args = {"content": content, "title": title, "ttype": ttype, "id": id, "role": role, "name": name}

#     # Helper function to process outputs
#     def _process_output(output: str | BaseBlock | List[str] | List[BaseBlock] | None):
#         if output is None:
#             return None
#         if isinstance(output, str):
#             return strblk(output)
#         elif isinstance(output, dict):
#             return dictblk(output)
#         elif isinstance(output, list) or isinstance(output, tuple):
#             return [to_renderable(item) for item in output]
#         elif isinstance(output, BaseBlock):
#             return output
#         else:
#             raise ValueError(f"Invalid output type: {type(output)}. Must be str, list, tuple, or TitleBlock")

#     # Context manager methods
#     class BlockContextManager:
#         def __enter__(self):
#             return TitleBlock(**block_args)
        
#         def __exit__(self, exc_type, exc_value, traceback):
#             pass

#     # Decorator wrapper
#     def decorator(func: Callable[P, Any]) -> Callable[P, TitleBlock]:
#         @wraps(func)
#         def wrapper(*args, **kwargs) -> TitleBlock:
#             output = func(*args, **kwargs)
#             output = _process_output(output)            
#             block = TitleBlock(**block_args)
#             if isinstance(output, list) or isinstance(output, tuple):
#                 block.extend(output)
#             else:
#                 block.append(output)
#             return block            
#         return wrapper

#     # Handle the different use cases
#     if func is not None:
#         # Used as a decorator without parameters: @block
#         return decorator(func)
#     elif callable(content):
#         # Used as a decorator without parentheses: @block
#         return decorator(content)
#     elif isinstance(content, str) or content is None:
#         # Used as a context manager: with block(...) as b:
#         # or as a factory function: block(...)
#         cm = BlockContextManager()
#         return cm if hasattr(cm, '__enter__') else TitleBlock(**block_args)

#     return decorator   
    
    



class ResponseBlock(TitleBlock):
    model: str | None = Field(default=None, description="The model used to generate the response")
    did_finish: bool = Field(default=True, description="Whether the response generation finished")
    run_id: str | None = Field(default=None, description="The run ID for this response")
    action_calls: List[ActionCall] = Field(default_factory=list, description="List of action calls made by the response")
    usage: LlmUsage | None = Field(default=None, description="Token usage information")
    raw: Any = Field(default=None, description="Raw response data")
    # block_role: Literal["assistant"] = Field(default="assistant", alias="role")
    
    def __init__(
        self,
        content: str | None = None,
        model: str | None = None,
        did_finish: bool = True,
        run_id: str | None = None,
        action_calls: List[ActionCall] | None = None,
        usage: LlmUsage | None = None,
        raw: Any = None,
        title: str | None = None,
        ttype: TitleType = "md",    
        name: str | None = None,
        db_msg_id: str | None = None, 
        platform_id: str | None = None,
        id: int | str | None = None,
        tag: str | None = None,
        bclass: str = "block",
        created_at: dt.datetime | None = None,
    ):
        super().__init__(
            content=content, 
            title=title, 
            ttype=ttype, 
            name=name, 
            db_msg_id=db_msg_id, 
            platform_id=platform_id,
            id=id, 
            role="assistant", 
            tag=tag, 
            bclass=bclass,
            created_at=created_at
        )
        self.model = model
        self.did_finish = did_finish
        self.run_id = run_id
        self.action_calls = action_calls or []
        self.usage = usage
        self.raw = raw
        self._tool_responses: dict[str, Any] = {}
        self._tools: dict[str, BaseModel] = {}

    @property
    def actions(self) -> List[BaseModel]:
        return list(self._tools.values())
    
    @property
    def output(self) -> BaseModel | None:
        if not self.action_calls:
            return None
        action = self.action_calls[0].action
        if isinstance(action, BaseModel):
            return action
        return None
    
    def set_actions(self, value: List[BaseModel]) -> None:
        self._tools = {str(i): action for i, action in enumerate(value)}
        self.action_calls = [
            ActionCall(id=str(i), name=action.__class__.__name__, action=action)
            for i, action in enumerate(value)
        ]
    
    def add_action_output(self, tool_id: str, output: BaseModel | str | dict) -> None:
        self._tool_responses[tool_id] = output
        
    def add_action(self, tool_id: str, action: BaseModel) -> None:
        self._tools[tool_id] = action


class ActionBlock(TitleBlock):
    # block_role: Literal["tool"] = Field(default="tool", alias="role")
    tool_call_id: str | None = Field(default=None, description="The ID of the tool call this action is responding to")
    
    def __init__(
        self,
        content: str | None = None,
        tool_call_id: str | None = None,
        title: str | None = None,
        ttype: TitleType = "md",    
        name: str | None = None,
        db_msg_id: str | None = None,   
        platform_id: str | None = None,
        id: int | str | None = None,
        tag: str | None = None,
        bclass: str = "block",
        created_at: dt.datetime | None = None,
    ):
        super().__init__(
            content=content, 
            title=title, 
            ttype=ttype, 
            role="tool", 
            name=name, 
            db_msg_id=db_msg_id, 
            platform_id=platform_id,
            id=id or tool_call_id, 
            tag=tag, 
            bclass=bclass,
            created_at=created_at
        )
        self.tool_call_id = tool_call_id







