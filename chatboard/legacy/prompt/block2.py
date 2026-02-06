import contextvars
import textwrap
from typing import List, Literal, Type
from uuid import uuid4

from pydantic import BaseModel

from chatboard.utils.model_utils import schema_to_ts
from chatboard.utils.string_utils import int_to_roman


TitleType = Literal["md", "xml"]
# ListType = Literal["list", "table"]
BulletType = Literal["number", "alpha", "roman", "roman_upper", "*", "-"]

block_ctx = contextvars.ContextVar("block_ctx")


def sanitize_value(value: str):
    return textwrap.dedent(value).strip()


def instantiate_block(instance: "StrBlock"):
    instance._items = []
    instance._token = None
    instance._id = str(uuid.uuid4())
    return instance

def connect_block(instance: "StrBlock"):
    try:
        instance._parent = block_ctx.get()
        if instance._parent is not None:            
            instance._parent.append(instance)
            instance._depth = instance._parent._depth + 1
    except LookupError:
        pass
    return instance


class LlmUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int




class ToolCall(BaseModel):
    id: str
    name: str
    tool: dict | BaseModel
    
    @property
    def type(self):
        return type(self.tool)
    
    def to_json(self):
        return self.tool.model_dump_json() if isinstance(self.tool, BaseModel) else json.dumps(self.tool)




class StrBlock(str):
    
    _items: list["StrBlock | TitleBlock | ListBlock"]
    _tool_calls: list[ToolCall]
    _token: contextvars.Token | None = None
    _parent: "StrBlock | None" = None
    _id: str | None = None
    _style: dict | None = None
    _depth: int = 1
    _indent: int = 3
    _role: str | None = None
    _name: str | None = None
    _uuid: str | None = None
    
    
    def __new__(
        cls, 
        value, 
        role: str | None = None, 
        id: str | None = None, 
        _auto_append: bool = True, 
        name: str | None = None, 
        uuid: str | None = None,
        tool_calls: list[ToolCall] | None = None
        ):
        # Remove leading and trailing whitespace and dedent the value
        value = textwrap.dedent(value).strip() 
        # Create the instance using str's __new__        
        instance = super().__new__(cls, value)
        # Attach extra attributes to the instance
        instance._items = []
        instance._tool_calls = tool_calls or []
        instance._token = None
        instance._id = id
        instance._role = role
        instance._uuid = uuid if uuid else str(uuid4())
        instance._name = name
        
        if _auto_append:
            try:
                instance._parent = block_ctx.get()
                if instance._parent is not None:            
                    instance._parent.append(instance)
                    instance._depth = instance._parent._depth + 1
            except LookupError:
                pass
        return instance

    @property
    def role(self):
        return self._role
    
    @role.setter
    def role(self, value: str):
        self._role = value
        
    @property
    def id(self):
        return self._id
    
    @property
    def tool_calls(self):
        return self._tool_calls
    
    @tool_calls.setter
    def tool_calls(self, value: list[ToolCall]):
        self._tool_calls = value

    
    @property
    def name(self):
        return self._name
    
    @property
    def uuid(self):
        return self._uuid
    
    def append(self, item: "StrBlock | str"):
        if isinstance(item, StrBlock):
            self._items.append(item)
        elif isinstance(item, str):
            self._items.append(StrBlock(item, _auto_append=False))
        else:
            self._items.append(item)
     
    def render_items(self):
        content = "\n".join([item.render() for item in self._items]) 
        if self._indent:     
            content = textwrap.indent(content, " " * self._indent)
        return content
        
    def render(self):
        content = self
        if self._items:
            content = content + "\n" + self.render_items()
        return content
        
    def show(self):
        # self is the string content
        print("Content:", self)
        # print("Extra info:", self._type)
        
    def __enter__(self):
        self._token = block_ctx.set(self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._token is not None:
            block_ctx.reset(self._token)
        return False
    
    
    
    
class TitleBlock(StrBlock):
    _type: TitleType
    _sub_items_bullet: BulletType = "number"
    _registered_models = {}
    
    def __new__(
        cls, 
        value, 
        type: TitleType, 
        bullet: BulletType = "number", 
        role: str | None = None, 
        id: str | None = None, 
        indent: int = 0, 
        _auto_append: bool = True,
        name: str | None = None,
        uuid: str | None = None
        ):
        instance = super().__new__(cls, value, role, id, _auto_append, name, uuid)
        instance._type = type
        instance._sub_items_bullet = bullet
        instance._indent = indent
        return instance
    
    def register_model(self, action_name: str, model_class: Type[BaseModel]):
        """Register a model class for a specific action name."""
        from ..block_parser import ActionContent
        ActionContent.register_model(action_name, model_class)
        return self
    
    def render(self):
        content = self.render_items()
        if self._type == "md":
            content = "#" * self._depth + " " + self + "\n" + content
            
        elif self._type == "xml":
            content = "<" + self + ">\n" + content + "\n</" + self + ">\n"
        return content
    
    def parse(self, text: str):
        """
        Parse the rendered output text into a structured object.
        The text should be in XML format with Observation, Thought, and Actions sections.
        Actions can contain registered model types that will be automatically instantiated.
        """
        from ..block_parser2 import parse_blocks
        return parse_blocks(text)
    
    
    
class ListBlock(StrBlock):
    _type: BulletType = "number"
    _idx: int = 0
    
    def __new__(
        cls, 
        value, 
        type: BulletType = "number", 
        role: str | None = None, 
        id: str | None = None, 
        _auto_append: bool = True,
        name: str | None = None,
        uuid: str | None = None 
    ):
        instance = super().__new__(cls, value, role, id, _auto_append, name, uuid)
        instance._type = type
        if instance._parent:
            instance._idx = len([item for item in instance._parent._items if isinstance(item, ListBlock)])
        else:
            instance._idx = 0
        return instance
    
    def get_bullet_type(self):
        if self._parent and isinstance(self._parent, TitleBlock):
            return self._parent._sub_items_bullet
        else:
            return self._type
    
    def _get_prefix(self):
        idx = self._idx
        bullet_type = self.get_bullet_type()
        if bullet_type == "number":
            return f"{idx}. "
        elif bullet_type == "alpha":
            return f"{chr(96+idx)}. "
        elif bullet_type == "roman_upper":
            return int_to_roman(idx, upper=True) + ". "
        elif bullet_type == "roman":
            return int_to_roman(idx, upper=False) + ". "
        return f"{bullet_type} "

    def render(self):
        content = self._get_prefix() + self        
        if self._items:
            content = content + "\n" + self.render_items()
        return content
    

    
    
class XmlBlock(StrBlock):
    _attributes: dict = {}
    _sub_items_bullet: BulletType = "number"
    
    def __new__(
        cls, 
        value, 
        bullet: BulletType = "number", 
        role: str | None = None, 
        id: str | None = None, 
        _auto_append: bool = True,
        name: str | None = None,    
        uuid: str | None = None,
        **kwargs
    ):
        instance = super().__new__(cls, value, role, id, _auto_append, name, uuid)
        instance._attributes = kwargs
        instance._sub_items_bullet = bullet
        return instance
    
    def render(self):
        content = self
        if self._attributes:
            content = content + " " + " ".join([f"{k}=\"{v}\"" for k, v in self._attributes.items()])
        if self._items:
            item_content = self.render_items()
            content = "<" + content + ">\n" + item_content + "\n</" + self + ">"
        else:
            content = "<" + content + " />"
        return content
    
# class OutputFormatBlock(StrBlock):
#     _model: Type[BaseModel]
    
#     def __new__(
#         cls, 
#         model: Type[BaseModel],
#         title: str = "Output Format",
#         role: str = "system",
#         name: str | None = None,
#     ):
#         instance = super().__new__(cls, title, role="system", id="output_format")
#         instance._model = model        
#         return instance
    
    
        

class EncloseBlock(StrBlock):
    _enclose_fmt: type
    
    def __new__(
        cls, 
        value, 
        fmt: type = tuple, 
        role: str | None = None, 
        id: str | None = None, 
        _auto_append: bool = True,
        name: str | None = None,
        uuid: str | None = None
    ):
        instance = super().__new__(cls, value, role, id, _auto_append, name, uuid)
        instance._enclose_fmt = fmt
        return instance
    
    def render(self):
        if self._enclose_fmt == list:
            open_char, close_char = '[', ']'
        elif self._enclose_fmt == tuple:
            open_char, close_char = '(', ')'
        else:
            raise ValueError("fmt must be either list or tuple")
        content = ', '.join(repr(item) for item in self._items)
        return f"{open_char}{content}{close_char}"
 
 
 

    
# class Block:
    
    
#     @staticmethod
#     def title(value: str, type: TitleType):
#         return TitleBlock(value, type)
    
#     @staticmethod
#     def list(value: str, type: TitleType):
#         return ListBlock(value, type)
    
    
#     @staticmethod
#     def model_dump(model: Type[BaseModel], format: str = "ts"):    
#         if format == "ts":
#             content = schema_to_ts(model)
#         else:
#             content = model.model_json_schema()        
#         return StrBlock(content)
    
#     @staticmethod
#     def li(item: str):
#         return StrBlock(item, _auto_append=False)

class ListBuilder:
    # def __init__(self, type: BulletType = "number"):
    #     self._type = type
    
    def __add__(self, other: StrBlock | List[str] | str):
        ListBlock(other)
        return self
        
    def __call__(self, value: str, type: BulletType = "number"):
        return ListBlock(value, type)

class Block:
    
    def __init__(
        self, 
        value: str | None = None, 
        role: str = "user", 
        id: str | None = None,
        name: str | None = None,
        uuid: str | None = None
        ):
        self.li = ListBuilder()
        if value:
            self.value = StrBlock(value, role, id, name=name, uuid=uuid)
        else:
            self.value = None
    

    def title(
        self, 
        value: str, 
        type: TitleType = "md", 
        bullet: BulletType = "number", 
        role: str | None = None, 
        id: str | None = None, 
        indent: int = 0, 
        name: str | None = None, 
        uuid: str | None = None
    ):
        """
        Create a title block.
        
        Args:
            value: The text of the title.
            type: The type of the title: `md` or `xml`.
            bullet: `number`, `alpha`, `roman`, `roman_upper`, `*`, `-`. The bullet type of underlying list items
            indent: The indent of the sub-items.
            name: The name of the title.
            uuid: The uuid of the title.
        Returns:
            TitleBlock: The title block.
        """
        return TitleBlock(value, type, bullet, role, id, indent, name=name, uuid=uuid)
    
    def xml(self, value: str, bullet: BulletType = "number", **kwargs):
        return XmlBlock(value, bullet, **kwargs)
    
    def output_format(self, model: Type[BaseModel], title: str = "Output Format", role: str = "system", name: str | None = None):
        return OutputFormatBlock(model, title, role, name)

    def model_dump(self, model: Type[BaseModel], format: str = "ts"):    
        if format == "ts":
            content = schema_to_ts(model)
        else:
            content = model.model_json_schema()        
        return StrBlock(content)
    
    def _add_item(self, target: StrBlock | List[str] | str, role: str = "user", id: str | None = None):
        _target = None
        if isinstance(target, StrBlock):
            _target = target
        elif isinstance(target, str):
            _target = StrBlock(target, role, id, _auto_append=True)
        elif isinstance(target, list):
            raise ValueError("List of strings is not supported")
            # _target = [StrBlock(item, role, id, _auto_append=True) for item in target]
        else:
            raise ValueError(f"Invalid target type: {type(target)}")

        return _target
    
    def __add__(self, other: StrBlock | List[str] | str):
        self._add_item(other)
        return self
    
    def __call__(self, target: StrBlock | List[str] | str, role: str = "user", id: str | None = None):
        
        return self._add_item(target, role, id)
    
    def __radd__(self, other: StrBlock | List[str] | str):
        return self._add_item(other)
    
    def __enter__(self) -> "StrBlock":
        if not self.value:
            self.value = StrBlock("", _auto_append=True)
        return self.value
    
    def __exit__(self, exc_type, exc_value, traceback):
        return False
    
    
block = Block()