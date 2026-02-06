import contextvars
import textwrap
from typing import Any, Generator, Generic, Text, Type, TypeVar
from typing import get_type_hints, List
from typing import get_args, get_origin, List
from uuid import uuid4
import uuid
from .block import AttrBlock, Block, BlockSequence, BlockSent, BlockChunk, BlockSchema


BaseContent = str | int | float | bool | None

CONTENT = TypeVar("CONTENT")


style_registry_ctx = contextvars.ContextVar("style_registry_ctx", default={})

def _style_key(style: str, target: str, effects: str="all") -> str:
    return f"{style}_{target}_{effects}"   

class StyleContext:
        
    def __init__(
        self,
        styles: dict
    ):
        self.styles = styles
        self._token = None
        
        
    def __enter__(self):
        current = style_registry_ctx.get()
        self._token = style_registry_ctx.set(self.styles)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._token:
            style_registry_ctx.reset(self._token)
        return False
        
        


class StyleMeta(type):
    # _registry: dict[str, "StyleMeta"] = {}
    # _styles: dict[str, "BlockStyle"] = {}
    
    
    def __new__(cls, name, bases, attrs):        
        new_cls = super().__new__(cls, name, bases, attrs)
        # cls._registry[name] = new_cls
        if styles := attrs.get("styles"):
            for style in styles:
                # cls._styles[style] = new_cls()
                # current = style_registry_ctx.get()
                # if style not in current:
                #     current[style] = {
                #         "text": None,
                #         "block": None,
                #         "chunk": None,
                #     }
                # style_obj = new_cls()
                # current[style_obj.target] = style_obj
                style_obj = new_cls()
                style_registry_ctx.get()[_style_key(style, style_obj.target, style_obj.effects)] = style_obj
        return new_cls
    
    
    
    @classmethod
    def list_styles(cls) -> list[str]:
        current = style_registry_ctx.get()
        return list(current.keys())
    
    # @classmethod
    # def resolve(cls, styles: list[str], target: str, effects: str="all", default: "BlockStyle | TextStyle | ChunkStyle | None"=None) -> "BlockStyle | TextStyle | ChunkStyle | None":
    #     current = style_registry_ctx.get()
    #     for style in styles:
    #         if style_obj := current.get(_style_key(style, target, effects)):
    #             return style_obj
    #     return default
    @classmethod
    def resolve(cls, styles: list[str], target: str, effects: str="all", default: "BlockStyle | TextStyle | ChunkStyle | None"=None) -> "Generator[BlockStyle | TextStyle | ChunkStyle]":
        current = style_registry_ctx.get()
        if not styles and default is not None:
            yield default
        for style in styles:
            if style_obj := current.get(_style_key(style, target, effects)):
                yield style_obj
        # yield default



# def apply_style(target):
#     if styler:= StyleMeta.resolve(target.styles, "text", "content", default=TextStyle()):
#         target.content = styler(target.content)
#     if target.parent:
#         if styler:= StyleMeta.resolve(target.parent.styles, "text", "children"):
#             target.content = styler(target.content)
#     if styler:= StyleMeta.resolve(target.styles, "block", default=BlockStyle()):
#         target = styler(target)
    
#     return target


# def apply_chunk_style(target):
#     # lookup = {}
#     if target.parent is not None and target.parent.parent is not None:
#     # if styler:=lookup.get("chunk", ChunkStyle()):
#         if styler:=StyleMeta.resolve(target.parent.parent.styles, "chunk"):
#             target = styler(target)
#     return target


def apply_style(target):
    for styler in StyleMeta.resolve(target.styles, "text", "content", default=TextStyle()):
        target.content = styler(target.content)
    if target.parent:
        for styler in StyleMeta.resolve(target.parent.styles, "text", "children"):
            target.content = styler(target.content)
    for styler in StyleMeta.resolve(target.styles, "block", default=BlockStyle()):
        target = styler(target)
    return target


def apply_chunk_style(target):
    # lookup = {}
    if target.parent is not None and target.parent.parent is not None:
    # if styler:=lookup.get("chunk", ChunkStyle()):
        for styler in StyleMeta.resolve(target.parent.parent.styles, "chunk"):
            target = styler(target)
    return target


def render_text(target) -> str:
    prefix = ""
    content = ""
    children_content = ""
    postfix = ""
    if target.prefix:
        prefix = render(target.prefix)
    if target.content:
        content = render(target.content)
    if target.postfix:
        postfix = render(target.postfix)
    if target.children:
        children_content = "".join(render(c) for c in target.children)
    return f"{prefix}{content}{children_content}{postfix}"


def render(target) -> str:    
    if type(target) is str:
        return target
    elif type(target) in (int, float, bool):
        return str(target)
    elif type(target) is list:
        return "".join(render(c) for c in target)
    # elif type(target) is Block:
    elif isinstance(target, Block):
        target = apply_style(target)
        prefix = ""
        content = ""
        postfix = ""
        children_content = ""
        if target.prefix is not None:
            prefix = render_text(target.prefix)
        if target.content is not None:
            content = render_text(target.content)
        if target.postfix is not None:
            postfix = render_text(target.postfix)
        if target.children:
            cl = [render(c) for c in target.children]
            children_content = "".join(cl)
            # children_content = textwrap.indent(children_content, "  ")
        return f"{prefix}{content}{children_content}{postfix}"
    elif type(target) is BlockSent:
        return render_text(target)

    elif type(target) is BlockChunk:
        if target.content is not None:
            target = apply_chunk_style(target)
            return f"{target.prefix}{render(target.content)}{target.postfix}"
        else:
            return ""
    else:
        raise ValueError(f"Unknown type: {type(target)}") 
    

class BlockStyle(metaclass=StyleMeta):
    styles = []
    target = "block"
    effects = "all"
    
    def __call__(self, block: Block):
        for child in block.children:
            child.prefix.insert("\n", 0)
        return block
    
    
class TextStyle(metaclass=StyleMeta):
    styles = []
    target = "text"
    effects = "all"
    
    def __call__(self, text: BlockSent):
        return text
    
    
class ChunkStyle(metaclass=StyleMeta):
    styles = []
    target = "chunk"
    effects = "all"
    
    def __call__(self, chunk: BlockChunk):
        if chunk.index > 0:
            chunk.prefix += " "
        return chunk
    
    
class ChunkStreamStyle(ChunkStyle):
    styles = ["stream"]
    
    def __call__(self, chunk: BlockChunk):        
        chunk.content = chunk.content.replace("\n", "")
        return chunk
    

    
    
class BlockStreamViewStyle(BlockStyle):
    styles = ["stream-view"]
    
    def __call__(self, block: Block):
        # block = super().__call__(block)
        block.prefix.insert("\n", 0)
        block.postfix.insert("\n", 0)
        return block
    
    
class Markdown(TextStyle):
    styles = ["md"]
    effects = "content"
    
    def __call__(self, text: BlockSent):
        text.prefix = "# "
        # text.postfix = "\n"
        return text
        # return Block(
        #     prefix=Block("#", postfix=" "),
        #     content=block,
        #     postfix="\n"            
        # )
# class BlockPath(BlockStyle):
    
#     styles = ["path"]
#     # effects = "content"
    
#     def __call__(self, block: Block):
#         block.prefix += f"{block.path} "
#         return block   



class PathStyle(TextStyle):
    
    styles = ["path"]
    effects = "all"
    
    def __call__(self, text: BlockSent):
        text.prefix += ".".join([str(p) for p in text.parent.path]) + ": "
        return text   


class AstrixList(TextStyle):
    styles = ["ast-li"]
    effects = "children"
    
    def __call__(self, text: BlockSent):
        text.prefix ="* " + text.prefix
        return text





class NumberedList(TextStyle):
    styles = ["numbered-list", "num-li"]
    effects = "children"
    
    def __call__(self, text: BlockSent):
        text.prefix += f"{text.index + 1}. "
        return text



class XMLChildrenStyle(BlockStyle):
    styles = ["xml"]
    effects = "children"
    
    def __call__(self, block: Block):
        for child in block.children:
            child.prefix.insert("\n", 0)
        return block
    
    
# class XMLChunkStyle(ChunkStyle):
#     styles = ["xml"]
    
#     def __call__(self, chunk: BlockChunk):
#         if chunk.index > 0:
#             chunk.prefix += " "
#         return chunk
    
    
class XMLStyle(BlockStyle):
    styles = ["xml"]
    # effects = "content"
    
    
    def render_attr(self, attr: AttrBlock):
        content = f"{attr.name}=\""
        instructions = ""
        if attr.type in (int, float):
            instructions = " wrap in quotes"
        content += f"(\"{attr.type.__name__}\"{instructions}) {attr.description}"
        
        if attr.description:
            content += f" {attr.description}"
        if attr.gt is not None:
            content += f" gt={attr.gt}"
        if attr.lt is not None:
            content += f" lt={attr.lt}"
        if attr.ge is not None:
            content += f" ge={attr.ge}"
        if attr.le is not None:
            content += f" le={attr.le}"
        return content + "\""
    
    def render_all_attrs(self, block: Block):
        attrs = ""
        for attr in block.attrs.values():
            attrs += " " + self.render_attr(attr)
        return attrs

    
    def __call__(self, block: Block):        
        if not block.children:
            block.content.prefix = f"<"
            block.content.postfix = self.render_all_attrs(block) + f"/>"
        else:
            block = super().__call__(block)
            block.content.prefix = f"<"        
            block.content.postfix = self.render_all_attrs(block) + f">"
            block.postfix = block.content.copy()
            block.postfix.prefix = f"\n</"
            block.postfix.postfix = f">"
        return block
    
    
    
class XSDStyle(BlockStyle):
    styles = ["xsd"]
    
    def __call__(self, block: Block):
        with Block(prefix="\n") as s:            
            with s("xs:element", style="xml") as e:
                e += ' name=\"', block.content, '\"'
                if block.attrs:
                    with e("xs:complexType", style="xml") as ct:
                        for attr in block.attrs.values():
                            ct /= f"<xs:attribute name=\"{attr.name}\" type=\"{attr.type.__name__}\"/>"
                s.extend(block.children)
        return s
    
    
    