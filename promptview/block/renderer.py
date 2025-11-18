

import textwrap
from .style import BulletType, StyleManager
from typing import TYPE_CHECKING, Any, List, Literal, Protocol, Type, TypedDict, Union

from ..utils.string_utils import int_to_roman
from .block7 import Block





class RendererMeta(type):
    
    _renderers: dict[str, Type["Renderer"]] = {}
    _content_renderers: dict[str, Type["ContentRenderer"]] = {}
    _item_renderers: dict[str, Type["ItemsRenderer"]] = {}
    
    def __new__(cls, name, bases, dct):        
        renderer = super().__new__(cls, name, bases, dct)
        if name not in ["Renderer", "ContentRenderer", "ItemsRenderer"]:
            if "tags" not in dct:
                raise ValueError("Tags must be defined")
            for tag in renderer.tags:
                # if tag in cls._renderers:
                #     raise ValueError(f"Renderer {tag} already defined")
                cls._renderers[tag] = renderer
        return renderer
    
    def __getitem__(self, key: str) -> Type["Renderer"]:
        return self._renderers[key]


class Renderer(metaclass=RendererMeta):
    tags: list[str]
    target: Literal["content", "items"]
    
    def __call__(self, *args: Any, **kwargs: Any) -> str | List[str]:
        ...



class ContentRenderer(Renderer):
    tags: list[str]
    target = "content"
    
    def __call__(self, block: Block, inner_content: List[str], depth: int) -> str:
        raise NotImplementedError("Subclass must implement this method")
 
    
class ItemsRenderer(Renderer):
    tags: list[str]
    target = "items"
        
    def __call__(self, block: Block, inner_content: List[str], depth: int) -> List[str]:
        raise NotImplementedError("Subclass must implement this method")
    


class IndentRenderer(ContentRenderer):
    tags = ["block"]
    target = "content"
    
    def __call__(self, block: Block, inner_content: List[str], depth: int) -> str:
        if not block.content:
            return "\n".join(inner_content)
        content = block.content
        if inner_content:
            content += "\n" + textwrap.indent("\n".join(inner_content), "   ")
        return content


class TupleRenderer(ContentRenderer):
    tags = ["tuple", "tuple-col"]
    
    def __call__(self, block: Block, inner_content: List[str], depth: int) -> List[str]:
        # if block.inline_style.get("tuple-col"):
        #     return "(" + ",\n".join(inner_content) + "\n)"
        # else:
        #     return "(" + ", ".join(inner_content) + ")"
        if block.inline_style.get("tuple-col"):
            return ["(" + ",\n".join(inner_content) + "\n)"]
        else:
            return ["(" + ", ".join(inner_content) + ")"]
        
        
class FuncRenderer(ContentRenderer):
    tags = ["func", "func-col"]
    
    def __call__(self, block: Block, inner_content: List[str], depth: int) -> str:
        if block.inline_style.get("block_type") == "func-col":
            return block.content + "(\n" + textwrap.indent(",\n".join(inner_content), "    ") + "\n)"
        else:
            return block.content + "(" + ", ".join(inner_content) + ")"



# class RowRenderer(ContentRenderer):
#     tags = ["row"]
    
#     def __call__(self, block: Block, inner_content: List[str], depth: int) -> str:
#         return " ".join(inner_content)


# class RowItemsRenderer(ItemsRenderer):
#     tags = ["row"]
    
#     def __call__(self, block: Block, inner_content: List[str], depth: int) -> List[str]:
#         return inner_content



class MarkdownTitleRenderer(ContentRenderer):
    tags = ["md"]
    
    def __call__(self, block: Block, inner_content: List[str], depth: int) -> str:
        content = ""
        if block.content:
            if block.is_block:
                content = f"{'#' * (depth + 1)} {block.content}" + "\n" 
            else:
                content = block.content
        if inner_content:
            content += "\n".join(inner_content)
            
        # if block.is_block:
        #     content += "\n"
        return content
        


class MarkdownParagraphRenderer(ItemsRenderer):
    tags = ["p"]
    
    def __call__(self, block: Block, inner_content: List[str], depth: int) -> List[str]:
        return inner_content



def get_bullet_prefix(bullet_type: BulletType, idx: int) -> str:
    if bullet_type == "" or bullet_type == "number":
        return f"{idx}. "
    elif bullet_type == "alpha":
        return f"{chr(96+idx)}. "
    elif bullet_type == "roman_upper":
        return int_to_roman(idx, upper=True) + ". "
    elif bullet_type == "roman":
        return int_to_roman(idx, upper=False) + ". "
    elif bullet_type == "*":
        return "* "
    elif bullet_type == "-":
        return "- "
    else:
        return f"{bullet_type} "

class MarkdownListRenderer(ItemsRenderer):
    tags = ["list","list:number", "list:alpha", "list:roman", "list:roman_lower", "list:*", "list:-"]
    
    def indent(self, item: str) -> str:
        lines = item.splitlines()
        if len(lines) == 1:
            return lines[0]
        else:
            return lines[0] + "\n" + textwrap.indent("\n".join(lines[1:]), "   ")
            
    def __call__(self, block: Block, inner_content: List[str], depth: int) -> List[str]:
        if not inner_content:
            return []
        bullet_type = block.inline_style.get("bullet_type", "number")
        if not bullet_type:
            raise ValueError("Bullet type not found")
        return [get_bullet_prefix(bullet_type, idx + 1) + self.indent(item) for idx, item in enumerate(inner_content)]



# class MarkdownListContentRenderer(ContentRenderer):
#     tags = ["li"]
    
#     def __call__(self, block: BaseBlock, inner_content: List[str], depth: int) -> str:
#         bullet_type = block.inline_style.get("bullet_type", "number")
#         if not bullet_type:
#             raise ValueError("Bullet type not found")
#         return get_bullet_prefix(bullet_type, idx) + item


class XMLRenderer(ContentRenderer):
    tags = ["xml"]
    
    def __call__(self, block: Block, inner_content: List[str], depth: int) -> str:
        if not block.content:
            return "\n".join(inner_content) or ""
        if block.is_block:
            if block.attrs:
                attrs = " " + " ".join([f"{k}=\"{v}\"" for k, v in block.attrs.items()])
            else:
                attrs = ""
            if inner_content:
                inner_content_str = textwrap.indent("\n".join(inner_content), "  ")
                return f"<{block.content}{attrs}>\n" + inner_content_str + f"\n</{block.content}>"
            else:
                return f"<{block.content}{attrs}/>"        
        return block.content
        


    
# class ListRenderer(Renderer):
    
#     def can_render(self, block: BaseBlock) -> bool:
#         return block.tags == ["list"]
    
#     def render(self, block: BaseBlock) -> str:
#         return block.content

# class MarkdownStyles(TypedDict):
#     title: bool
#     block_type: BlockType
#     bullet_type: BulletType
#     heading_level: int
#     indent_depth: bool
