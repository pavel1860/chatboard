from ast import TypeVar
import copy
import json
import textwrap
from typing import Any, Callable, Generator, Generic, List, Literal, Self, Type
from pydantic_core import core_schema
from pydantic import BaseModel, GetCoreSchemaHandler
from ...utils.model_utils import is_list_type
from .base_blocks import BaseBlock, BaseContent, BlockSequence
from ...utils.string_utils import camel_to_snake
import annotated_types



def parse_style(style: str | List[str] | None) -> List[str]:
    if isinstance(style, str):
        return list(style.split(" "))
    elif type(style) is list:
        return style
    else:
        return []


# def dict_to_block(data: dict) -> BaseBlock:
#     if data.get("_type") == "Block":
#         return Block.model_validate(data)
#     elif data.get("_type") == "BlockSent":
#         return BlockSent.model_validate(data)
#     elif data.get("_type") == "BlockChunk":
#         return BlockChunk.model_validate(data)
#     else:
#         raise ValueError(f"Invalid block type: {data.get('_type')}")

def _block_traverse_tag(block: "Block", tag: str):
    block_list = BlockList()
    for blk in block.traverse():
        if tag in blk.tags:
            block_list.append(blk)
    return block_list


def _block_list_find_tag(block_list: "BlockList", tag: str):
    new_block_list = BlockList()
    for blk in block_list:
        sub_bl = _block_traverse_tag(blk, tag)
        new_block_list.extend(sub_bl)
    return new_block_list

class BlockChunk(BaseBlock[str]):
    
    __slots__ = [
        "logprob",
        "type",
    ]
    
    def __init__(
        self, 
        content: str, 
        logprob: float | None = None, 
        prefix: str | None = None, 
        postfix: str | None = None,
        parent: "BlockSequence | None" = None,
        id: str | None = None,
    ):
        # if content.endswith("\n"):
        #     content = content[:-1]
        #     postfix = "\n"
        super().__init__(content, prefix=prefix or "", postfix=postfix or "", id=id, parent=parent)
        self.logprob: float | None = logprob
        self.type: Type = type(content)
        
        
    @property
    def is_eol(self) -> bool:
        if self.type is str and self.content is not None:
            return self.content.endswith("\n")
        return False
    
    
    def repr_tree(self, verbose: bool = False):
        logprob = f" logprob={self.logprob:.3f}" if self.logprob is not None else ""
        content = self.content.replace("\n", "\\n")
        space = " empty" if self.is_space() else ""
        # prefix = " prefix='" + self.prefix.replace("\n", "\\n") + "'" if self.prefix is not None else ""
        # postfix = " postfix='" + self.postfix.replace("\n", "\\n") + "'"if self.postfix is not None else ""
        return f"{self.path}  BlockChunk('{content}'{logprob}){space}"
    
    def __repr__(self):
        return f"BlockChunk(content={self.content} , logprob={self.logprob})"
    
    def copy(
        self,
        overrides: dict[str, Any] | None = None,
        copy_parent: bool = False,
        copy_id: bool = False,
    ):
        return BlockChunk(
            content=self.content if not overrides or "content" not in overrides else overrides["content"], 
            logprob=self.logprob if not overrides or "logprob" not in overrides else overrides["logprob"], 
            prefix=self.prefix if not overrides or "prefix" not in overrides else overrides["prefix"], 
            postfix=self.postfix if not overrides or "postfix" not in overrides else overrides["postfix"], 
            parent=self.parent if copy_parent else None,
            id=self.id if copy_id else None
        )
        
    def is_space(self) -> bool:
        return self.content.isspace()
    
    @classmethod
    def model_validate(cls, data: Any) -> "BlockChunk":        
        return BlockChunk(
            content=data.get("content"),
            logprob=data.get("logprob"),
            prefix=data.get("prefix"),
            postfix=data.get("postfix"),
        )


SentContent = list[BlockChunk] | BlockChunk | str | None

class BlockSent(BlockSequence[str, BlockChunk]):
    
    
    
    def __init__(
        self,
        content: str | list[BlockChunk] | None = None,
        children: list[BlockChunk] | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
        parent: "BlockSequence | None" = None,
        id: str | None = None,

    ):
        children = children or []
        if content is None:
            content = ""
        elif isinstance(content, str):
            pass
        elif isinstance(content, list):            
            children = content
            content = ""
        else:
            raise ValueError(f"Invalid content type: {type(content)}")
        super().__init__(content=content,children=children, prefix=prefix or "", postfix=postfix or "", parent=parent, id=id)
    
    
    def promote_content(self, content: SentContent, prefix: str | None = None, postfix: str | None = None) -> BlockChunk:
        if isinstance(content, str):
            return BlockChunk(content, prefix=prefix, postfix=postfix)
        elif isinstance(content, int):
            return BlockChunk(str(content), prefix=prefix, postfix=postfix)
        elif isinstance(content, float):
            return BlockChunk(str(content), prefix=prefix, postfix=postfix)
        elif isinstance(content, bool):
            return BlockChunk(str(content), prefix=prefix, postfix=postfix)
        elif isinstance(content, BlockSent):
            raise ValueError("Cannot promote BlockSent to BlockChunk")
        elif isinstance(content, BlockChunk):
            if prefix is not None:
                content.prefix = prefix
            if postfix is not None:
                content.postfix = postfix
            return content
        else:
            raise ValueError(f"Invalid content type: {type(content)}")
    
    # def index_of(self, child: Block) -> int | None:
    #     if child
    #     return super().index_of(child)
    
    
    @property
    def is_no_content(self) -> bool:
        return len(self.children) == 0
    @property
    def logprob(self) -> float | None:
        logprob = sum(blk.logprob for blk in self.children if blk.logprob is not None) or 0
        return logprob
    
    
    def is_last_eol(self) -> bool:
        if len(self) == 0:
            return True
        return self.children[-1].is_eol
    
    def is_space(self) -> bool:
        return all(c.is_space() for c in self.children)
    
    def strip(self) -> "BlockSent":
        start_index, end_index = self.strip_range()
        chunks = self.children[start_index:end_index]
        return self.copy(overrides={"children": chunks})
    
    
    def render(self, verbose: bool = False) -> str:
        from .renderers3 import render
        return render(Block(self))
    
    def print(self, verbose: bool = False):
        print(self.render(verbose=verbose))
        
    def copy(
        self,
        overrides: dict[str, Any] | None = None,
        copy_id: bool = False,
        copy_parent: bool = False,
    ):
        return BlockSent(            
            content=self.content if not overrides or "content" not in overrides else overrides["content"],
            children=[c.copy() for c in self.children] if not overrides or "children" not in overrides else overrides["children"],
            prefix=self.prefix if not overrides or "prefix" not in overrides else overrides["prefix"],
            postfix=self.postfix if not overrides or "postfix" not in overrides else overrides["postfix"],
            id=self.id if copy_id else None,
            parent=self.parent if copy_parent else None,
        )
        
        
    def repr_tree(self, verbose: bool = False): 
        # prefix = " prefix='" + self.prefix.replace("\n", "\\n") + "'" if self.prefix is not None else ""
        # postfix = " postfix='" + self.postfix.replace("\n", "\\n") + "'"if self.postfix is not None else ""
        space = " empty" if self.is_space() else ""
        content = "content=" + self.content if self.content else ''
        res = f"{self.path}  BlockSent({content}id={self.id}){space}"
        for child in self.children:
            res += f"\n{child.repr_tree(verbose=verbose)}"
        return res
    
    @classmethod
    def model_validate(cls, data: dict) -> "BlockSent":
        if data.get("children") is None or len(data.get("children")) == 0:
            children = []
        else:
            children = [BlockChunk.model_validate(c) for c in data.get("children")]
        return BlockSent(
            content=data.get("content"),
            children=children,
            prefix=data.get("prefix"),
            postfix=data.get("postfix"),
        )
        
    def __iadd__(self, other: SentContent):
        other = self.promote_content(other)
        self.append(other)
        return self
    
BlockContent = BlockSent | BlockChunk | BaseContent 




def pydantic_object_description(name: str, obj: BaseModel) -> "Block":
    with Block(name, style="xml", tags=["tool", camel_to_snake(obj.__class__.__name__)]) as b:
        b.field("name", camel_to_snake(obj.__class__.__name__))
        # b.field("name", description=camel_to_snake(obj.__class__.__name__))
        # with b("name", style="xml", tags=["tool_name"]) as tn:
            # tn /= camel_to_snake(obj.__class__.__name__)
        for key, value in obj.model_dump().items():
            with b(key, style="xml", tags=[key, "field"]) as bf:
                bf /= value
    return b

# def pydantic_class_description(cls: Type[BaseModel], examples: list[BaseModel] | None = None) -> "Block":
#     tool_name = camel_to_snake(cls.__name__)
#     with Block(f"Tool: {tool_name}", style="md", tags=["tool", tool_name]) as b:
#         with b("tool", tags=["tool_name"], style="xml-def") as tn:
#             tn.field("name", tool_name)
#         if not cls.__doc__:
#             raise ValueError(f"description is required for Tool {cls.__name__}")
#         b /= cls.__doc__
#         with b("Argument tags", style="md") as args:
#             for field_name, field_info in cls.model_fields.items():
#                 if not field_info.description:
#                     raise ValueError(f"description is required for field '{field_name}' in Tool {cls.__name__}")
#                 with args(field_name, style="xml-def", tags=["field_name", field_name]) as bf:
#                     bf /= "description: ", field_info.description
#                     bf /= "type: ", field_info.annotation.__name__
            
#         if examples:
#             example_str = "Examples" if len(examples) > 1 else "Example"
#             with b(example_str, style="md", tags=["examples"]) as exs:
#                 for example in examples:                    
#                     exs /= pydantic_object_description(example)
#     return b

# def pydantic_class_description(name: str, cls: Type[BaseModel]) -> "Block":
#     tool_name = camel_to_snake(cls.__name__)
#     with Block(f"{name}: {tool_name}", style="md", tags=[tool_name]) as b:
#         with b("tool", tags=["tool_name"], style="xml-def") as tn:
#             tn.field("name", tool_name)
#         if not cls.__doc__:
#             raise ValueError(f"description is required for Tool {cls.__name__}")
#         b /= cls.__doc__
#         with b("Argument tags", style="md") as args:
#             for field_name, field_info in cls.model_fields.items():
#                 if not field_info.description:
#                     raise ValueError(f"description is required for field '{field_name}' in Tool {cls.__name__}")
#                 with args(field_name, style="xml-def", tags=["field_name", field_name]) as bf:
#                     bf /= "description: ", field_info.description
#                     bf /= "type: ", field_info.annotation.__name__            
#     return b



def pydantic_class_description(name: str, cls: Type[BaseModel], key_field: str, key_type: Type, class_model: Type[BaseModel] | None = None) -> "BlockSchema":
    if key_field is None:
        raise ValueError("key_field is required")
    tool_name = camel_to_snake(cls.__name__)
    with BlockSchema(name, type=class_model, tags=[tool_name], style="xml") as b:
        b.field(key_field, tool_name, type=key_type)
        if not cls.__doc__:
            raise ValueError(f"description is required for Tool {cls.__name__}")
        b(cls.__doc__, tags=["description"])
        
        for field_name, field_info in cls.model_fields.items():
            if not field_info.description:
                raise ValueError(f"description is required for field '{field_name}' in Tool {cls.__name__}")
            with b.view(field_name, type=field_info.annotation, tags=[field_name, "field"]) as bf:
                bf /= field_info.description
    return b


KIND = TypeVar("KIND")

def _copy_kind_aux[KIND](
    target: "Block",
    kind: KIND, 
    # include_subkinds: bool = False,       
) -> KIND | list[KIND] | None:
    children = []
    children_copies = [_copy_kind_aux(c, kind) for c in target.children]
    for cld in children_copies:
        if cld is None:
            continue
        elif type(cld) is list:
            children += cld
        else:
            children.append(cld)
    
    # if type(target) is kind or (include_subkinds and isinstance(target, kind)):
    if isinstance(target, kind):
        block_copy = target.copy(overrides={"children": children, "parent": None})
        for child in children:
            child.parent = block_copy
        # for c in children:
        #     block_copy.append(c)
        return block_copy    
    if not children:
        return None
    return children




class Block(BlockSequence[BlockSent, "Block"]):
    
    __slots__ = [
        "role",
        "tags",
        "styles",
        "attrs",
        "postfix",
        "prefix",
        "artifact_id",
        "schema",
        "model",
    ]
    
    def __init__(
        self, 
        content: BlockContent | None = None,
        children: list["Block"] | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
        styles: list[str] | None = None,
        attrs: dict[str, str] | None = None,
        id: str | None = None,
        prefix: BlockSent | str | None = None,
        postfix: BlockSent | str | None = None,
        parent: "Block | None" = None,
        artifact_id: int | None = None,
        is_wrapper: bool = False,
        schema: "BlockSchema | None" = None,
    ):
        styles = styles or parse_style(style)
        super().__init__(
            content=self._init_content(content),
            children=children or [], 
            parent=parent, 
            prefix=self._parse_sent(prefix), 
            postfix=self._parse_sent(postfix), 
            id=id,
            is_wrapper=is_wrapper or len([s for s in styles if s.startswith("wrapper")]) > 0,
        )
        self.role: str | None = role
        self.tags: list[str] = tags or []
        self.styles: list[str] = styles
        # self.attrs: dict[str, AttrBlock] = get_attrs(attrs)
        # self.attrs: dict[str, str] | None = attrs or {}
        self.attrs: dict[str, AttrBlock] = get_attrs(attrs) if attrs is not None else {}
        self.artifact_id: str | None = artifact_id
        self.content = self._init_content(content) 
        # if content is None:
        #     self.content = BlockSent(parent=self)
        # elif isinstance(content, BlockSent):
        #     self.content = content
        # elif isinstance(content, list):
        #     self.content = BlockSent(parent=self)
        #     self.content.extend(content)
        # else:
        #     self.content = BlockSent(parent=self)
        #     self.content.append(content)
        self.schema = schema
        self.model = None
        
    @property
    def tag(self):
        """
        returns the first tag
        """
        if self.tags:
            return self.tags[0]
        return None
    
    
    @property
    def tag_path(self) -> list[str]:
        return list(reversed([b.tags[0] for b in self.traverse_path() if b.tags]))
    
    
    def get_field(self, name: str) -> Any:
        return self.attrs.get(name).value
    
    
    @property
    def value(self) -> Any:
        if self.schema is None:
            raise ValueError("Schema is required")
        return self.schema.parse(self)
        
            
    

    @property
    def subtree_size(self) -> int:
        total = len(self.children)
        for child in self.children:
            total += child.subtree_size
        return total
    
    @property
    def is_no_content(self) -> bool:
        return self.content.is_no_content
        
    def _parse_sent(self, sent: BlockSent | str | None) -> BlockSent:
        if sent is None:
            return BlockSent()
        elif isinstance(sent, str):
            return BlockSent(sent)
        elif isinstance(sent, BlockSent):
            return sent
        else:
            raise ValueError(f"Invalid sent type: {type(sent)}")
        
        
    @property
    def kind(self) -> str:
        return "block"

    def _init_content(self, content: BlockContent | None) -> BlockSent:
        if content is None:
            return BlockSent(parent=self)
        elif isinstance(content, BlockSent):
            content.parent = self
            return content
        elif isinstance(content, list):
            sent = BlockSent(parent=self)
            sent.extend(content)
            return sent
        else:
            sent = BlockSent(parent=self)
            sent.append(content)
            return sent
    
    def with_style(self, style: str | list[str]) -> "Block":
        if isinstance(style, str):
            self.styles.append(style)
        elif isinstance(style, list):
            self.styles.extend(style)
        else:
            raise ValueError(f"Invalid style type: {type(style)}")
        return self
        
    def promote_content(self, content: "Block | BlockSent | BaseContent", prefix: BlockSent | None = None, postfix: BlockSent | None = None) -> "Block":
        if isinstance(content, str):
            return Block(content, prefix=prefix, postfix=postfix)
        elif isinstance(content, int):
            return Block(str(content), prefix=prefix, postfix=postfix)
        elif isinstance(content, float):
            return Block(str(content), prefix=prefix, postfix=postfix)
        elif isinstance(content, bool):
            return Block(str(content), prefix=prefix, postfix=postfix)
        elif isinstance(content, Block):
            # content = content.copy()
            # content = copy.deepcopy(content)
            if prefix is not None:
                content.prefix = prefix
            if postfix is not None:
                content.postfix = postfix
            return content
        elif isinstance(content, BlockSent):
            return Block(content, prefix=prefix, postfix=postfix)
        elif isinstance(content, BlockChunk):
            return Block(content, prefix=prefix, postfix=postfix)        
        else:
            raise ValueError(f"Invalid content type: {type(content)}")
        
    
    def index_of(self, child: BaseBlock) -> int | None:
        if isinstance(child, Block):
            return super().index_of(child)
        else:
            return self.index
        
        
    @property
    def logprob(self) -> float | None:
        logprob = sum(blk.logprob for blk in self.children if blk.logprob is not None) or 0
        if self.content is not None:
            logprob += self.content.logprob or 0
        if self.postfix is not None:
            logprob += self.postfix.logprob or 0
        if self.prefix is not None:
            logprob += self.prefix.logprob or 0
        return logprob

    
    
    def view(
        self, 
        name: str, 
        type: Type | None = None, 
        attrs: dict[str, str] | None = None, 
        tags: list[str] | None = None, 
        style: str | None = None, 
        is_wrapper: bool = False
    ) -> "BlockSchema":
        # block = FieldBlock(name, type, attrs=attrs)
        block = BlockSchema(
            name,
            type=type,
            attrs=attrs,
            role=self.role,
            parent=self.parent,
            tags=tags,
            styles=["xml"] if style is None else parse_style(style),
            is_wrapper=is_wrapper,
            # styles=["xsd"],
        )
        self.append(block)
        return block
    
    def view_list(self, name: str, attrs: dict[str, str] | None = None, tags: list[str] | None = None) -> "BlockListSchema":
        # block = BlockSchema(
        #     name,
        #     type=list[Any],
        #     attrs=attrs,
        #     role=self.role,
        #     parent=self.parent,
        #     is_wrapper=True,
        #     tags=tags,
        #     # styles=["xsd"],
        # )
        block = BlockListSchema(
            name,
            attrs=attrs,
            role=self.role,
            parent=self.parent,
            tags=tags,
            styles=["xml"],
        )
        self.append(block)
        return block
    
    # def schema_view(self, target: BaseModel, tags: list[str] | None = None) -> "BlockSchema":
        
    #     block = BlockSchema(
    #         name=camel_to_snake(target.__class__.__name__),
    #         type=target.__class__,
    #         # attrs=attrs,
    #         role=self.role,
    #         parent=self.parent,
    #         tags=[camel_to_snake(target.__class__.__name__)] + (tags or []),
    #         is_wrapper=True,
    #         # styles=["xml"],
    #     )
    #     self.append(block)
    #     block.describe(target)
    #     return block
    
    # def describe(self, target: BaseModel | Type[BaseModel], examples: list[BaseModel] | None = None) -> "Block":
    #     if isinstance(target, BaseModel):
    #         block = pydantic_object_description(target)
    #     elif isinstance(target, type) and issubclass(target, BaseModel):
    #         block = pydantic_class_description(target, examples)
    #     else:
    #         raise ValueError(f"Invalid target type: {type(target)}")
    #     self.append(block)
    #     return block
    
    
    def example(self, value: dict, schema: "BlockSchema | None" = None, style: str | None = "md") -> "Block":
        from .block_builder import SchemaBuildContext
        ctx = SchemaBuildContext(schema or self)
        inst = ctx.inst_dict(value)
        with Block("Example", style=style or "md") as ex:
            ex /= inst
        self.append(ex)
        return ex
    
    
    
    def field(
        self, 
        name: str,
        value: Any | None = None,
        type: Type | None = None, 
        description: str | None = None,
        gt: int | float | None = None,
        lt: int | float | None = None,
        ge: int | float | None = None,
        le: int | float | None = None,
    ) -> "None":
        if gt is not None: annotated_types.Gt(gt)
        if lt is not None: annotated_types.Lt(lt)
        if ge is not None: annotated_types.Ge(ge)
        if le is not None: annotated_types.Le(le)
        self.attrs[name] = AttrBlock(
            name=name,
            value=value,
            _type=type,
            description=description,
            gt=gt,
            lt=lt,
            ge=ge,
            le=le,
        )
        
        
    def get_first(self, tag: str):
        tags = tag.split(".")
        curr = self
        for tag in tags:
            for blk in self.traverse():
                if tag in blk.tags:
                    curr = blk
                    break
        return curr

    
    def get_all2(self, tag: str):
        tags = tag.split(".")
        block_list = BlockList()        
        curr = self
        for tag in tags:
            block_list = BlockList()
            for blk in self.traverse():
                if tag in blk.tags:
                    block_list.append(blk)
        return block_list
        
        
    def get_all(self, tags: str | list[str]):
        if type(tags) is str:            
            tags = tags.split(".")
        block_list = BlockList()        
        curr = self
        for idx, tag in enumerate(tags):
            if idx == 0:
                block_list = _block_traverse_tag(curr, tag)
            else:
                block_list = _block_list_find_tag(block_list, tag)
                
        return block_list
    
    def get_one(self, tag: str | list[str]):
        res = self.get_all(tag)
        if not res:
            raise ValueError(f'path "{tag}" does not exists')
        return res[0]
    
    def get_one_or_none(self, tag:str):
        res = self.get_all(tag)
        if not res:
            return None
        return res[0]
        
            
            
    def get(self, tag: str):
        tag = tag.lower()
        if tag in self.tags:
            return self
        for child in self.children:
            if isinstance(child,Block):
                if tag in child.tags:
                    return child                            
                if (block:= child.get(tag)) is not None:
                    return block
        return None
    
    def get_last(self, tag: str):
        tag = tag.lower()
        if tag in self.tags:
            return self
        candidates = []
        for child in reversed(self.children):
            if isinstance(child, Block):
                if tag in child.tags:
                    candidates.append(child)                            
                if (block:= child.get_last(tag)) is not None:
                    return block
        if not candidates:
            return None
        return candidates[-1]

        
    def _should_add_sentence(self):
        if not len(self):
            return True
        if not isinstance(self.last, BlockSent):
            return True
        if self.children[-1].is_last_eol:
            return True
        return False
    
    def is_last_eol(self) -> bool:
        if len(self.children) == 0:
            return True
        return self.children[-1].content.is_last_eol()

    
    def inline_append(self, content: "BlockChunk | BaseContent"):
        if self.last_child is None:
            raise ValueError("Block has no children")
        return self.last_child.content.append(content)
    # def inline_append(self, content: "Block | BlockSent | BaseContent", sep: str = " "):        
    #     if isinstance(content, Block):
    #         self.append(content, sep=sep)
    #         return self.children[-1]
    #     elif isinstance(content, BlockSent):
    #         block = Block(content=content)
    #         self.append(block, sep=sep)
    #         return self.children[-1]
    #     else:
    #         if self.is_last_eol():
    #             self.append(content, sep=sep)
    #             return self.children[-1]
    #         else:
    #             last = self.children[-1]
    #             if last.content.is_last_eol:
    #                 self.append(content, sep=sep)
    #                 return self.children[-1]
    #             else:
    #                 last.content.append(content, sep=sep)
    #                 return last.content.children[-1]



    def gather_trees(self, is_target: Callable[[BaseBlock], bool] | None = None, clone_target_node = None, connect_target_node = None) -> "list[BaseBlock]":
        """Return a forest containing only target-type nodes, attached under their
        nearest target-type ancestor from the original tree."""
        dummy_children: List[BaseBlock] = []
        stack: List[Block] = []  # stack of cloned target nodes
        
        
        def _connect_target_node(u: Block, parent: Block):
            if connect_target_node:
                connect_target_node(u, parent)
            else:
                parent.append(u)

        def dfs(u: Block):
            created = None
            if is_target(u):
                created = clone_target_node(u)
                if stack:
                    # stack[-1].append(created)
                    _connect_target_node(created, stack[-1])
                else:
                    dummy_children.append(created)
                stack.append(created)

            if isinstance(u, Block):
                for child in u.children:
                    dfs(child)
                    
            if created is not None:
                stack.pop()

        dfs(self)
        if not dummy_children:
            raise ValueError("No target nodes found")
        # res = Block(children=BlockList(dummy_children))
        return dummy_children

    def model_dump(self):
        try:
            dump = {
                **super().model_dump(),
                "content": self.content.model_dump(),
                "prefix": self.prefix.model_dump(),
                "postfix": self.postfix.model_dump(),
                "styles": [s for s in self.styles],
                "tags": [t for t in self.tags],
                "attrs": {k: attr.model_dump() if isinstance(attr, AttrBlock) else attr for k, attr in self.attrs.items() if attr is not None},
                # "attrs": self.attrs,
                "role": self.role,  
                "is_wrapper": self.is_no_content,
            }
        except Exception as e:
            print(e)
            raise e
        return dump
    
    @classmethod
    def model_validate(self, data: dict):
        
        block = Block(
            content=BlockSent.model_validate(data.get("content")),
            children=[Block.model_validate(c) for c in data.get("children")],
            role=data.get("role"),
            tags=data.get("tags"),
            styles=data.get("styles"),
            attrs=data.get("attrs"),
            is_wrapper=data.get("is_wrapper"),
        )
        return block
    # def model_dump(self):
    #     dump = super().model_dump()
    #     dump["_type"] = "Block"
    #     dump["content"] = self.content.model_dump()
    #     dump["children"] = [c.model_dump() for c in self.children]
    #     dump["styles"] = self.styles
    #     dump["tags"] = self.tags
    #     dump["attrs"] = self.attrs
    #     dump["role"] = self.role
    #     dump["id"] = self.id
    #     dump["path"] = [p for p in self.path]
    #     dump["prefix"] = self.prefix
    #     dump["postfix"] = self.postfix
    #     dump["parent_id"] = self.parent.id if self.parent else None
    #     return dump
    
    
    
    def __enter__(self):        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    
    def __call__(
        self, 
        content: BaseContent | BaseBlock | list[str] | None = None, 
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
        attrs: "dict[str, AttrBlock] | None" = None,
        is_wrapper: bool = False,
    ) -> "Block":  
        if isinstance(content, Block):
            override = {}
            if role is not None:
                override["role"] = role
            if tags is not None:
                override["tags"] = tags
            if style is not None:
                override["styles"] = parse_style(style)
            if attrs is not None:
                override["attrs"] = attrs
            block = content.copy(overrides=override)
        else:
            block = Block(
                content,
                role=role,
                tags=tags,            
                parent=self,            
                style=style,
                attrs=attrs,
                is_wrapper=is_wrapper,
            )      
        self.append_child(block)
        return block
    
    
    def iter_kind(self, kinds: set[Type[BaseBlock]]) -> Generator[BaseBlock, None, None]:
        if self.__class__ in kinds:
            yield self
        for child in self.children:
            if isinstance(child, Block):
                yield from child.iter_kind(kinds)
            else:
                yield child
    
    def iter_schema(self):
        schema = self.copy_kind(BlockSchema)
        return schema.iter_kind(set([BlockSchema, BlockListSchema]))
        
    
    def _process_tuple_content(self, other: tuple[BaseContent, ...]):
        block = BlockSent()
        for o in other:
            block.append(o)
        return block
    
    def copy(
        self,
        overrides: dict[str, Any] | None = None,
        copy_id: bool = False,
        copy_parent: bool = False,
    ):
        return Block(
            content=self.content.copy() if not overrides or "content" not in overrides else overrides["content"],
            children=[c.copy() for c in self.children] if not overrides or "children" not in overrides else overrides["children"],
            attrs=self.attrs if not overrides or "attrs" not in overrides else overrides["attrs"],
            prefix=self.prefix if not overrides or "prefix" not in overrides else overrides["prefix"],
            postfix=self.postfix if not overrides or "postfix" not in overrides else overrides["postfix"],
            role=self.role if not overrides or "role" not in overrides else overrides["role"],
            tags=self.tags if not overrides or "tags" not in overrides else overrides["tags"],
            styles=self.styles if not overrides or "styles" not in overrides else overrides["styles"],
            id=self.id if copy_id else None,
            parent=self.parent if copy_parent else None,
            is_wrapper=self.is_wrapper if not overrides or "is_wrapper" not in overrides else overrides["is_wrapper"],
        )
        
    def is_space(self) -> bool:
        return self.content.is_space() and all(c.is_space() for c in self.children)
        
    def strip(self) -> "Block":
        self.content = self.content.strip()
        start_index, end_index = self.strip_range()
        return self.copy(overrides={"children": self.children[start_index:end_index]})
        
    
    # def copy_kind_children(
    #     self,
    #     kind: "Type[Block]",        
    # ):
    #     if type(self) is kind:
    #         block_copy = self.copy(overrides={"children": [], "parent": None})  # pyright: ignore[reportUnusedVariable]
    #     else:
    
    # def _copy_kind_aux(
    #     self,
    #     kind: "Type[Block]",        
    # ):
    #     children = []
    #     for cld in [c._copy_kind_aux(kind) for c in self.children]:
    #         if cld is None:
    #             continue
    #         if type(cld) is list:
    #             children += cld
    #         children.append(cld)
            
    #     if type(self) is kind:
    #         block_copy = self.copy(overrides={"children": children, "parent": None})
    #         # for c in children:
    #         #     block_copy.append(c)
    #         return block_copy
    #     if not children:
    #         return None
    #     return children
    
    def copy_kind(self, kind: "Type[Block]") -> "Block":
        res = _copy_kind_aux(self, kind)
        if type(res) is list:
            return Block(
                children=res
            )
        return res
            
            
        
    
        
    def traverse(self, target: Literal["schema", "block", "all"] = "all"):
        if target in ("block", "all"):
            yield self
        for child in self.children:
            if isinstance(child, Block):
                yield from child.traverse(target)
            else:
                yield child

        
    def _add_sent_content(self, sent: BlockSent):
        for c in sent.children:
            self.content.append(c)
            
    
    def __iadd__(self, other: BaseContent | BlockSent | tuple[BaseContent, ...]):
        if isinstance(other, tuple):
            for o in other:
                if isinstance(o, BlockSent):
                    self._add_sent_content(o)
                else:
                    self.content.append(o)
        elif isinstance(other, BlockSent):
            self._add_sent_content(other)
        else:
            self.content.append(other)
        return self
    
    def __iand__(self, other: BaseContent | BlockSent | BlockChunk | tuple[BaseContent, ...]):
        if isinstance(other, tuple):
            for o in other:
                self.content.append(o, sep="")
        else:
            self.content.append(other, sep="")
        return self
            
    
    
    def __itruediv__(self, other: "BaseContent | BlockChunk | Block | BlockSent | tuple[BaseContent, ...] | list[BaseContent]"):        
        if isinstance(other, tuple) or isinstance(other, list):
            other =self._process_tuple_content(other)        
        self.append(other)
        return self
    
        
    def __repr__(self) -> str:
        content = self.content.render() if self.content else ''
        tags = ','.join(self.tags) if self.tags else ''
        tags = f"[{tags}] " if tags else ''
        styles = "styles=[" + ','.join(self.styles) + "] " if self.styles else ''
        return f"{self.__class__.__name__}({tags}{styles}content={content}, children={self.children})"

    def repr_tree(self, verbose: bool = False):
        tags = ','.join(self.tags) if self.tags else ''
        tags = f"[{tags}] " if tags else ''
        role = f" role={self.role} " if self.role else ''
        space = " empty" if self.is_space() else ""
        styles = "styles=[" + ','.join(self.styles) + "] " if self.styles else ''
        # prefix = " prefix='" + self.prefix.replace("\n", "\\n") + "'" if self.prefix is not None else ""
        # postfix = " postfix='" + self.postfix.replace("\n", "\\n") + "'"if self.postfix is not None else ""
        res = f"{self.path}  Block({tags}{role}{styles}id={self.id}){space}"
        if self.content and verbose:
            # res += f"\ncontent-{self.content.repr_tree()}"
            res += f"\n{self.content.repr_tree()}"
        for child in self.children:
            res += f"\n{child.repr_tree(verbose=verbose)}"        
        return res
    
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            )
        )
        
    @staticmethod
    def _validate(v: Any) -> Any:
        if isinstance(v, Block):
            return v
        elif isinstance(v, dict):
            if "_type" in v and v["_type"] == "Block":
                return Block.model_validate(v)
        else:
            raise ValueError(f"Invalid block: {v}")

    @staticmethod
    def _serialize(v: Any) -> Any:
        if isinstance(v, Block):
            return v.model_dump()
        else:
            raise ValueError(f"Invalid block: {v}")

    
    
    
    def render(self, verbose: bool = False) -> str:
        from .renderers3 import render
        return render(self)

    def render_children(self, verbose: bool = False) -> str:
        from .renderers3 import render
        return render(self, children_only=True)

    def print(self, verbose: bool = False):
        print(self.render(verbose=verbose))
        
        
def get_attrs(attrs: dict[str, "str | AttrBlock"] | None) -> "dict[str, AttrBlock]":
    if attrs is None:
        return {}
    return {k: v if isinstance(v, AttrBlock) else AttrBlock(name=k, description=v) for k, v in attrs.items()}


def Attr(
    type: Type,
    description: str,
    name: str | None = None,
    gt: annotated_types.Gt | None = None,
    lt: annotated_types.Lt | None = None,
    ge: annotated_types.Ge | None = None,
    le: annotated_types.Le | None = None,
):
    return AttrBlock(
        name=name,
        type=type,
        description=description,
        gt=gt,
        lt=lt,
        ge=ge,
        le=le,
    )
    


TYPE_REGISTRY = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}


def parse_content(content: str, type: Type) -> Any:
    if type == int:
        return int(content)
    elif type == float:
        return float(content)
    elif type == bool:
        return bool(content)
    elif type == str:
        return content
    elif type == list:
        return content.split(",")
    elif type == dict:
        return json.loads(content)
    else:
        raise ValueError(f"Invalid type: {type}")

REVERSE_TYPE_REGISTRY = {v: k for k, v in TYPE_REGISTRY.items()}


class AttrBlock:
    name: str
    value: Any | None = None
    type: Type = str
    description: str | None = None
    gt: annotated_types.Gt | None = None
    lt: annotated_types.Lt | None = None
    ge: annotated_types.Ge | None = None
    le: annotated_types.Le | None = None
    
    def __init__(
        self, 
        name: str,
        description: str | None = None,
        value: Any | None = None,
        _type: Type | None = None,
        gt: annotated_types.Gt | None = None, 
        lt: annotated_types.Lt | None = None, 
        ge: annotated_types.Ge | None = None, 
        le: annotated_types.Le | None = None
    ):
        # if _type is None and value is not None:
        #     _type = type(value)
        # else:
        #     raise ValueError(f"Type is required when value is not provided")
        
        self.name = name
        self.type = _type
        self.description = description
        self.value = value
        self.gt = gt
        self.lt = lt
        self.ge = ge
        self.le = le
        
        
    def parse(self, content: str):
        content = content.strip()
        content = textwrap.dedent(content)
        if self.type == int:
            return int(content)
        elif self.type == float:
            return float(content)
        elif self.type == bool:
            return bool(content)
        elif self.type == str:
            return content
        elif self.type == list:
            return content.split(",")
        elif self.type == dict:
            return json.loads(content)
        else:
            raise ValueError(f"Invalid type: {self.type}")

    def model_dump(self):
        return {
            "name": self.name,
            "type": REVERSE_TYPE_REGISTRY.get(self.type, "str"),
            "description": self.description,
            "gt": self.gt,
            "lt": self.lt,
            "ge": self.ge,
            "le": self.le,
        }
        
    @classmethod
    def from_dict(cls, data: dict):
        data = data.copy()
        data["type"] = TYPE_REGISTRY.get(data["type"], str)
        return cls(**data)




class BlockSchema(Block):
    
    __slots__ = [
        "type",
        "name",
        "is_list",
        "is_list_item",
    ]
    
    def __init__(
        self, 
        name: str,
        type: Type | None = None,
        children: list["Block"] | None = None,
        attrs: dict[str, AttrBlock] | None = None,        
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
        id: str | None = None,
        parent: "BaseBlock | None" = None,  
        styles: list[str] | None = None,
        prefix: BaseContent | None = None,
        postfix: BaseContent | None = None,
        is_wrapper: bool = False,
    ):
        tags = tags or []        
        if name not in tags:
            tags.insert(0, name)
        styles = styles or parse_style(style)
        if not styles:
            styles = ["xml"]
        super().__init__(name if not is_wrapper else None, children=children, tags=tags, role=role or"view", styles=styles, parent=parent, attrs=attrs, prefix=prefix, postfix=postfix, is_wrapper=is_wrapper)
        # if not type:
            # raise ValueError("type is required")
        self.type = type
        self.name = name
        self.is_list = is_list_type(type)
        self.is_list_item = False
        
    @property
    def schema_path(self):
        path = []
        for b in self.traverse_path():
            idx = b.index_in_parent(BlockSchema)
            if idx > -1:
                path.insert(0, idx)
        return path
            # if type(b) is BlockSchema:
            #     if b.parent is None:
            #         path.append(0)
            #     else:
            
            
    def inst_dict(self, value: dict):
        from .block_builder import SchemaBuildContext
        ctx = SchemaBuildContext(self)
        ctx.inst_dict(value)
        return ctx.result
            
    def instantiate(
        self, 
        value = None, 
        content = None, 
        attrs: dict[str, str] | None = None,
        ignore_tags = False, 
        ignore_style = False, 
        is_wrapper: bool | None = None,        
    ) -> "Block":
        # if self.type and not type(value) is self.type:
            # raise ValueError(f'Error instantiating "{self.name}" block. Block type is "{self.type}" but supplied value is "{type(value)}" ')
        is_wrapper = is_wrapper if is_wrapper is not None else self.is_wrapper
        content = content or self.name if not is_wrapper else None
        blk = Block(
                content,
                # self.name,
                styles=self.styles if not ignore_style else None,
                tags=self.tags if not ignore_tags else None,
                # role=self.role,
                prefix=self.prefix,
                postfix=self.postfix,
                id=self.id,
                schema=self,
                is_wrapper=is_wrapper,
                # is_wrapper=is_wrapper,
            )
        if attrs:
            for k, v in attrs.items():
                attr_schema = self.attrs.get(k)
                if attr_schema is None:
                    raise ValueError(f"Attribute '{k}' not found in schema")
                parsed_value = attr_schema.parse(v)                
                blk.field(k, parsed_value, type=attr_schema.type)
                
        if value is not None:
            blk /= value
        return blk
    
    
    def parse(self, block: Block):        
        if self.type is not None:
            if issubclass(self.type, BaseModel):
                root_path = block.tag_path
                def set_dict_attr(target: dict, path: list[str], value: Any):
                    curr = target
                    for p in path[:-1]:
                        curr = curr[p]
                    curr[path[-1]] = value
                    return curr

                target = {}  
                for f in block.traverse():
                    if root_path == f.tag_path or f.schema is None:
                        continue
                    set_dict_attr(target, f.tag_path[len(root_path):], f.value)
                return self.type(**target)

            else:
                content = block.render_children()
                content = content.strip()
                return parse_content(content, self.type)
        raise ValueError(f"Invalid type: {self.type}")
        
    def copy(
        self,
        overrides: dict[str, Any] | None = None,
        copy_id: bool = False,
        copy_parent: bool = False,
    ):
        blk = BlockSchema(
            name=self.name if not overrides or "name" not in overrides else overrides["name"],
            type=self.type if not overrides or "type" not in overrides else overrides["type"],
            attrs=self.attrs if not overrides or "attrs" not in overrides else overrides["attrs"],
            role=self.role if not overrides or "role" not in overrides else overrides["role"],
            tags=self.tags if not overrides or "tags" not in overrides else overrides["tags"],
            styles=self.styles if not overrides or "styles" not in overrides else overrides["styles"],
            prefix=self.prefix if not overrides or "prefix" not in overrides else overrides["prefix"],
            postfix=self.postfix if not overrides or "postfix" not in overrides else overrides["postfix"],
            id=self.id if copy_id else None,
            parent=self.parent if copy_parent else None,
            is_wrapper=self.is_wrapper if not overrides or "is_wrapper" not in overrides else overrides["is_wrapper"],
        )
        blk.content = self.content.copy() if not overrides or "content" not in overrides else overrides["content"]
        blk.children = [c.copy() for c in self.children] if not overrides or "children" not in overrides else overrides["children"]
        
        
        return blk
    
    def copy_kind(self, kind: "Type[Block] | Type[BlockSchema]") -> "BlockSchema":
        res = _copy_kind_aux(self, kind)
        if type(res) is list:
            return BlockSchema(
                "wrapper",
                children=res,
                is_wrapper=True,
            )
        return res
    

    
    
    def traverse(self, target: Literal["schema", "block", "all"] = "all"):
        if target in ("schema", "all"):
            yield self
        for child in self.children:
            if isinstance(child, Block):
                yield from child.traverse(target)
            else:
                yield child
            
    


    
    # def instantiate(payload: dict):
    #     for 
        
        
    def repr_tree(self, verbose: bool = False):
        tags = ','.join(self.tags) if self.tags else ''
        tags = f"[{tags}] " if tags else ''
        space = " empty" if self.is_space() else ""
        role = f"role={self.role} " if self.role else ''
        res = f"{self.path}  BlockSchema({tags}id={self.id}, type={self.type}){space}"
        for child in self.children:
            res += f"\n{child.repr_tree(verbose=verbose)}"
        return res




        
    
    


PropNameType = Literal["content", "prefix", "postfix", "role", "tags", "styles", "attrs"]



MutatorType = TypeVar("MutatorType")

class ListMutator[MutatorType]:
    
    def __init__(self, lst: list[Block]):
        self.lst = lst
        
    def __itruediv__(self, value: MutatorType):
        self.append(value)
        return self
    
    def __iand__(self, value: MutatorType):
        raise NotImplementedError("Subclass must implement this method")
        
        
    def append(self, value: MutatorType):
        raise NotImplementedError("Subclass must implement this method")

    def prepend(self, value: MutatorType):
        raise NotImplementedError("Subclass must implement this method")

    def replace(self, index: int, value: MutatorType):
        raise NotImplementedError("Subclass must implement this method")

    # def __getitem__(self, index: int) -> Block:
    #     return self.lst[index]

    # def __setitem__(self, index: int, value: Block):
    #     self.lst[index] = value
    
# class ListItemMutator(ListMutator):


class ContentPrefixMutator(ListMutator[str]):
    def append(self, value: str):
        for blk in self.lst:
            blk.content.prefix += value
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.content.prefix = value + blk.content.prefix

    def replace(self, index: int, value: str):
        self.lst[index].content.prefix = value
        return self


class ContentPostfixMutator(ListMutator[str]):
    def append(self, value: str):
        for blk in self.lst:
            blk.content.postfix += value
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.content.postfix = value + blk.content.postfix
        return self

    def replace(self, index: int, value: str):
        self.lst[index].content.postfix = value
        return self
    
    
class BlockListPrefixMutator(ListMutator[str]):
    
    def append(self, value: str):
        for blk in self.lst:
            blk.prefix += value
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.prefix = value + blk.prefix

    def replace(self, index: int, value: str):
        self.lst[index].prefix = value
        return self


class BlockListPostfixMutator(ListMutator[str]):
    def append(self, value: str):
        for blk in self.lst:
            blk.postfix += value
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.postfix = value + blk.postfix

    def replace(self, index: int, value: str):
        self.lst[index].postfix = value
        return self


class SentMutator(ListMutator):
    
    def append(self, value: str):
        for blk in self.lst:
            blk.content.append(value)
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.content.prepend(value)
        return self
    

class BlockListContentMutator(ListMutator[str]):
    
    
    @property
    def prefix(self) -> ListMutator:
        return ContentPrefixMutator(self.lst)
    
    @prefix.setter
    def prefix(self, value: str):
        pass
    
    @property
    def postfix(self) -> ListMutator:
        return ContentPostfixMutator(self.lst)
    
    @postfix.setter
    def postfix(self, value: str):
        pass
    
    def append(self, value: str):
        for blk in self.lst:
            blk.content.append(value)
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.content.prepend(value)
        return self

    def replace(self, index: int, value: str):
        self.lst[index] = value
        return self
    
    

class BlockList(list[Block]):
    
    
    @property
    def content(self) -> BlockListContentMutator:
        return BlockListContentMutator(self)
    
    
    @content.setter
    def content(self, value: str):
        pass
    
    @property
    def prefix(self) -> BlockListPrefixMutator:
        return BlockListPrefixMutator(self)
    
    @prefix.setter
    def prefix(self, value: str):
        pass
    
    @property
    def postfix(self) -> BlockListPostfixMutator:
        return BlockListPostfixMutator(self)
    
    @postfix.setter
    def postfix(self, value: str):
        pass
    
    def replace_all(self, prop: PropNameType, value: Any):
        for blk in self:
            setattr(blk, prop, value)
        return self
            
            
    def prefix_append(self, value: str):
        for blk in self:
            blk.prefix += value
        return self
            
    def postfix_append(self, value: str):
        for blk in self:
            blk.postfix += value
        return self        
    
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    
    
    
    
    
    
    
    
    
    
class BlockListSchema(BlockSchema):
    
    __slots__ = [
        "list_schemas",
        "list_models",
        "key",
    ]
    
    
    def __init__(
        self, 
        name: str,
        type: Type | None = None,
        attrs: dict[str, AttrBlock] | None = None,        
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
        id: str | None = None,
        parent: "BaseBlock | None" = None,  
        styles: list[str] | None = None,
        prefix: BaseContent | None = None,
        postfix: BaseContent | None = None,
        is_wrapper: bool = True,
    ):
        super().__init__(name, tags=tags, role=role or"view", style=style, parent=parent, attrs=attrs, styles=styles, prefix=prefix, postfix=postfix, is_wrapper=is_wrapper)
        self.list_schemas = []
        self.list_models = {}
        self.key = None
        
        
    def key_field(self, name: str, type: Type | None = None, description: str | None = None):
        super().field(name, type=type, description=description)
        self.key = name
    
    
    
    
    
    def register(self, target: BlockSchema | Type[BaseModel]):
        # if isinstance(target, BaseModel):
            # block = pydantic_object_description(target)
        if isinstance(target, type) and issubclass(target, BaseModel):
            if self.key is None:
                raise ValueError("key_field is required")
            block = pydantic_class_description(self.name, target, self.key, self.attrs[self.key].type, class_model=target)
            self.list_models[target.__name__] = target
        elif isinstance(target, BlockSchema):
            block = target
        else:
            raise ValueError(f"Invalid target type: {type(target)}")
        self.list_schemas.append(block)
        self.append(block)
        return block
    
    
    
    
    def instantiate_item(
        self, 
        value = None, 
        content = None, 
        attrs: dict[str, str] | None = None,
        ignore_tags = False, 
        ignore_style = False, 
    ):
        if isinstance(value, BaseModel):
            if not value.__class__.__name__ in self.list_models:
                raise ValueError(f"Model {value.__class__.__name__} is not registered")
            blk = pydantic_object_description(self.name, value)
            return blk
        else:
            blk = super().instantiate(
                value, 
                content, 
                attrs=attrs, 
                ignore_tags=ignore_tags, 
                ignore_style=ignore_style,
                is_wrapper=False,
            )
            return blk

    def copy(
        self,
        overrides: dict[str, Any] | None = None,
        copy_id: bool = False,
        copy_parent: bool = False,
    ):
        blk = BlockListSchema(
            name=self.name if not overrides or "name" not in overrides else overrides["name"],
            type=self.type if not overrides or "type" not in overrides else overrides["type"],
            attrs=self.attrs if not overrides or "attrs" not in overrides else overrides["attrs"],
            role=self.role if not overrides or "role" not in overrides else overrides["role"],
            tags=self.tags if not overrides or "tags" not in overrides else overrides["tags"],
            styles=self.styles if not overrides or "styles" not in overrides else overrides["styles"],
            prefix=self.prefix if not overrides or "prefix" not in overrides else overrides["prefix"],
            postfix=self.postfix if not overrides or "postfix" not in overrides else overrides["postfix"],
            id=self.id if copy_id else None,
            parent=self.parent if copy_parent else None,
            is_wrapper=self.is_wrapper if not overrides or "is_wrapper" not in overrides else overrides["is_wrapper"],
        )
        blk.key = self.key if not overrides or "key" not in overrides else overrides["key"]
        blk.content = self.content.copy() if not overrides or "content" not in overrides else overrides["content"]
        blk.children = [c.copy() for c in self.children] if not overrides or "children" not in overrides else overrides["children"]
        blk.list_schemas = [c.copy() for c in self.list_schemas] if not overrides or "list_schemas" not in overrides else overrides["list_schemas"]
        blk.list_models = self.list_models if not overrides or "list_models" not in overrides else overrides["list_models"]
        
        
        return blk
