from __future__ import annotations
from dataclasses import dataclass
from typing import Union


# --------------------------------------
# Path utilities
# --------------------------------------

def compute_path(node: Block) -> list[int]:
    path = []
    while node.parent is not None:
        path.append(node.index)
        node = node.parent
    return list(reversed(path))



def parse_style(style: str | list[str] | None) -> list[str]:
    if isinstance(style, str):
        return list(style.split(" "))
    elif type(style) is list:
        return style
    else:
        return []


# --------------------------------------
# Metadata
# --------------------------------------

class BlockMetadata:
    __slots__ = ["tag", "role", "dtype", "labels", "styles", "attrs"]

    def __init__(
        self,
        role: str | None = None,
        dtype: type | None = None,
        labels: list[str] | None = None,
        style: str | None = None,
        styles: list[str] | None = None,
        attrs: dict[str, str] | None = None,
    ):
        self.role = role
        self.dtype = dtype
        self.labels = labels or []
        if styles is None and style is not None:
            styles = parse_style(style)
        self.styles = styles or []
        self.attrs = attrs or {}


# --------------------------------------
# Child wrapper
# --------------------------------------

@dataclass
class Child:
    content: Union[str, "Block"]
    sep: str = " "
    logprob: float | None = None


# --------------------------------------
# Block
# --------------------------------------

class Block:
    __slots__ = [
        "tag",
        "children",
        "parent",
        "index",
        "metadata",
        "default_sep",
        "has_eol",
    ]

    def __init__(
        self,
        content: Union[str, Block] | None = None,
        *,
        tag: str | None = None,
        parent: Block | None = None,
        index: int | None = None,
        sep: str = " ",
        logprob: float | None = None,
        default_sep: str = " ",        
        role: str | None = None,
        dtype: type | None = None,
        labels: list[str] | None = None,
        style: str | None = None,
        styles: list[str] | None = None,
        attrs: dict[str, str] | None = None,
        has_eol: bool | None = None,
    ):
        self.tag = tag
        self.children: list[Child] = []
        self.parent = parent
        self.index = index if index is not None else 0
        self.metadata: BlockMetadata | None = None
        self.default_sep = default_sep
        if has_eol is None:            
            self.has_eol = True if "\n" in sep else False
        else:
            self.has_eol = has_eol

        # init metadata if provided
        if any([tag, role, dtype, labels, styles, attrs, style]):
            self.metadata = BlockMetadata(
                role=role, dtype=dtype,
                labels=labels, style=style, styles=styles, attrs=attrs
            )

        # add initial content
        if content is not None:
            self.append(content, sep=sep, logprob=logprob)

    # --- metadata accessor ---
    @property
    def meta(self) -> BlockMetadata:
        if self.metadata is None:
            self.metadata = BlockMetadata()
        return self.metadata

    # --- convenience properties ---

    @property
    def role(self): return self.metadata.role if self.metadata else None
    @property
    def labels(self): return self.metadata.labels if self.metadata else []
    @property
    def styles(self): return self.metadata.styles if self.metadata else []
    @property
    def attrs(self): return self.metadata.attrs if self.metadata else {}
    @property
    def dtype(self): return self.metadata.dtype if self.metadata else None

    @property
    def path(self) -> list[int]:
        return compute_path(self)

    @property
    def logprob(self) -> float | None:
        values = [c.logprob for c in self.children if c.logprob is not None]
        return sum(values) if values else None

    # --- child management ---
    def _connect(self, content: Union[str, Block], index: int) -> Union[str, Block]:
        if isinstance(content, Block):
            content.parent = self
            content.index = index
        return content

    def insert_child(
        self,
        index: int,
        content: Union[str, Block],
        *,
        sep: str | None = None,
        logprob: float | None = None,
    ) -> Child:
        sep = sep or self.default_sep
        self._shift_indices(index)
        content = self._connect(content, index)
        child = Child(content=content, sep=sep, logprob=logprob)
        self.children.insert(index, child)
        return child

    def append(
        self,
        content: Union[str, Block],
        *,
        sep: str | None = None,
        logprob: float | None = None,
        path: list[int] | None = None,
    ) -> Child:
        """
        Append a child.
        - If path is None, appends directly to self.
        - If path is provided, navigates to that node and appends there.
        """
        sep = sep or self.default_sep

        if path is None:
            return self.insert_child(len(self.children), content, sep=sep, logprob=logprob)

        # navigate
        target = self.get_path(path)
        if not isinstance(target, Block):
            # auto-promote string to Block if needed
            promoted = Block(target, parent=self, index=path[-1])
            parent = self.get_path(path[:-1])
            if not isinstance(parent, Block):
                raise ValueError(f"Cannot promote non-Block at {path}")
            parent.replace_child(path[-1], promoted, sep=sep)
            target = promoted

        return target.insert_child(len(target.children), content, sep=sep, logprob=logprob)


    def replace_child(
        self,
        index: int,
        content: Union[str, Block],
        *,
        sep: str | None = None,
        logprob: float | None = None,
    ) -> Child:
        sep = sep or self.default_sep
        content = self._connect(content, index)
        child = Child(content=content, sep=sep, logprob=logprob)
        self.children[index] = child
        return child

    def _shift_indices(self, start: int, shift: int = 1):
        for i in range(start, len(self.children)):
            if isinstance(self.children[i].content, Block):
                self.children[i].content.index += shift

    # --- path operations ---
    def get_path(self, idx_path: list[int]) -> Union[str, Block]:
        target: Union[Block, str] = self
        for idx in idx_path:
            if not isinstance(target, Block):
                raise ValueError(f"Invalid path {idx_path}: reached non-Block node.")
            target = target.children[idx].content
        return target

    def insert_by_path(
        self, idx_path: list[int], content: Union[str, Block],
        *, sep: str | None = None, logprob: float | None = None
    ):
        if not idx_path:
            raise ValueError("Path must not be empty")
        parent = self.get_path(idx_path[:-1])
        if not isinstance(parent, Block):
            raise ValueError("Cannot insert into a string node")
        return parent.insert_child(idx_path[-1], content, sep=sep, logprob=logprob)
    
    
    def check_invariants(self, recursive: bool = True):
        """
        Sanity checks to ensure the Block tree is well-formed.
        Raises AssertionError if something is inconsistent.
        """
        # 1. Children must be Child objects
        for i, child in enumerate(self.children):
            assert isinstance(child, Child), f"Child at {i} is not a Child: {child!r}"

            # 2. Separator sanity
            assert isinstance(child.sep, str), f"Invalid separator type at {i}"

            # 3. Logprob sanity
            assert child.logprob is None or isinstance(child.logprob, (float, int)), \
                f"Invalid logprob at {i}: {child.logprob}"

            # 4. If child is a Block, parent/index must match
            if isinstance(child.content, Block):
                c = child.content
                assert c.parent is self, f"Child {c} has wrong parent"
                assert c.index == i, f"Child {c} index mismatch (expected {i}, got {c.index})"
                if recursive:
                    c.check_invariants(recursive=True)

        # 5. Metadata sanity
        if self.metadata is not None:
            assert isinstance(self.metadata, BlockMetadata)
            if self.metadata.labels is not None:
                assert isinstance(self.metadata.labels, list)
            if self.metadata.styles is not None:
                assert isinstance(self.metadata.styles, list)
            if self.metadata.attrs is not None:
                assert isinstance(self.metadata.attrs, dict)

        # 6. Path sanity
        if self.parent:
            path = compute_path(self)
            # Walk down from root using path must yield back this node
            root = self
            while root.parent:
                root = root.parent
            found = root.get_path(path)
            assert found is self, f"Path check failed for {self} (got {found})"

        return True
    
    
    def print_tree(self, indent: int = 0):
        """
        Pretty-print the tree with one row per child (Block or Text).
        Shows metadata inline, indentation for hierarchy.
        """
        pad = "  " * indent

        # current block line
        meta = []
        tag = self.tag + " " if self.tag else ""
        if self.role:   meta.append(f"role={self.role}")
        if self.labels:   meta.append(f"labels={self.labels}")
        if self.dtype:  meta.append(f"dtype={self.dtype.__name__}")
        meta_str = " ".join(meta)
        print(f"{pad}Block({tag}idx={self.index}{' ' + meta_str if meta_str else ''})")

        # children
        for child in self.children:
            c = child.content
            if isinstance(c, Block):
                c.print_tree(indent + 1)
            else:
                info = []
                if child.logprob is not None:
                    info.append(f"logprob={child.logprob:.3f}")
                if child.sep and child.sep != " ":
                    info.append(f"sep={repr(child.sep)}")
                info_str = " " + " ".join(info) if info else ""
                print(f"{pad}  Text({repr(c)}){info_str}")

    # --- rendering ---
    def render(self) -> str:
        out = []
        for child in self.children:
            part = child.content.render() if isinstance(child.content, Block) else child.content
            out.append(part + child.sep)
        return "".join(out)
    
    
    # syntactic sugar
    def __enter__(self):
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def __call__(
        self, 
        tag: str,
        style: str | None = None,
        labels: list[str] | None = None,
        role: str | None = None,
        dtype: type | None = None,
        styles: list[str] | None = None,
        attrs: dict[str, str] | None = None,
    ) -> Block:
        tag_block = Block(
            tag=tag, 
            labels=labels, 
            role=role, 
            dtype=dtype, 
            styles=styles, 
            attrs=attrs,
            sep="\n",
        )
        self.append(tag_block)
        return tag_block
    
    def _iappend_tuple(self, other: tuple[str | Block, ...], sep: str = " "):
        block = Block(default_sep=" ")
        for o in other:
            block.append(o, sep=sep)
        self.append(block)
    
    def __itruediv__(self, other: Union[str, Block, tuple[str | Block, ...]]):        
        if isinstance(other, tuple):
            self._iappend_tuple(other, sep=" ")
        else:
            self.append(other, sep="\n")
        return self
    
    def __iadd__(self, other: Union[str, Block]):
        if isinstance(other, tuple):
            self._iappend_tuple(other, sep=" ")
        else:
            self.append(other, sep=" ")
        return self
    
    def __iand__(self, other: Union[str, Block]):
        if isinstance(other, tuple):
            self._iappend_tuple(other, sep="")
        else:
            self.append(other, sep="")
        return self

    def __repr__(self):
        return f"Block(tag={self.tag}, role={self.role}, idx={self.index}, children={len(self.children)})"


class ContextStack:
    """
    A stack-based context for managing nested block structures.
    """
    _ctx_stack: "list[Block]"
    
    def __init__(self):
        self._ctx_stack = []
        
    def __getitem__(self, idx: int) -> "Block":
        return self._ctx_stack[idx]
    
    def __len__(self) -> int:
        return len(self._ctx_stack)
    
    def root(self) -> "Block":
        if not self._ctx_stack:
            raise ValueError("No context stack")
        return self._ctx_stack[0]
    
    def push(self, block: "Block"):
        self._ctx_stack.append(block)
        
    def pop(self):
        return self._ctx_stack.pop()
    
    def top(self):
        return self._ctx_stack[-1]




class BlockContext:
    
    def __init__(
        self,
        content: Union[str, Block] | None = None,
        *,
        parent: Block | None = None,
        index: int | None = None,
        sep: str = " ",
        logprob: float | None = None,
        default_sep: str = " ",
        tag: str | None = None,
        role: str | None = None,
        dtype: type | None = None,
        labels: list[str] | None = None,
        styles: list[str] | None = None,
        attrs: dict[str, str] | None = None,
    ):
        self.root = Block(
            content=content,
            parent=parent,
            index=index,
            sep=sep,
            logprob=logprob,
            default_sep=default_sep,
            tag=tag,
            role=role,
            dtype=dtype,
            labels=labels,
            styles=styles,
            attrs=attrs,
        )
        self.ctx = ContextStack()
        self.ctx.push(self.root)
        
        
    def append(
        self,
        content: Union[str, Block],
        *,
        sep: str | None = None,
        logprob: float | None = None,
        path: list[int] | None = None,
    ) -> Child:
        return self.ctx.top.append(content, sep=sep, logprob=logprob, path=path)
        
        
    def __call__(
        self, 
        title: Union[str, Block],
        labels: list[str] | None = None,
        role: str | None = None,
        dtype: type | None = None,
        styles: list[str] | None = None,
        attrs: dict[str, str] | None = None,
    ) -> BlockContext:
        title_block = Block(
            title, 
            labels=labels, 
            role=role, 
            dtype=dtype, 
            styles=styles, 
            attrs=attrs,
            sep="\n",
        )
        self.ctx.push(title_block)
        return self
    
    
    # def __enter__(self):




def block(
    content: Union[str, Block] | None = None,
    *,
    parent: Block | None = None,
    index: int | None = None,
    sep: str = " ",
    logprob: float | None = None,
    default_sep: str = " ",
    tag: str | None = None,
    role: str | None = None,
    dtype: type | None = None,
    labels: list[str] | None = None,
    styles: list[str] | None = None,
    attrs: dict[str, str] | None = None,
) -> BlockContext:
    return BlockContext(
        content=content,
        parent=parent,
        index=index,
        sep=sep,
        logprob=logprob,
        default_sep=default_sep,
        tag=tag,
        role=role,
        dtype=dtype,
        labels=labels,
        styles=styles,
        attrs=attrs,
    )