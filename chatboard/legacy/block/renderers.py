


import textwrap
from .block7 import Block, BlockList, FieldAttrBlock
from .renderers_base import BlockRenderer, ListRenderer, RenderContext


class MarkdownHeaderRenderer(BlockRenderer):
    styles = ["markdown", "md"]
    def render(self, ctx: RenderContext, block: Block, title_content: str, inner_content: str) -> str:
        level = min(len(block.path), 6)        
        header_content = f"{'#' * level} {title_content}"
        return f"{header_content}\n{inner_content}"


class NumberedListRenderer(ListRenderer):
    styles = ["list-num", "li"]
    
    
    def render(self, ctx: RenderContext, block: BlockList, inner_content: list[str]) -> str:
        path = ".".join([str(i) for i in block.path[1:]])
        if path:
            path += "."
        return "\n".join([f"{path}{i + 1}. {c}" for i, c in enumerate(inner_content)])
    
    # def _get_prefix(self, ctx: RenderContext) -> str:
    #     idx_list = []
    #     curr = ctx.parent_ctx
    #     while curr:
    #         if curr.is_context and curr.get_style("list-format") == "numbered-list" and not curr.is_root and curr.parent_ctx.get_style("list-format") == "numbered-list":
    #             idx_list.append(curr.index)
    #         curr = curr.parent_ctx
            
    #     if not idx_list:
    #         return ""
    #     return ".".join([str(i + 1) for i in reversed(idx_list)]) + "."




class AsteriskListRenderer(ListRenderer):
    styles = ["list-asterisk", "li-a"]
    
    def render(self, ctx: RenderContext, block: BlockList, inner_content: list[str]) -> str:
        return "\n".join([f"* {c}" for c in inner_content])
    
    

class DashListRenderer(ListRenderer):
    styles = ["list-dash", "li-d"]
    
    def render(self, ctx: RenderContext, block: BlockList, inner_content: list[str]) -> str:
        return "\n".join([f"- {c}" for c in inner_content])
    
    

class PlusListRenderer(ListRenderer):
    styles = ["list-plus", "li-p"]
    
    def render(self, ctx: RenderContext, block: BlockList, inner_content: list[str]) -> str:
        return "\n".join([f"+ {c}" for c in inner_content])
    
    

class BulletListRenderer(ListRenderer):
    styles = ["list-bullet", "li-b"]
    
    def render(self, ctx: RenderContext, block: BlockList, inner_content: list[str]) -> str:
        return "\n".join([f"• {c}" for c in inner_content])
    
    

class CheckboxListRenderer(ListRenderer):
    styles = ["list-checkbox", "li-c"]
    
    def render(self, ctx: RenderContext, block: BlockList, inner_content: list[str]) -> str:
        return "\n".join([f"[ ] {c}" for c in inner_content])



class XmlRenderer(BlockRenderer):
    styles = ["xml"]
    
    def render(self, ctx: RenderContext, block: Block, title_content: str, inner_content: str, postfix_content: str | None = None) -> str:  
        if block.attrs:
            attrs = ""
            for k, v in block.attrs.items():
                if isinstance(v, FieldAttrBlock):
                    instructions = ""
                    if v.type in (int, float):
                        instructions = " wrap in quotes"
                    attr_info = f"(\"{v.type.__name__}\"{instructions}) {v.description}"
                    if v.gt is not None:
                        attr_info += f" gt={v.gt}"
                    if v.lt is not None:
                        attr_info += f" lt={v.lt}"
                    if v.ge is not None:
                        attr_info += f" ge={v.ge}"
                    if v.le is not None:
                        attr_info += f" le={v.le}"
                    attrs += f"{k}=[{attr_info}] "
                else:
                    attrs += f"{k}=\"{v}\""
        else:
            attrs = ""
    
        if not inner_content:
            return f"<{title_content}{attrs} />"
        
        postfix_content = postfix_content or f"</{title_content}>"
        inner_content = textwrap.indent(inner_content, " ")
        return f"<{title_content}{attrs}>\n{inner_content}\n{postfix_content}"
        
            


# class ListColumnTupleLayoutRenderer(BaseRenderer):
    
#     def render_list_layout(self, ctx: RenderContext, content: list[str]) -> str:
#         return ",\n".join(content)

# class ListStreamLayoutRenderer(BaseRenderer):
    
#     def render_list_layout(self, ctx: RenderContext, content: list[str]) -> str:
#         return "".join(content)
    
    
    
#     # def render(self, block, style: dict, depth: int) -> str:
#     #     def indent(text: str) -> str:
#     #         return "\n".join([f"{' ' * depth}{line}" for line in text.split("\n")])
#     #     return indent(block.sep.join([indent(str(c)) for c in block.content]))
    
        
        
# class MarkdownHeaderRenderer(BaseRenderer):
    
#     def render(self, ctx: RenderContext, content: ContentType, inner_content: ContentType | None = None) -> str:
#         level = min(ctx.depth + 1, 6)        
#         header_content = f"{'#' * level} {content}"
#         if inner_content is not None and isinstance(inner_content, str):
#             return f"{header_content}\n{inner_content}"
#         else:
#             return header_content
    
    
# class NumberedListRenderer(BaseRenderer):
    
#     # def _get_prefix(self, ctx: RenderContext) -> str:
#     #     idx_list = []
#     #     curr = ctx.parent_ctx
#     #     while curr:
#     #         if curr.parent_ctx and curr.parent_ctx.block and curr.style.get("list-format") == "numbered-list":                
#     #             if not curr.is_list:
#     #                 idx_list.append(curr.index)
#     #             curr = curr.parent_ctx
#     #         else:
#     #             break
#     #     if not idx_list:
#     #         return ""
#     #     return ".".join([str(i + 1) for i in reversed(idx_list)]) + "."
    
#     def _get_prefix(self, ctx: RenderContext) -> str:
#         idx_list = []
#         curr = ctx.parent_ctx
#         while curr:
#             if curr.is_context and curr.get_style("list-format") == "numbered-list" and not curr.is_root and curr.parent_ctx.get_style("list-format") == "numbered-list":
#                 idx_list.append(curr.index)
#             curr = curr.parent_ctx
            
#         if not idx_list:
#             return ""
#         return ".".join([str(i + 1) for i in reversed(idx_list)]) + "."

    
#     def render(self, ctx: RenderContext, content: ContentType, inner_content: str | None = None) -> str:
#         return f"{ctx.index + 1}. {content}"
    
#     def render_list(self, ctx: RenderContext, content: list[str]) -> list[str]:
#         prefix = self._get_prefix(ctx)
#         postfix = "." if not prefix else ""
#         return [f"{prefix}{i + 1}{postfix} {c}" for i, c in enumerate(content)]
    
    
# class AsteriskListRenderer(BaseRenderer):
    
#     def render(self, ctx: RenderContext, content: ContentType, inner_content: str | None = None) -> str:
#         return f"* {content}"
    
#     def render_list(self, ctx: RenderContext, content: list[str]) -> list[str]:
#         return [f"* {c}" for c in content]
    

# class DashListRenderer(BaseRenderer):
    
#     def render(self, ctx: RenderContext, content: ContentType, inner_content: str | None = None) -> str:
#         return f"- {content}"
    
#     def render_list(self, ctx: RenderContext, content: list[str]) -> list[str]:
#         return [f"- {c}" for c in content]
    
    
# class PlusListRenderer(BaseRenderer):
    
#     def render(self, ctx: RenderContext, content: ContentType, inner_content: str | None = None) -> str:
#         return f"+ {content}"
    
#     def render_list(self, ctx: RenderContext, content: list[str]) -> list[str]:
#         return [f"+ {c}" for c in content]
    
# class BulletListRenderer(BaseRenderer):
    
#     def render(self, ctx: RenderContext, content: ContentType, inner_content: str | None = None) -> str:
#         return f"• {content}"
    
#     def render_list(self, ctx: RenderContext, content: list[str]) -> list[str]:
#         return [f"• {c}" for c in content]


# class CheckboxListRenderer(BaseRenderer):
    
#     def render(self, ctx: RenderContext, content: ContentType, inner_content: str | None = None) -> str:
#         return f"[ ] {content}"
    
#     def render_list(self, ctx: RenderContext, content: list[str]) -> list[str]:
#         return [f"[ ] {c}" for c in content]

  
# class XmlTitleRenderer(BaseRenderer):
    
#     def render(self, ctx: RenderContext, content: ContentType, inner_content: str | None = None) -> str:
        
#         if ctx.block.attrs:
#             attrs = ""
#             for k, v in ctx.block.attrs.items():
#                 if isinstance(v, FieldAttrBlock):
#                     instructions = ""
#                     if v.type in (int, float):
#                         instructions = " wrap in quotes"
#                     attr_info = f"(\"{v.type.__name__}\"{instructions}) {v.description}"
#                     # attrs += f"{k}=({v.type.__name__}) \"{v.description}\""
#                     if v.gt is not None:
#                         attr_info += f" gt={v.gt}"
#                     if v.lt is not None:
#                         attr_info += f" lt={v.lt}"
#                     if v.ge is not None:
#                         attr_info += f" ge={v.ge}"
#                     if v.le is not None:
#                         attr_info += f" le={v.le}"
#                     attrs += f"{k}=[{attr_info}] "
#                 else:
#                     attrs += f"{k}=\"{v}\""
#             # attrs = " " + " ".join([f"{k}=\"{v}\"" for k, v in ctx.block.attrs.items()])
#         else:
#             attrs = ""
        
#         if inner_content is not None:
#             inner_content = textwrap.indent(inner_content, " ")
#             return f"<{content}{attrs}>\n{inner_content}\n</{content}>"
#         else:
#             return f"<{content}{attrs} />"
    
    
    
    
    
    
