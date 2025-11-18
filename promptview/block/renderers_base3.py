from typing import Any
from .block7 import BlockSequence, BlockChunk, Block, BlockSent, FieldAttrBlock
from .style2 import StyleManager



class RendererRegistry:
    
    
    def __init__(self):
        self._registry = {}
        self._default_registry = {}
    
    def register(self, name: str, renderer: "BaseRenderer"):
        self._registry[name] = renderer
        
    def register_default(self, name: str, renderer: "BaseRenderer"):
        self._default_registry[name] = renderer

    def get(self, name: str | None = None, default_name: str | None = None) -> "BaseRenderer":
        if name:
            renderer =  self._registry[name]
        else:
            renderer =  self._default_registry[default_name]
        return renderer()

    def list_renderers(self):
        return list(self._registry.keys());




style_manager = StyleManager()       
renderer_registry = RendererRegistry()


class MetaRenderer(type):
    
    def __new__(cls, name, bases, dct):
        cls_obj = super().__new__(cls, name, bases, dct)
        if name == "ChunkRenderer":
            renderer_registry.register_default("chunk-format", cls_obj)
        elif name == "SequenceRenderer":
            renderer_registry.register_default("sequence-format", cls_obj)
        elif name == "BlockRenderer":
            renderer_registry.register_default("block-format", cls_obj)
        elif bases:
            if "styles" not in dct or not dct["styles"]:
                raise ValueError(f"Renderer {name} must define styles")                
            if bases[0] == ChunkRenderer:
                renderer_registry.register(name, cls_obj)
                style_manager.add_style(dct["styles"], {"chunk-format": name})
            elif bases[0] == SequenceRenderer:
                renderer_registry.register(name, cls_obj)
                style_manager.add_style(dct["styles"], {"sequence-format": name})
            elif bases[0] == BlockRenderer:
                renderer_registry.register(name, cls_obj)
                style_manager.add_style(dct["styles"], {"block-format": name})
            else:
                raise ValueError(f"Renderer {name} must inherit from ChunkRenderer, SequenceRenderer, or BlockRenderer")            
        
        return cls_obj
    

            
            
class BaseRenderer(metaclass=MetaRenderer):
    styles = []
    
    
    
    
    def render_chunk(self, block: BlockChunk) -> str:
        # fmt = style_manager.resolve(block)
        renderer = renderer_registry.get(None, "chunk-format")
        return renderer.render(block)
    
    def render_sequence(self, block: BlockSequence) -> str:        
        fmt = style_manager.resolve(block)
        renderer = renderer_registry.get(fmt.get("sequence-format"), "sequence-format")
        return renderer.render(block)
    
    def render_block(self, block: Block) -> str:
        fmt = style_manager.resolve(block)
        renderer = renderer_registry.get(fmt.get("block-format"), "block-format")
        return renderer.render(block)
    
    
    def render(self, block: Any) -> str:
        """
        Render the current block itself.
        """
        if isinstance(block, BlockChunk):
            return self.render_chunk(block)
        elif isinstance(block, Block):
            return self.render_block(block)
        elif isinstance(block, BlockSequence):
            return self.render_sequence(block)
        else:
            raise ValueError(f"Invalid block type: {type(block)}")        

     


class ChunkRenderer(BaseRenderer):

    def render(self, block: BlockChunk) -> str:
        if isinstance(block.content, str):
            return block.content
        elif isinstance(block.content, int):
            return str(block.content)
        elif isinstance(block.content, float):
            return str(block.content)
        elif isinstance(block.content, bool):
            return str(block.content)
        else:
            raise ValueError(f"Invalid content type: {type(block.content)}")
        return block.content

    
    
class SequenceRenderer(BaseRenderer):

    def render(self, block: BlockSequence) -> str:
        content_list= []
        for sep, block in block.iter_chunks():
            content = BaseRenderer().render(block)  
            content_list += [sep, content]        
        return "".join(content_list)
    
    
class BlockRenderer(BaseRenderer):    
    
    def render(self, block: Block) -> str:        
      content = BaseRenderer().render_sequence(block.root)      
      if children_content := BaseRenderer().render_sequence(block):
        content += f"\n{children_content}"
      return content
      

# class MarkdownRenderer(SequenceRenderer):
#     styles = ["markdown", "md"]
    
#     def render(self, block: BlockSequence) -> str:
#         content = super().render(block)
#         return f"# {content}"


class MarkdownRenderer(BlockRenderer):
    styles = ["markdown", "md"]
    
    def render(self, block: Block) -> str:        
        content = BaseRenderer().render_sequence(block.root)   
        content = f"# {content}"    
        if children_content := BaseRenderer().render_sequence(block):
            content += f"\n{children_content}"
        return content


def render(block: Block) -> str:
    return BaseRenderer().render(block)




class NumeratedListRenderer(SequenceRenderer):
    styles = ["numerated-list", "num-list"]
    
    def render(self, block: BlockSequence) -> str:
        content_list= []
        for i, (sep, block) in enumerate(block.iter_chunks()):
            content = BaseRenderer().render(block)  
            content = f"{i+1}. {content}"
            content_list += [sep, content]
        return "".join(content_list)
    
    
    
    
class XMLRenderer(BlockRenderer):
    styles = ["xml"]
    
    
    def render_attrs(self, block: Block) -> str:
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
        return attrs
    
    def render(self, block: Block) -> str:
        head_content = BaseRenderer().render_sequence(block.root)
        attrs_content = " " + self.render_attrs(block) + "" if block.attrs else ""
        
        if children_content := BaseRenderer().render_sequence(block):
            content = f"<{head_content}{attrs_content}>"
            content += f"\n{children_content}"
            content += f"\n</{head_content}>"
        else: 
            content = f"<{head_content}{attrs_content}/>"  
        return content