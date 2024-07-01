from typing import Type
from chatboard.text.llms.model import iterate_class_fields
from chatboard.text.llms.prompt import Prompt
from langchain_core.utils.function_calling import convert_to_openai_tool
import asyncio




from pydantic import BaseModel

from chatboard.text.vectors.rag_documents2 import RagDocuments


def serialize_asset(asset_cls):
    asset = asset_cls()
    return {
        # "input_class": convert_to_openai_tool(asset.input_class).get('function', None),
        # "output_class": convert_to_openai_tool(asset.output_class).get('function', None),
        "input_class": convert_to_openai_tool(asset.input_class) if asset.input_class is not None else None,
        "output_class": convert_to_openai_tool(asset.output_class),
        "metadata_class": convert_to_openai_tool(asset.metadata_class),
    }
    

def serialize_profile(profile_cls, sub_cls_filter=None, exclude=False):
    
    PYTHON_TO_JSON_TYPES = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }

    response = [{
        "field": field,
        "type": PYTHON_TO_JSON_TYPES.get(info.annotation.__name__, info.annotation.__name__),
    } for field, info in iterate_class_fields(profile_cls, sub_cls_filter, exclude=exclude)]

    return response
    

class AppManager:

    def __init__(self):
        self.rag_spaces = {}
        self.assets = {}
        self.prompts = {}
        self.profiles = {}


    def register_rag_space(self, namespace: str, metadata_class: Type[BaseModel] | Type[str], prompt: Prompt | None = None):
        if namespace in self.rag_spaces:
            return
        self.rag_spaces[namespace] = {
            "metadata_class": metadata_class,
            "namespace": namespace,
            "prompt": prompt
        }


    def register_asset(self, asset_cls):
        self.assets[asset_cls.__name__] = asset_cls


    async def verify_rag_spaces(self):        
        rag_spaces_futures = [
            RagDocuments(
                namespace, 
                metadata_class=rag_space["metadata_class"]
            ).verify_namespace() for namespace, rag_space in self.rag_spaces.items()]
        rag_spaces_futures += [asset_cls().verify_namespace() for asset_cls in self.assets.values()]
        await asyncio.gather(*rag_spaces_futures)
        
            
    
    def register_prompt(self, prompt):
        self.prompts[prompt.name] = prompt

    def register_profile(self, profile):
        self.profiles[profile.__class__.__name__] = profile

    def get_metadata(self):
        rag_space_json = [{
            "namespace": namespace,
            "metadata_class": convert_to_openai_tool(rag_space["metadata_class"])
        } for namespace, rag_space in self.rag_spaces.items()]

        asset_json = [{
            "name": asset_name,
            "asset_class": serialize_asset(asset_cls)
        } for asset_name, asset_cls in self.assets.items()]

        profile_json = [{
            "name": profile_name,
            "profile_fields": serialize_profile(profile_cls)
        } for profile_name, profile_cls in self.profiles.items()]
        

        return {
            "rag_spaces": rag_space_json,            
            "assets": asset_json,
            "profiles": profile_json
        }
    
    # def get_rag_manager(self, namespace: str):
    #     rag_cls = self.rag_spaces[namespace]["metadata_class"]
    #     ns = self.rag_spaces[namespace]["namespace"]
    #     return RagDocuments(ns, rag_cls)

app_manager = AppManager()