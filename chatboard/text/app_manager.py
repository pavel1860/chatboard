import asyncio
from typing import Type, get_args

from chatboard.text.llms.completion_parsing import (is_list_model,
                                                    unpack_list_model)
from chatboard.text.llms.model import iterate_class_fields, serialize_class
from chatboard.text.llms.prompt import Prompt
from chatboard.text.vectors.rag_documents2 import RagDocuments
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel


def get_prompt_output_class(prompt_cls):
    return get_args(prompt_cls.__fields__['output_class'].annotation)[0]


def extract_json_schema(cls_):
    if hasattr(cls_, 'model_json_schema'):
        return cls_.model_json_schema()
    return convert_to_openai_tool(cls_)




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

    # response = [{
    #     "field": field,
    #     "type": PYTHON_TO_JSON_TYPES.get(info.annotation.__name__, info.annotation.__name__),
    # } for field, info in iterate_class_fields(profile_cls, sub_cls_filter, exclude=exclude)]
    response = {
        field : PYTHON_TO_JSON_TYPES.get(info.annotation.__name__, info.annotation.__name__)
    for field, info in iterate_class_fields(profile_cls, sub_cls_filter, exclude=exclude)}

    return response


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]



class AppManager(metaclass=SingletonMeta):

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
        prompt_name = prompt.__fields__.get('name')
        if hasattr(prompt_name, 'default'):
            self.prompts[prompt_name.default] = prompt
        else:
            self.prompts[prompt.__name__] = prompt
        


    def register_profile(self, profile):
        self.profiles[profile.__name__] = profile


    def get_metadata(self):
        rag_space_json = [{
            "namespace": namespace,
            # "metadata_class": extract_json_schema(rag_space["metadata_class"]),
            "metadata_class": serialize_class(rag_space["metadata_class"]),
            "prompt_name": rag_space["prompt"].__fields__.get('name').default,
            "prompt_rag": rag_space['prompt'].__fields__.get('rag_namespace').default
        } for namespace, rag_space in self.rag_spaces.items()]

        asset_json = [{
            "name": asset_name,
            "asset_class": serialize_asset(asset_cls)
        } for asset_name, asset_cls in self.assets.items()]

        profile_json = [{
            "name": profile_name,
            "profile_fields": serialize_profile(profile_cls)
        } for profile_name, profile_cls in self.profiles.items()]


        prompt_json = [{
            "name": prompt_name,
            "output_class": serialize_class(get_prompt_output_class(prompt_cls)),
            "namespace": prompt_cls.__fields__.get("rag_namespace", None).default
        } for prompt_name, prompt_cls in self.prompts.items()]
        

        return {
            "rag_spaces": rag_space_json,
            "assets": asset_json,
            "profiles": profile_json,
            "prompts": prompt_json
        }
    
    # def get_rag_manager(self, namespace: str):
    #     rag_cls = self.rag_spaces[namespace]["metadata_class"]
    #     ns = self.rag_spaces[namespace]["namespace"]
    #     return RagDocuments(ns, rag_cls)

app_manager = AppManager()