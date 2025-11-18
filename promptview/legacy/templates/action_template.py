from enum import Enum
import inspect
from pydantic import BaseModel
from promptview.prompt.legacy.mvc import view
from typing import Optional, Union, get_type_hints, get_origin, get_args, Type, Literal
import datetime as dt
from promptview.utils.model_utils import describe_enum, get_complex_fields, stringify_field_info


class ActionAttributeError(Exception):
    pass


@view()
def action_field_view(field_name: str, field_info):
    try:                        
        # return f"{field_name} ({get_field_info_origin(field_info)}): {field_info.description}"
        if field_info.description is None:
            raise ActionAttributeError(f"Missing description for field: {field_name}")
        extra_format = ""
        if hasattr(field_info, "annotation"): 
            if field_info.annotation == dt.date:
                extra_format = "use YYYY-MM-DD format. "
            elif field_info.annotation == dt.datetime:
                extra_format = "use YYYY-MM-DD HH:MM:SS format. "
        return f"""{field_name} ({stringify_field_info(field_info, "|")}):{extra_format}{field_info.description}"""
    except AttributeError:
        raise ValueError(f"Invalid field_info: {field_info}")
    


@view(title="Parameters", bullet="bullet")
def action_param_view(action: BaseModel):    
    return [
        action_field_view(field_name, field_info)
        for field_name, field_info in
        action.model_fields.items()
    ]

@view(indent=1)
def action_sub_view(action: BaseModel):
    return (
        f"Purpose: " + action.__doc__,
        action_param_view(action)
    )
    
@view()
def action_view(action_name: str, action: BaseModel):
    try:
        return (
            action_name,
            action_sub_view(action)        
        )
    except ActionAttributeError as e:
        raise ActionAttributeError(f"Action render error in {action.__name__}\n" + e.args[0] ) from e
    

        

@view(
    title="You have the following actions available",
    bullet="number",
)
def system_action_view(actions):
    action_views = []
    for action_name, action in actions:
        action_views += [action_view(action_name, action)]
        if clx_fields:= get_complex_fields(action):
            for field_name, field_info in clx_fields.items():
                action_views += [action_view(field_name, field_info)]        
    return action_views
    