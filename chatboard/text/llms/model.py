from typing import Optional
from pydantic import BaseModel, create_model
import inspect


class Config:
    arbitrary_types_allowed = True


def make_optional(model: BaseModel) -> BaseModel:
    optional_fields = {k: (Optional[v], None) for k, v in model.__annotations__.items()}
    return create_model(
        model.__name__, 
        **optional_fields, 
        __config__=Config
    )


def get_model_fields(model_instance, model_class):
    fields = {field: getattr(model_instance, field) for field in model_class.__fields__.keys()}
    return model_class(**fields)



# def iterate_class_fields(cls_, sub_cls_filter=None, exclude=False):
#     for field, info in cls_.__fields__.items():
#         if sub_cls_filter is not None and (not inspect.isclass(info.annotation) or not issubclass(info.annotation, sub_cls_filter)):
#             continue
#         yield field, info


# def iterate_class_fields(cls_, sub_cls_filter=None, exclude=False):
#     for field, info in cls_.__fields__.items():
#         if sub_cls_filter is not None:
#             if not exclude and (inspect.isclass(info.annotation) and issubclass(info.annotation, sub_cls_filter)):
#                 yield field, info
#             if exclude and (not inspect.isclass(info.annotation) or not issubclass(info.annotation, sub_cls_filter)):
#                 yield field, info                
#             continue
#         yield field, info



def iterate_class_fields(cls_, sub_cls_filter=None, exclude=False):

    def _issubclass(cls_, sub_cls_filter):
        if type(sub_cls_filter) == list:
            return any([issubclass(cls_, sub_cls) for sub_cls in sub_cls_filter])
        return issubclass(cls_, sub_cls_filter)

    for field, info in cls_.__fields__.items():
        if sub_cls_filter is not None:
            if not exclude and (inspect.isclass(info.annotation) and _issubclass(info.annotation, sub_cls_filter)):
                yield field, info
            if exclude and (not inspect.isclass(info.annotation) or not _issubclass(info.annotation, sub_cls_filter)):
                yield field, info                
            continue
        yield field, info



