



from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Turn, Artifact



class ArtifactLog:
    
        
    @classmethod
    async def populate_turns(cls, turns: List["Turn"]):
        from collections import defaultdict
        from ..namespace_manager2 import NamespaceManager
        def kind2table(k: str):
            if k == "parameter":
                return "parameters"
            return k

        models_to_load = defaultdict(list)

        for turn in turns:
            for span in turn.spans:
                print(span.id, span.name)
                for value in span.values:
                    if value.kind != "span":
                        print(value.path, value.kind, value.artifact_id)
                        models_to_load[value.kind].append(value.artifact_id)
                    
        model_lookup = {"span": {s.artifact_id: s for turn in turns for s in turn.spans}}
        for k in models_to_load:
            if k == "list":
                models = await Artifact.query(include_branch_turn=True).where(Artifact.id.isin(models_to_load[k]))
                model_lookup["list"] = {m.id: m for m in models}
            elif k == "block_trees":
                models = await get_blocks(models_to_load[k], dump_models=False, include_branch_turn=True)
                model_lookup[k] = models
            # elif k == "execution_spans":
            #     value_dict[k] = {s.artifact_id: s for s in spans}
            else:
                ns = NamespaceManager.get_namespace(kind2table(k))
                models = await ns._model_cls.query(include_branch_turn=True).where(ns._model_cls.artifact_id.isin(models_to_load[k]))
                model_lookup[k] = {m.artifact_id: m for m in models}

        for turn in turns:
            for span in turn.spans:
                for value in span.values:
                    value._value = model_lookup[value.kind][value.artifact_id]
                    
        return turns
