from .models import (
    Branch, 
    Turn, 
    TurnStatus, 
    VersionedModel, 
    ArtifactModel, 
    Artifact, 
    Parameter, 
    SpanType,
    ArtifactKindEnum,
)

from .dataflow_models import (
    DataFlowNode,
    ExecutionSpan,
    DataArtifact,
    Log,
    ValueIOKind,
    LlmCall,
)

from .eval_models import (
    TestTurn,
    TestCase,
    TestRun,
    TurnEval,
    ValueEval,
    EvaluatorConfig,
    EvaluationFailure,
)
from .block_storage import BlockLog, BlockLogQuery, BlockModel as StoredBlockModel, compute_block_hash

# =============================================================================
# Resolve forward references
# =============================================================================
# Pydantic requires model_rebuild() after all forward-referenced models are imported.
# Must be done here in __init__.py after ALL model files are loaded.
Branch.model_rebuild()
Turn.model_rebuild()
Artifact.model_rebuild()
VersionedModel.model_rebuild()
ExecutionSpan.model_rebuild()
DataFlowNode.model_rebuild()
Log.model_rebuild()

__all__ = [
    "Branch",
    "Turn",
    "TurnStatus",
    "VersionedModel",
    "ArtifactModel",
    "ArtifactKindEnum",
    "StoredBlockModel",  # New simplified model
    "ExecutionSpan",
    "Log",
    "DataFlowNode",
    "Artifact",
    "Parameter",
    "DataArtifact",
    "TestTurn",
    "TestCase",
    "TestRun",
    "TurnEval",
    "ValueEval",
    "EvaluatorConfig",
    "EvaluationFailure",
    "BlockLog",
    "BlockLogQuery",
    "compute_block_hash",
    "SpanType",
    "ValueIOKind",
    "LlmCall",
]