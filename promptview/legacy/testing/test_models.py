from typing import TYPE_CHECKING, List, Literal
from uuid import UUID
from pydantic import BaseModel, Field
from ..model import NamespaceManager

from datetime import datetime

# # if TYPE_CHECKING:
from ..model import Model, ModelField
from ..model import Branch, Turn, KeyField, RelationField




class EvaluatorConfig(BaseModel):
    name: str = Field(default="")
    metadata: dict = Field(default={})


class TestTurn(Model):
    id: int = KeyField(primary_key=True)
    created_at: datetime = ModelField(default_factory=datetime.now, order_by=True)
    updated_at: datetime = ModelField(default_factory=datetime.now)
    test_case_id: int = ModelField(default=None, foreign_key=True)
    turn_id: int = ModelField(default=None, foreign_key=True, foreign_cls=Turn)
    evaluators: List[EvaluatorConfig] = ModelField(default=[])
    turn: Turn = RelationField(
        primary_key="turn_id", 
        foreign_key="id", 
    )


class Evaluation(Model):
    id: int = KeyField(primary_key=True)
    created_at: datetime = ModelField(default_factory=datetime.now, order_by=True)
    updated_at: datetime = ModelField(default_factory=datetime.now)
    evaluator: str = ModelField(default="", description="the evaluator that was used to evaluate the turn")    
    metadata: dict = ModelField(default={}, description="the metadata of the evaluation")
    score: float | None = ModelField(default=None, description="the score of the test run")
    turn_eval_id: int = ModelField(default=None, foreign_key=True)



class TurnEval(Model):
    id: int = KeyField(primary_key=True)
    created_at: datetime = ModelField(default_factory=datetime.now, order_by=True)
    updated_at: datetime = ModelField(default_factory=datetime.now)
    test_turn_id: int = ModelField(default=None)
    ref_turn_id: int = ModelField(default=None)
    test_run_id: int = ModelField(default=None, foreign_key=True)
    score: float | None = ModelField(default=None, description="the score of the test run")
    evaluations: List[Evaluation] = RelationField(foreign_key="turn_eval_id")
    trace_id: str = ModelField(default="", description="the id of the test run")


    
    
class TestRun(Model):
    id: int = KeyField(primary_key=True)
    created_at: datetime = ModelField(default_factory=datetime.now, order_by=True)
    updated_at: datetime = ModelField(default_factory=datetime.now)
    test_case_id: int = ModelField(default=None, foreign_key=True)
    branch_id: int = ModelField(default=1, description="the branch this test run belongs to")
    score: float | None = ModelField(default=None, description="the score of the test run")
    status: Literal["running", "success", "failure"] = ModelField(default="running", description="the status of the test run")
    turn_evals: List[TurnEval] = RelationField(foreign_key="test_run_id")
    
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):        
        if exc_type is not None:
            self.status = "failure"
        else:
            self.status = "success"
        await self.save()
        
    

# class EvaluatorConfig(BaseModel):
#     name: str = Field(default="")
#     metadata: dict = Field(default={})
    
    
class TurnEvaluator(BaseModel):
    turn_id: int = Field()
    evaluators: List[EvaluatorConfig] = Field(default=[])
    
    


class TestCaseEvaluators(BaseModel):
    evaluators: dict[int, TurnEvaluator] = Field(default={})
    
    
    def add(self, turn: "Turn | int", evaluators: List[EvaluatorConfig]):
        from ..model import Turn
        if isinstance(turn, Turn):
            self.evaluators[turn.id] = TurnEvaluator(turn_id=turn.id, evaluators=evaluators)
        else:
            self.evaluators[turn] = TurnEvaluator(turn_id=turn, evaluators=evaluators)
            
    def get(self, turn: "Turn | int") -> TurnEvaluator:
        from ..model import Turn
        if isinstance(turn, Turn):
            return self.evaluators[turn.id]
        else:
            return self.evaluators[turn]

# class InputTurn(BaseModel):
#     input: str = ModelField(default="")
#     expected: List[str] = ModelField(default=[])
#     evaluators: List[EvaluatorConfig] = ModelField(default=[])
    
    
# class PromptEval(BaseModel):
#     description: str = ModelField(default="")
    

# class TestTurn(Model):
#     id: int = KeyField(primary_key=True)
#     created_at: datetime = ModelField(default_factory=datetime.now, is_default_temporal=True)
#     updated_at: datetime = ModelField(default_factory=datetime.now)
#     test_case_id: int = ModelField(default=None, foreign_key=True)
#     turn_id: int = ModelField(default=None, foreign_key=True)
#     evaluators: List[EvaluatorConfig] = ModelField(default=[])
#     turns: Relation["Turn"] = RelationField(
#         primary_key="turn_id", 
#         foreign_key="id", 
#         create_foreign_key=False
#     )




    


class TestCase(Model):
    id: int = KeyField(primary_key=True)
    created_at: datetime = ModelField(default_factory=datetime.now, order_by=True)
    updated_at: datetime = ModelField(default_factory=datetime.now)
    title: str = ModelField(default="")
    description: str = ModelField(default="")
    # evaluators: TestCaseEvaluators = ModelField(default_factory=TestCaseEvaluators)
    branch_id: int = ModelField(default=1, foreign_key=True, foreign_cls=Branch, description="the branch this test case belongs to")
    # start_turn_id: int = ModelField(foreign_key=True, description="the turn this test case will start from")
    # end_turn_id: int = ModelField(foreign_key=True, description="the turn this test case will end at")
    # limit: int = ModelField(description="the number of turns to test")
    test_turns: List[TestTurn] = RelationField(foreign_key="test_case_id")
    user_id: UUID = ModelField(description="the user this test case belongs to")
    test_runs: List[TestRun] = RelationField(foreign_key="test_case_id")
