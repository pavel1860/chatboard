# import enum
# from typing import List
# # from ..model.head_model import HeadModel
# # from ..model.model import Model, Relation
# # from ..model.fields import ModelField, ModelRelation
# from ..model2 import Model, Relation, ModelField, RelationField
# from pydantic import BaseModel, PrivateAttr

# from ..model2.fields import KeyField
# from ..testing.evaluator import evaluate_prompt




# class EvalResult(BaseModel):
#     reasoning: str = ModelField(default="")
#     score: float = ModelField(default=-1)


# class TurnResult(BaseModel):
#     output: str = ModelField(default="")
#     evaluations: list[EvalResult] = ModelField(default=[])
#     score: float = ModelField(default=-1)


# class TestRunStatus(enum.StrEnum):
#     STARTED = "started"
#     SUCCESS = "success"
#     FAILURE = "failure"



# class TestRun(Model):
#     id: int = KeyField(primary_key=True)
#     message: str = ModelField(default="")
#     results: list[TurnResult] = ModelField(default=[])
#     final_score: float = ModelField(default=-1)
#     status: TestRunStatus = ModelField(default=TestRunStatus.STARTED)
#     test_case_id: int = ModelField(default=None, foreign_key=True)
#     branch_id: int = ModelField()
        
#     def add_turn(self, output: str):
#         self.results.append(TurnResult(output=output))
        
#     def add_eval_result(self, reasoning: str, score: float):
#         if len(self.results) == 0:
#             raise ValueError("No turn results to add to")
#         self.results[-1].evaluations.append(EvalResult(reasoning=reasoning, score=score))
#         self.final_score = sum([result.score for result in self.results[-1].evaluations]) / len(self.results[-1].evaluations)
#         # self.score = sum([result.score for result in self.results[-1].evaluations])
    
#     def failed(self, error: Exception):
#         self.status = TestRunStatus.FAILURE
#         # self.error = error
        
#     def success(self):
#         self.status = TestRunStatus.SUCCESS


# class EvaluatorRelation(BaseModel):
#     id: int = ModelField(default=None)



# class TestTurn(BaseModel):
#     input: str = ModelField(default="")
#     output: str = ModelField(default="")
#     evaluators: list[EvaluatorRelation] = ModelField(default=[])


# class TestInputs(BaseModel):
#     turns: List[TestTurn] = ModelField(default=[])


# class TestCase(Model):
#     id: int = KeyField(primary_key=True)
#     title: str = ModelField(default="")
#     description: str = ModelField(default="")
#     input_turns: List[TestTurn] = ModelField(default=[])        
#     branch_id: int = ModelField(default=1)
#     turn_id: int = ModelField(default=1)
#     test_runs: Relation[TestRun] = RelationField(foreign_key="test_case_id")
    



# class TestRunEvaluator:
    
#     def __init__(self, test_run: TestRun, expected_output: str):
#         self.test_run = test_run
#         self.expected_output = expected_output
        
        
#     async def _run_evaluate(self, output: str):
#         return await evaluate_prompt(task=self.expected_output, response=output)
    
#     async def __call__(self, output: str):
#         self.test_run.add_turn(output)
#         eval_res = await self._run_evaluate(output)
#         self.test_run.add_eval_result(eval_res.reasoning, eval_res.score)
#         await self.test_run.save()
#         return eval_res


# class TestManager:
    
#     def __init__(self, test_case_id: int):
#         self._test_case_id = test_case_id
#         self._test_case = None
#         self._test_run = None
        
        
#     @property
#     def head_id(self):
#         if not self._test_run:
#             raise ValueError("Test run not initialized")
#         return self._test_run.id
        
#     @property
#     def status(self):
#         if not self._test_run:
#             raise ValueError("Test run not initialized")
#         return self._test_run.status
    
#     @property
#     def final_score(self):
#         if not self._test_run:
#             raise ValueError("Test run not initialized")
#         return self._test_run.final_score
        
#     @property
#     def test_case(self):
#         if not self._test_case:
#             raise ValueError("Test case not initialized")
#         return self._test_case
    
#     @property
#     def test_run(self):
#         if not self._test_run:
#             raise ValueError("Test run not initialized")
#         return self._test_run
        
#     async def _initialize_test(self):
#         self._test_case = await TestCase.get(self._test_case_id)
#         if not self._test_case:
#             raise Exception("Test case not found")
#         self._test_run = await self._test_case.add(TestRun())
#         await self._test_run.head.branch_from(head=self._test_case.head)
#         self._test_run
    
    
#     def iter_turns(self):
#         for turn in self.test_case.input_turns:
#             yield turn.input, TestRunEvaluator(self.test_run, turn.output)
    
#     async def __aenter__(self):
#         await self._initialize_test()
#         return self
    
    
#     async def __aexit__(self, exc_type, exc_value, traceback):
#         if exc_type:
#             self.test_run.failed(exc_value)
#         else:
#             self.test_run.success()            
#         await self.test_run.save()




    

