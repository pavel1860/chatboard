from typing import TYPE_CHECKING, Generic, TypeVar
from ..testing import TestCase, TestRun
from ..testing.evaluator_decorator import evaluator_store
from ..testing.test_models import Evaluation, EvaluatorConfig, TurnEval, TurnEvaluator
import asyncio
if TYPE_CHECKING:
    from ..model import Turn, Branch




TURN_MODEL = TypeVar("TURN_MODEL", bound="Turn")

class EvalRunner(Generic[TURN_MODEL]):
    
    def __init__(self, test_case: TestCase, test_run: TestRun, evaluators: list[EvaluatorConfig]):
        self.evaluators = []
        for e in evaluators:            
            eval_builder = evaluator_store.get(e.name)
            if eval_builder is not None:
                self.evaluators.append(eval_builder(test_case, test_run, e))
                
    async def __call__(self, ref_turn: TURN_MODEL, test_turn: TURN_MODEL):        
        results = await asyncio.gather(*[e(ref_turn, test_turn) for e in self.evaluators])
        score = sum(r.score for r in results) / len(results)
        return TurnEval(score=score, evaluations=results, test_turn_id=test_turn.id, ref_turn_id=ref_turn.id)








class TestRunner(Generic[TURN_MODEL]):
    
    def __init__(self, test_case: TestCase):
        self.test_case = test_case
        self.test_run = None
        self.test_branch = None
        
    
    async def run(self, branch: "Branch", turn: TURN_MODEL) -> TURN_MODEL:
        raise NotImplementedError("Subclass must implement this method")
    
    
    async def iter_turns(self, test_case: TestCase, test_run: TestRun):
        for turn in test_case.test_turns:
            turn_eval = test_case.evaluators.get(turn.id)
            yield turn, EvalRunner(test_case, test_run, turn_eval.evaluators)
    
    async def __call__(self):
        from ..model import Branch
        branch = await Branch.get(self.test_case.test_turns[0].branch_id)
        test_branch = await branch.fork(self.test_case.test_turns[0])

        with test_branch:
            test_run = await self.test_case.add(TestRun())
            async with test_run:
                async for ref_turn, evals in self.iter_turns(self.test_case, test_run):                     
                    # Run the Agent
                    test_turn = await self.run(test_branch, ref_turn)
                    evaluation = await evals(ref_turn, test_turn)            
                    await test_run.add(evaluation)
                test_run.score = sum(e.score for e in test_run.turn_evals) / len(test_run.turn_evals)
            
        print("Test Run " + str(test_run.id) + " completed with score " + str(test_run.score))

# async def run_test(test_case_id: int):
    # test_case = await TestCase.query().where(id=1).include(Turn.query().where(is_test_definition=True).include(BlockModel)).last()

    

            
