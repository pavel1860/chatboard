from typing import Any, Callable, Dict

from fastapi import HTTPException
from chatboard import testing
from chatboard.api.model_router2 import create_crud_router





# def connect_testing_routers(app, run_test: Callable[[testing.TestCase], Any], prefix: str = "/api"):
def connect_testing_routers(app, prefix: str = "/api"):
    
    test_case_router = create_crud_router(testing.TestCase)
    
    # @test_case_router.post("/run")
    # async def run_test_case_endpoint(data: dict):
    #     test_case_id = data.get("test_case_id")
    #     if not test_case_id:
    #         raise HTTPException(status_code=400, detail="test_case_id is required")
    #     test_case = await testing.TestCase.get(test_case_id)
    #     test_run = testing.TestRun(test_case=test_case)
    #     await test_case.test_runs.add(test_run)
    #     async def run_test_case(test_case: testing.TestCase):
    #         return await test_case.run()
    #     return await test_case.run()
    
    
    evaluator_router = create_crud_router(testing.Evaluator)
    test_run_router = create_crud_router(testing.TestRun)
    
    app.include_router(test_case_router, prefix=prefix + "/testing")
    app.include_router(evaluator_router, prefix=prefix + "/evaluators")
    app.include_router(test_run_router, prefix=prefix + "/testing")
    
    
    