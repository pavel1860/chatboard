
from chatboard.conversation.message_log import MessageLog
from chatboard.conversation.models import Branch, MessageBackend, TestCase, TestRun






class TestManager:
    def __init__(self):
        self._backend = MessageBackend()
        self.message_log = MessageLog(self._backend)

    async def create_test_case(self, test_case: TestCase) -> TestCase:
        if test_case.start_message is not None and not test_case.start_message.id:
            raise ValueError("Start message id is required. Message is not committed.")
        return await self._backend.add_test_case(test_case)
    
    async def list_test_cases(self, limit: int = 10, offset: int = 0, is_desc: bool = True) -> list[TestCase]:
        return await self._backend.list_test_cases(limit, offset, is_desc)
    
    async def get_test_case(self, id: int) -> TestCase:
        test_case = await self._backend.get_test_case(id)
        if test_case is None:
            raise ValueError(f"Test case with id {id} not found")
        return test_case
    
    async def delete_test_case(self, id: int):
        await self._backend.delete_test_case(id)
        
        
        
    async def start_test_run(self, test_case: TestCase):
        branch = await self._backend.add_branch(Branch(
            session_id=test_case.session_id,
            forked_from_branch_id=test_case.start_message.branch_id if test_case.start_message else None,
            forked_from_message_order=test_case.start_message.branch_order if test_case.start_message else None,
            is_test=True
        ))
        test_run = TestRun(
            test_case_id=test_case.id,
            status="INITIALIZED",
            branch_id=branch.id
        )
        test_run = await self._backend.add_test_run(test_run)
        return test_run
    
    
    async def update_test_run(
        self, 
        test_run: TestRun, 
        status: str | None = None, 
        error_message: str | None = None, 
        score: float | None = None
    ):
        data = {}
        if status is not None:
            data["status"] = status
        if error_message is not None:
            data["error_message"] = error_message
        if score is not None:
            data["score"] = score
        if len(data) == 0:
            raise ValueError("No data to update")
        test_run = await self._backend.update_test_run(test_run.id, **data)
        return test_run
    
    
    async def get_test_run(self, id: int) -> TestRun:
        test_run = await self._backend.get_test_run(id)
        if test_run is None:
            raise ValueError(f"Test run with id {id} not found")
        return test_run
        
