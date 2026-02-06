from typing import List, Optional
from ....utils.db_connections import PGConnectionManager
from ..models import Branch, Turn, TurnStatus

class PostgresBranchManager:
    async def get_ancestors(self, branch_id: int) -> List[Branch]:
        sql = """
            WITH RECURSIVE branch_hierarchy AS (
                SELECT * FROM branches WHERE id = $1
                UNION ALL
                SELECT b.* FROM branches b
                JOIN branch_hierarchy bh ON b.id = bh.forked_from_branch_id
            )
            SELECT * FROM branch_hierarchy;
        """
        rows = await PGConnectionManager.fetch(sql, branch_id)
        return [Branch(**row) for row in rows]

    async def get_descendants(self, branch_id: int) -> List[Branch]:
        sql = """
            WITH RECURSIVE branch_hierarchy AS (
                SELECT * FROM branches WHERE id = $1
                UNION ALL
                SELECT b.* FROM branches b
                JOIN branch_hierarchy bh ON b.forked_from_branch_id = bh.id
            )
            SELECT * FROM branch_hierarchy;
        """
        rows = await PGConnectionManager.fetch(sql, branch_id)
        return [Branch(**row) for row in rows]

    async def fork_branch(self, from_turn: Turn, name: Optional[str] = None) -> Branch:
        sql = """
            INSERT INTO branches (name, forked_from_index, forked_from_turn_id,
                                   current_index, forked_from_branch_id, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
            RETURNING *;
        """
        params = [
            name,
            from_turn.index,
            from_turn.id,
            from_turn.index,
            from_turn.branch_id
        ]
        row = await PGConnectionManager.fetch_one(sql, *params)
        return Branch(**row)


class PostgresTurnManager:
    async def history(
        self,
        branch_id: int,
        status: Optional[TurnStatus] = None,
        limit: Optional[int] = None
    ) -> List[Turn]:
        branch_mgr = PostgresBranchManager()
        ancestors = await branch_mgr.get_ancestors(branch_id)
        ancestor_ids = [b.id for b in ancestors]

        sql = """
            SELECT * FROM turns
            WHERE branch_id = ANY($1)
        """
        params = [ancestor_ids]

        if status is not None:
            sql += " AND status = $2"
            params.append(status.value)

        sql += " ORDER BY index DESC"

        if limit is not None:
            sql += f" LIMIT {limit}"

        rows = await PGConnectionManager.fetch(sql, *params)
        return [Turn(**row) for row in rows]

    async def start_turn(
        self, 
        branch: Branch, 
        message: str | None = None, 
        status: TurnStatus = TurnStatus.STAGED,
        metadata: dict | None = None
    ) -> Turn:
        sql = """
            WITH updated_branch AS (
                UPDATE branches
                SET current_index = current_index + 1
                WHERE id = $1
                RETURNING id, current_index
            ),
            new_turn AS (
                INSERT INTO turns (branch_id, index, created_at, status, message, metadata)
                SELECT id, current_index, NOW(), $2, $3, $4
                FROM updated_branch
                RETURNING *
            )
            SELECT * FROM new_turn;
        """
        row = await PGConnectionManager.fetch_one(sql, branch.id, status.value, message, metadata)
        return Turn(**row)

    async def update_status(self, turn: Turn, status: TurnStatus, message: Optional[str] = None) -> Turn:
        sql = """
            UPDATE turns
            SET status = $2, ended_at = NOW(), message = $3
            WHERE id = $1
            RETURNING *;
        """
        row = await PGConnectionManager.fetch_one(sql, turn.id, status.value, message)
        return Turn(**row)

    async def rewind(self, branch: Branch, to_turn: Turn):
        """
        Deletes all turns (and related assets) created after the given turn in the branch.
        """
        sql = """
            DELETE FROM turns
            WHERE branch_id = $1 AND index > $2
        """
        await PGConnectionManager.execute(sql, branch.id, to_turn.index)
