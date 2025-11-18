from typing import Any, Dict, Literal, Union, Optional, List
from sqlalchemy.orm import sessionmaker
from sqlalchemy import literal_column, select, func, create_engine
from sqlalchemy.orm.attributes import flag_modified
import os
import datetime as dt
from .models2 import Branch, MessageSession, Message, Turn, Base





class MessageList(list):
    
    
    def render_html(self):
        html_str = f"""<html>
<div style="display: flex; flex-direction: column; width: 400px; margin: 0;">
    {"".join([m.render_html() for m in self])}
</div>
</html>"""
        return html_str
    
    def show(self, display=True):
        from IPython.display import display, HTML        
        # display()
        html = HTML(self.render_html())
        if display:
            return display(html)            
        return html



MsgRole = Literal["user", "assistant", "tool"]

class History:
    """
    History manages conversation data, providing easy access to message history,
    branching, and session management.
    
    Basic usage:
    ```python
    # Create and initialize
    history = History()
    history.init()
    
    # Add messages
    history.add_message("Hello!", role="user")
    history.add_message("Hi there!", role="assistant")
    history.commit()  # Commits the current turn
    
    # Branch from current point
    new_branch = history.create_branch("Try different response")
    history.switch_to(new_branch)
    
    # Get conversation context
    messages = history.get_last_messages(limit=10)
    ```
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize History with optional database URL."""
        HISTORY_DB_URL = os.getenv("HISTORY_DB_URL", "postgresql://snack:Aa123456@localhost:5432/snackbot")
        self._engine = create_engine(db_url or HISTORY_DB_URL)
        SessionLocal = sessionmaker(bind=self._engine)
        self._db = SessionLocal()
        
        self._current_session: Optional[MessageSession] = None
        self._current_branch: Optional[Branch] = None
        self._current_turn: Optional[Turn] = None
        self._uncommitted_messages: List[Message] = []
        
    def init_new_session(self):
        """Initialize a new session and main branch."""
        Base.metadata.create_all(self._engine)
        session = MessageSession()
        branch = Branch(session=session)
        turn = Turn(branch=branch)
        self._db.add_all([session, branch, turn])
        self._db.commit()
        
        self._current_session = session
        self._current_branch = branch
        self._current_turn = turn
        
    def init_last_session(self):
        """Initialize the last session."""
        session = self._db.query(MessageSession).order_by(MessageSession.created_at.desc()).first()
        if not session:
            self.init_new_session()
            return
        self._current_session = session
        self._current_branch = session.branches[0]  
        if not self._current_branch:
            raise ValueError("No branches found in the last session")
        
        if self._current_branch.turns:
            self._current_turn = self._current_branch.turns[-1]
        else:
            self.add_turn()
        
        
    @property
    def turn(self) -> Turn:
        return self._current_turn
    
    @property
    def branch(self) -> Branch:
        return self._current_branch
    
    @property
    def session(self) -> MessageSession:
        return self._current_session
        
    def add_turn(self, branch: Branch | None = None):
        """
        Add a new turn to the specified branch or current branch.
        
        Args:
            branch: Optional branch to add turn to (defaults to current branch)
        """
        if branch is None:
            if not self._current_branch:
                raise ValueError("No current branch")
            branch = self._current_branch
        
        parent_turn = self._current_turn if self._current_turn else branch.forked_from_turn
        turn = Turn(
            branch=branch, 
            parent_turn=parent_turn, 
            local_state=parent_turn.local_state if parent_turn else {}
        )
        self._db.add(turn)
        self._db.flush()
        self._db.commit()
        self._current_turn = turn
        return turn
    
    def get_or_add_turn(self) -> Turn:
        if not self._current_turn:
            self._current_turn = Turn(branch=self._current_branch)
            self._db.add(self._current_turn)
            self._db.flush()
            self._db.commit()
        return self._current_turn
    
    def update_turn_local_state(self, state_key: str, local_state: Dict[str, Any]):
        if not self._current_turn:
            raise ValueError("No current turn")
        self._current_turn.local_state[state_key] = local_state
        flag_modified(self._current_turn, "local_state")
        self._db.add(self._current_turn)
        self._db.commit()
        
    def add_message(
        self, 
        content: str, 
        role: MsgRole = "user", 
        run_id: str | None = None,
        prompt: str | None = None,
        platform_id: str | None = None,
        action_calls: List[dict] | None = None,
        created_at: dt.datetime | None = None,
        # **kwargs,
    ) -> Message:
        """
        Add a message to the current turn.
        
        Args:
            content: The message content
            role: Message role ("user", "assistant", or "tool")
            **kwargs: Additional message attributes (action_calls, etc.)
        """
        try:
            if not self._current_branch:
                raise ValueError("No current branch")
            if not self._current_turn:
                self._current_turn = Turn(branch=self._current_branch)
                self._db.add(self._current_turn)
                self._db.flush()
                
            message = Message(
                content=content,
                role=role,
                turn=self._current_turn,
                branch_id=self._current_branch.id,
                run_id=run_id,
                prompt=prompt,
                platform_id=platform_id,
                action_calls=action_calls,
                created_at=created_at,
                # **kwargs
            )
            self._db.add(message)
            self._db.flush()  # Ensure message has an ID
            
            # Now that message exists, set it as turn's start message if needed
            if self._current_turn.start_message_id is None:
                self._current_turn.start_message_id = message.id
                self._db.add(self._current_turn)
                self._db.flush()
                
            self._uncommitted_messages.append(message)
            return message
            
        except Exception as e:
            self._db.rollback()  # Rollback on any error
            raise e
    
    def commit(self):
        """Commit the current turn and its messages."""
        if self._uncommitted_messages:
            self._db.add_all(self._uncommitted_messages)
            self._db.commit()
            self._uncommitted_messages = []
            # self._current_turn = None
    
    def create_branch(self, name: Optional[str] = None, from_turn: Optional[Turn] = None) -> Branch:
        """
        Create a new branch from the current point or specified turn.
        
        Args:
            name: Optional branch name
            from_turn: Optional turn to branch from (defaults to current)
        """
        if from_turn is None and self._current_turn:
            from_turn = self._current_turn
            
        branch = Branch(
            session=self._current_session,
            forked_from_turn=from_turn
        )
        self._db.add(branch)
        self._db.commit()
        return branch
    
    
    
    def get_branch_by_id(self, branch_id: int) -> Branch:
        return self._db.query(Branch).filter(Branch.id == branch_id).first()
    
    def switch_to(self, branch: Branch| int):
        """
        Switch to a different branch.
        
        Args:
            branch: The branch to switch to
        """
        if isinstance(branch, int):
            branch = self.get_branch_by_id(branch)
        self.commit()  # Commit any pending changes
        self._current_branch = branch
        self._current_turn = None
        return branch
    
    
    
    # def get_last_messages(self, limit: int = 10) -> List[Message]:
    #     """
    #     Get the last N messages from the current branch.
        
    #     Args:
    #         limit: Maximum number of messages to return
    #     """
    #     # Include uncommitted messages in the result
    #     committed = (
    #         self._db.query(Message)
    #         .filter(Message.branch_id == self._current_branch.id)
    #         .order_by(Message.created_at.desc())
    #         .limit(limit)
    #         .all()
    #     )
        
    #     all_messages = self._uncommitted_messages + committed
    #     return sorted(all_messages, key=lambda m: m.created_at)[-limit:]
    
    def get_branches(self) -> List[Branch]:
        """Get all branches in the current session."""
        return self._current_session.branches
    
    def get_branch_messages(self, branch: Branch) -> List[Message]:
        """Get all messages in a specific branch."""
        return (
            self._db.query(Message)
            .filter(Message.branch_id == branch.id)
            .order_by(Message.created_at)
            .all()
        )
    
    def rewind_to(self, turn: Turn):
        """
        Rewind the current branch to a specific turn.
        Creates a new branch from that turn.
        """
        branch = self.create_branch(from_turn=turn)
        self.switch_to(branch)
        
    def get_turn_messages(self, turn: Turn) -> List[Message]:
        """Get all messages in a specific turn."""
        return (
            self._db.query(Message)
            .filter(Message.turn_id == turn.id)
            .order_by(Message.created_at)
            .all()
        )
    
    def get_turns(self, branch: Optional[Branch] = None) -> List[Turn]:
        """Get all turns in the current or specified branch."""
        branch = branch or self._current_branch
        return (
            self._db.query(Turn)
            .filter(Turn.branch_id == branch.id)
            .order_by(Turn.created_at)
            .all()
        )
    
    def create_test_branch(self, name: Optional[str] = None) -> Branch:
        """Create a new test branch from the current point."""
        branch = self.create_branch(name)
        branch.is_test = True
        self._db.commit()
        return branch
    
    def get_test_branches(self) -> List[Branch]:
        """Get all test branches in the current session."""
        return (
            self._db.query(Branch)
            .filter(Branch.session_id == self._current_session.id)
            .filter(Branch.is_test == True)
            .all()
        )
        
    def clear_session(self):
        """Clear the current session."""
        self._db.query(MessageSession).delete()
        self._db.commit()
        self.init_new_session()
        
    def delete_branch(self, branch: Branch):
        """Delete a branch and all its messages."""
        if branch == self._current_branch:
            raise ValueError("Cannot delete the current branch")
        self._db.delete(branch)
        self._db.commit()
    
    def cleanup(self):
        """Close the database session."""
        self._db.close()
        
        
    def get_last_turns(self, limit: int = 10):
        if not self._current_branch:
            raise ValueError("No current branch")
        last_turn =self._current_branch.turns[-1]
        if not last_turn:
            return []
        turn_id = last_turn.id
        cte = self._turn_cte(turn_id, limit)        
        stmt = select(Turn).join(cte, Turn.id == cte.c.id).order_by(Turn.created_at.desc())
        return [t[0] for t in self._db.execute(stmt).all()]
    
    def get_last_messages(self, limit: int = 10):
        if not self._current_branch:
            raise ValueError("No current branch")
        last_turn =self._current_branch.turns[-1] if self._current_branch.turns else None
        if not last_turn:
            return []
        turn_id = last_turn.id
        cte = self._turn_cte(turn_id, limit)
        stmt = select(Message).join(cte, Message.turn_id == cte.c.id).order_by(Message.created_at.asc())
        return MessageList([m[0] for m in self._db.execute(stmt).all()])
        
    def _turn_cte(self, turn_id: int, limit: int = 10):
        t = Turn.__table__
        cte = (
            select(
                t.c.id.label("id"),
                t.c.parent_turn_id.label("parent_turn_id"),
                literal_column("1").label("depth")
            ).where(t.c.id == turn_id)
            .cte("turn_hierarchy", recursive=True)
        )
        cte = cte.union_all(
            select(
                t.c.id,
                t.c.parent_turn_id,
                (cte.c.depth + 1).label("depth")
            ).where(t.c.id == cte.c.parent_turn_id)
            .where(cte.c.depth < limit)
        )
        return cte
        
        
    
    def get_recent_turns_across_branches(self, limit: int = 10) -> List[Turn]:
        """
        Get the most recent turns across all branches using recursive CTE.
        Follows the branch fork history to get turns from all related branches.
        
        Args:
            limit: Maximum number of turns to return
        """
        # Start with all turns in the current branch
        current_branch_turns = (
            self._db.query(Turn)
            .filter(Turn.branch_id == self._current_branch.id)
            .subquery()
        )
        
        # Build recursive CTE starting from current branch's turns
        turn_cte = (
            select(Turn.__table__.c.id, Turn.__table__.c.created_at, Turn.__table__.c.branch_id)
            .select_from(current_branch_turns)
            .cte(recursive=True, name="turn_hierarchy")
        )
        
        # Follow the branch fork history through turns
        branch_alias = Branch.__table__.alias()
        turn_alias = Turn.__table__.alias()
        turn_cte = turn_cte.union_all(
            select(turn_alias.c.id, turn_alias.c.created_at, turn_alias.c.branch_id)
            .select_from(branch_alias)
            .join(turn_alias, turn_alias.c.branch_id == branch_alias.c.id)
            .where(branch_alias.c.forked_from_turn_id.in_(
                select(turn_cte.c.id)
            ))
        )
        
        # Query turns using the CTE
        turns = (
            self._db.query(Turn)
            .join(turn_cte, Turn.id == turn_cte.c.id)
            .order_by(Turn.created_at.desc())
            .limit(limit)
            .all()
        )
        
        return turns
    
    def get_recent_messages_across_branches(self, limit: int = 10) -> List[Message]:
        """
        Get the most recent messages across all branches using recursive CTE.
        First gets related turns, then finds messages in those turns.
        
        Args:
            limit: Maximum number of messages to return
        """
        # Get related turns first
        turn_cte = (
            select(Turn.__table__.c.id)
            .filter(Turn.branch_id == self._current_branch.id)
            .cte(recursive=True, name="turn_hierarchy")
        )
        
        # Follow the branch fork history through turns
        branch_alias = Branch.__table__.alias()
        turn_alias = Turn.__table__.alias()
        turn_cte = turn_cte.union_all(
            select(turn_alias.c.id)
            .select_from(branch_alias)
            .join(turn_alias, turn_alias.c.branch_id == branch_alias.c.id)
            .where(branch_alias.c.forked_from_turn_id.in_(
                select(turn_cte.c.id)
            ))
        )
        
        # Query messages from related turns
        messages = (
            self._db.query(Message)
            .join(Turn, Message.turn_id == Turn.id)
            .filter(Turn.id.in_(select(turn_cte.c.id)))
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )
        
        return messages
    
    def get_branch_fork_history(self, branch: Optional[Branch] = None) -> List[Branch]:
        """
        Get the fork history of a branch using recursive CTE.
        Returns list of branches in fork order (most recent first).
        
        Args:
            branch: Branch to get history for (defaults to current branch)
        """
        branch = branch or self._current_branch
        
        # Get all turns in the branch
        branch_turns = (
            self._db.query(Turn)
            .filter(Turn.branch_id == branch.id)
            .subquery()
        )
        
        # Build recursive CTE starting from branch's turns
        turn_cte = (
            select(Turn.__table__.c.id, Turn.__table__.c.branch_id)
            .select_from(branch_turns)
            .cte(recursive=True, name="turn_hierarchy")
        )
        
        # Follow the branch fork history through turns
        branch_alias = Branch.__table__.alias()
        turn_alias = Turn.__table__.alias()
        turn_cte = turn_cte.union_all(
            select(turn_alias.c.id, turn_alias.c.branch_id)
            .select_from(branch_alias)
            .join(turn_alias, turn_alias.c.branch_id == branch_alias.c.id)
            .where(branch_alias.c.forked_from_turn_id.in_(
                select(turn_cte.c.id)
            ))
        )
        
        # Query branches using the CTE
        branches = (
            self._db.query(Branch)
            .filter(Branch.id.in_(
                select(turn_cte.c.branch_id).distinct()
            ))
            .order_by(Branch.created_at.desc())
            .all()
        )
        
        return branches
    
    def copy_session(self, session_id: int) -> MessageSession:
        """
        Create a deep copy of a session including all its branches, turns, and messages.
        
        Args:
            session_id: ID of the session to copy
            
        Returns:
            The newly created session copy
        """
        # Get the source session
        source_session = self._db.query(MessageSession).get(session_id)
        if not source_session:
            raise ValueError(f"Session with id {session_id} not found")
            
        # Create new session
        new_session = MessageSession()
        self._db.add(new_session)
        self._db.flush()  # Get new session ID
        
        # Map of old IDs to new entities for reference
        branch_map = {}  # old_id -> new_branch
        turn_map = {}    # old_id -> new_turn
        
        # Copy branches
        for old_branch in source_session.branches:
            new_branch = Branch(
                session=new_session,
                is_test=old_branch.is_test
            )
            self._db.add(new_branch)
            self._db.flush()
            branch_map[old_branch.id] = new_branch
            
            # Copy turns for this branch
            for old_turn in old_branch.turns:
                new_turn = Turn(
                    branch=new_branch,
                    created_at=old_turn.created_at
                )
                self._db.add(new_turn)
                self._db.flush()
                turn_map[old_turn.id] = new_turn
                
                # Copy messages for this turn
                for old_message in old_turn.messages:
                    new_message = Message(
                        content=old_message.content,
                        role=old_message.role,
                        prompt=old_message.prompt,
                        run_id=old_message.run_id,
                        action_calls=old_message.action_calls,
                        turn=new_turn,
                        branch=new_branch,
                        created_at=old_message.created_at
                    )
                    self._db.add(new_message)
                    
                    # If this was the start message, set it on the new turn
                    if old_turn.start_message_id == old_message.id:
                        self._db.flush()  # Ensure new message has ID
                        new_turn.start_message_id = new_message.id
        
        # Set up branch relationships
        for old_branch in source_session.branches:
            new_branch = branch_map[old_branch.id]
            if old_branch.forked_from_turn_id:
                new_branch.forked_from_turn = turn_map.get(old_branch.forked_from_turn_id)
                
        # Set up turn parent relationships
        for old_turn_id, new_turn in turn_map.items():
            old_turn = self._db.query(Turn).get(old_turn_id)
            if old_turn.parent_turn_id:
                new_turn.parent_turn = turn_map.get(old_turn.parent_turn_id)
        
        self._db.commit()
        return new_session
    