"""
A framework for version–controlling application tables (e.g. user_details, meal_plans, messages)
using PostgreSQL and SQLAlchemy.

The framework uses a git–like architecture:
  - Branches: a chain (or tree) of turns.
  - Turns: a commit point where one or more tables are snapshotted.
  - HEAD: a pointer to the current branch and turn.

The framework's own tables (Branch, Turn, Head, VersionedArtifact) are completely generic.
Application–specific tables can be "versioned" by saving a JSON snapshot into VersionedArtifact.
This allows you to later re–construct the full application state (by table) at any turn,
or even fork a branch to try alternative inputs.
"""

import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    Text,
    func,
    text,  # Add text import
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy_utils import LtreeType  # provides PostgreSQL ltree support
from sqlalchemy_utils.types.ltree import Ltree  # Import the Ltree value object

# Base class for ORM models.
Base = declarative_base()

###############################################################################
# Framework Tables for Versioning History
###############################################################################

class Branch(Base):
    __tablename__ = "branches"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    parent_id = Column(Integer, ForeignKey("branches.id", name="fk_branch_parent"), nullable=True)
    # The turn from which this branch was forked (if any)
    fork_turn_id = Column(Integer, ForeignKey("turns.id", name="fk_branch_fork_turn", use_alter=True, initially="DEFERRED"), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # LTree path (e.g. "1.5.12") for hierarchical queries (see PostgreSQL LTree)
    path = Column(LtreeType, index=True)
    
    # Self–referential relationship for parent/child branches.
    parent = relationship("Branch", remote_side=[id], backref="children")
    # Specify foreign_keys to avoid ambiguity with the Turn relationship
    fork_turn = relationship("Turn", foreign_keys=[fork_turn_id], post_update=True)
    # Add back_populates for the turns relationship
    turns = relationship("Turn", foreign_keys="[Turn.branch_id]", back_populates="branch")
    
    def __repr__(self):
        return f"<Branch(id={self.id}, name={self.name}, path={self.path})>"

class Turn(Base):
    __tablename__ = "turns"
    id = Column(Integer, primary_key=True)
    branch_id = Column(Integer, ForeignKey("branches.id", name="fk_turn_branch"), nullable=False)
    # Sequential number within the branch (like a commit number)
    turn_index = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # Optional commit message
    message = Column(Text)
    
    # Specify back_populates to avoid ambiguity
    branch = relationship("Branch", foreign_keys=[branch_id], back_populates="turns")
    
    def __repr__(self):
        return f"<Turn(id={self.id}, branch_id={self.branch_id}, turn_index={self.turn_index})>"

class Head(Base):
    """
    The HEAD table tracks the current branch and turn.
    """
    __tablename__ = "head"
    id = Column(Integer, primary_key=True)
    branch_id = Column(Integer, ForeignKey("branches.id"))
    turn_id = Column(Integer, ForeignKey("turns.id"))
    
    def __repr__(self):
        return f"<Head(branch_id={self.branch_id}, turn_id={self.turn_id})>"

class VersionedArtifact(Base):
    """
    This table stores snapshots of application–specific table records.
    
    Each record represents the state (as JSON) of a row from an application table at
    a given turn (and branch). For example, a snapshot of a row in the user_details table.
    """
    __tablename__ = "versioned_artifacts"
    id = Column(Integer, primary_key=True)
    branch_id = Column(Integer, ForeignKey("branches.id"), nullable=False)
    turn_id = Column(Integer, ForeignKey("turns.id"), nullable=False)
    table_name = Column(String, nullable=False)  # e.g. "user_details", "meal_plans", "messages"
    record_id = Column(String, nullable=False)  # the primary key (as string) of the versioned record
    data = Column(JSONB, nullable=False)         # JSON snapshot of the record
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    branch = relationship("Branch")
    turn = relationship("Turn")
    
    def __repr__(self):
        return f"<VersionedArtifact(id={self.id}, table={self.table_name}, record_id={self.record_id})>"

###############################################################################
# Application–Specific Models (Example: Nutrition Agent)
###############################################################################

# A simple mixin to help convert a SQLAlchemy model to a dict.
class VersionedMixin:
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

# class UserDetails(Base, VersionedMixin):
#     __tablename__ = "user_details"
#     id = Column(Integer, primary_key=True)
#     name = Column(String, nullable=False)
#     age = Column(Integer)
#     height = Column(Integer)  # e.g. centimeters
#     weight = Column(Integer)  # e.g. kilograms
    
#     def __repr__(self):
#         return f"<UserDetails(id={self.id}, name={self.name})>"

# class MealPlans(Base, VersionedMixin):
#     __tablename__ = "meal_plans"
#     id = Column(Integer, primary_key=True)
#     plan = Column(String, nullable=False)
#     user_id = Column(Integer, nullable=False)
    
#     def __repr__(self):
#         return f"<MealPlans(id={self.id}, user_id={self.user_id})>"

# class Messages(Base, VersionedMixin):
#     __tablename__ = "messages"
#     id = Column(Integer, primary_key=True)
#     sender = Column(String, nullable=False)
#     text = Column(Text, nullable=False)
#     user_id = Column(Integer, nullable=False)
    
#     def __repr__(self):
#         return f"<Messages(id={self.id}, sender={self.sender})>"

###############################################################################
# HistoryStore Framework Class
###############################################################################

class HistoryStore:
    """
    Implements framework–level operations: creating branches, committing turns,
    updating HEAD, and (importantly) versioning application–specific table records.
    
    The version_record() method is used by your application to "capture" the state
    (as JSON) of any record in any table. Later you can query (e.g. using get_versioned_state())
    to reconstruct the state of that table at any given turn.
    """
    def __init__(self, session):
        self.session = session

    def create_branch(self, name, parent_branch=None, fork_turn=None):
        new_branch = Branch(
            name=name,
            parent=parent_branch,
            fork_turn_id=fork_turn.id if fork_turn else None,
        )
        self.session.add(new_branch)
        self.session.flush()  # assign new_branch.id
        if parent_branch is None:
            new_branch.path = Ltree(str(new_branch.id))
        else:
            new_branch.path = Ltree(f"{parent_branch.path}.{new_branch.id}")
        self.session.commit()
        return new_branch

    def commit_turn(self, branch, message=None):
        last_turn = (
            self.session.query(Turn)
            .filter(Turn.branch_id == branch.id)
            .order_by(Turn.turn_index.desc())
            .first()
        )
        next_index = 1 if last_turn is None else last_turn.turn_index + 1
        new_turn = Turn(branch_id=branch.id, turn_index=next_index, message=message)
        self.session.add(new_turn)
        self.session.commit()
        return new_turn

    def update_head(self, branch, turn):
        head = self.session.query(Head).first()
        if head:
            head.branch_id = branch.id
            head.turn_id = turn.id
        else:
            head = Head(branch_id=branch.id, turn_id=turn.id)
            self.session.add(head)
        self.session.commit()
        return head

    def version_record(self, branch, turn, table_name, record_id, data):
        """
        Record a snapshot of a record (from any application table).
        
        :param branch: The branch in which this change occurs.
        :param turn: The Turn object (i.e. commit) under which this record version is saved.
        :param table_name: Name of the table being versioned (e.g., "user_details").
        :param record_id: The primary key (as string) of the record.
        :param data: A dict (typically via model.to_dict()) capturing the record's state.
        """
        artifact = VersionedArtifact(
            branch_id=branch.id,
            turn_id=turn.id,
            table_name=table_name,
            record_id=str(record_id),
            data=data,
        )
        self.session.add(artifact)
        self.session.commit()
        return artifact

    def get_versioned_state(self, branch, turn, table_name):
        """
        Reconstruct the state of a given table (e.g., "user_details") at the moment of a given turn.
        This uses a raw SQL query with PostgreSQL's DISTINCT ON to get the latest version of each record
        up to the given turn, including records from parent branches up to the fork point.
        """
        # Get the fork turn index if this is a child branch
        fork_turn_index = None
        if branch.fork_turn_id:
            fork_turn = self.session.query(Turn).filter(Turn.id == branch.fork_turn_id).first()
            if fork_turn:
                fork_turn_index = fork_turn.turn_index

        sql = text("""
            WITH branch_records AS (
                SELECT va.record_id, va.data, t.turn_index, b.path
                FROM versioned_artifacts va
                JOIN turns t ON va.turn_id = t.id
                JOIN branches b ON va.branch_id = b.id
                WHERE (b.path @> :branch_path OR b.id = :branch_id)
                  AND va.table_name = :table_name
                  AND (
                    -- For the current branch, include records up to the given turn
                    (b.id = :branch_id AND t.turn_index <= :turn_index)
                    -- For parent branches, include records up to the fork point
                    OR (b.path <@ :parent_path AND t.turn_index <= :fork_turn_index)
                  )
            )
            SELECT DISTINCT ON (record_id) record_id, data
            FROM branch_records
            ORDER BY record_id, path DESC, turn_index DESC
        """)
        
        # Get parent path by removing the last component
        parent_path = str(branch.path).rsplit('.', 1)[0] if '.' in str(branch.path) else ''
        
        result = self.session.execute(
            sql,
            {
                "branch_id": branch.id,
                "branch_path": str(branch.path),
                "parent_path": parent_path,
                "table_name": table_name,
                "turn_index": turn.turn_index,
                "fork_turn_index": fork_turn_index if fork_turn_index is not None else turn.turn_index
            },
        )
        state = {}
        for record_id, data in result:
            state[record_id] = data
        return state

    def get_context(self, branch, horizon=10):
        """
        For demonstration, reconstruct the overall application context by aggregating the versioned
        states for a list of registered tables. In a production framework you might register which
        tables are versioned; here we hard–code an example.
        """
        last_turn = (
            self.session.query(Turn)
            .filter(Turn.branch_id == branch.id)
            .order_by(Turn.turn_index.desc())
            .first()
        )
        if not last_turn:
            return {}
        context = {}
        for table in ["user_details", "meal_plans", "messages"]:
            context[table] = self.get_versioned_state(branch, last_turn, table)
        return context