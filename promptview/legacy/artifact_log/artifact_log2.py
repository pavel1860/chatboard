"""
A framework for version-controlling Model instances using a git-like architecture.
Models can opt-in to versioning by setting versioned=True in their Config class.

The framework uses the following concepts:
- Head: Points to the current branch and turn
- Branch: A chain of turns, can fork from any turn
- Turn: A commit point containing artifacts
- Artifact: A versioned instance of a Model
"""

import enum
import os
from datetime import datetime, timezone
from typing import (
    List, Optional, Type, TypeVar, Generic, Any, Dict, Union, Tuple, 
    Awaitable, cast, Mapping, AsyncGenerator, AsyncContextManager
)
from sqlalchemy import (
    Column, Integer, MetaData, String, DateTime, ForeignKey, Enum,
    create_engine, event, text, select, update, Boolean, Result, Table
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import relationship, declarative_base, Session, Mapped, mapped_column, joinedload
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.sql.expression import or_, and_
from sqlalchemy_utils import LtreeType
from sqlalchemy_utils.types.ltree import Ltree
from promptview.model.model import Model
from contextlib import asynccontextmanager

from promptview.model.postgres_client import model_to_table_name

Base = declarative_base()
MODEL = TypeVar('MODEL', bound=Model)

metadata = MetaData()
class SessionManager:
    _instance = None
    _engine = None
    _session_maker = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance

    async def initialize(self, url: str | None = None) -> None:
        if self._engine is None:
            url = url or os.environ.get("POSTGRES_URL", "postgresql://snack:Aa123456@localhost:5432/snackbot")
            # Convert the URL to use async driver
            if not url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://")
            self._engine = create_async_engine(url, future=True, echo=True)
            self._session_maker = async_sessionmaker(
                self._engine, 
                expire_on_commit=False, 
                future=True, 
                # autoflush=False
            )
        

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        if self._session_maker is None:
            await self.initialize()
        assert self._session_maker is not None  # for type checker
        async with self._session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
                
    async def initialize_tables(self) -> None:
        # async with self.session() as session:
        if self._engine is None:
            await self.initialize()
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS ltree"))
            await conn.run_sync(Base.metadata.create_all)
            
    async def drop_tables(self) -> None:
        if self._engine is None:
            await self.initialize()
        async with self._engine.begin() as conn:
            # Drop all tables with CASCADE to force drop even with dependencies
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            # Recreate the public schema
            await conn.execute(text("CREATE SCHEMA public"))
            # Drop all tables through SQLAlchemy as well
            await conn.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_maker = None

class TurnStatus(enum.Enum):
    STAGED = "staged"
    COMMITTED = "committed"
    REVERTED = "reverted"

class Head(Base):
    __tablename__ = "heads"
    id: Mapped[int] = mapped_column(primary_key=True)
    branch_id: Mapped[int] = mapped_column(ForeignKey("branches.id"), nullable=False)
    turn_id: Mapped[int] = mapped_column(ForeignKey("turns.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    branch: Mapped["Branch"] = relationship("Branch", back_populates="head")
    turn: Mapped["Turn"] = relationship("Turn", back_populates="head")
    branches: Mapped[List["Branch"]] = relationship("Branch", back_populates="head")

class Branch(Base):
    __tablename__ = "branches"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[Ltree] = mapped_column(LtreeType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    head: Mapped[Optional["Head"]] = relationship("Head", back_populates="branch", uselist=False)
    turns: Mapped[List["Turn"]] = relationship("Turn", back_populates="branch")

class Turn(Base):
    __tablename__ = "turns"
    id: Mapped[int] = mapped_column(primary_key=True)
    branch_id: Mapped[int] = mapped_column(ForeignKey("branches.id"), nullable=False)
    index: Mapped[int] = mapped_column(nullable=False)  # Sequential index within branch
    status: Mapped[TurnStatus] = mapped_column(Enum(TurnStatus), nullable=False, default=TurnStatus.STAGED)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    branch: Mapped["Branch"] = relationship("Branch", back_populates="turns")
    head: Mapped[Optional["Head"]] = relationship("Head", back_populates="turn", uselist=False)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    artifacts: Mapped[List["BaseArtifact"]] = relationship("BaseArtifact", back_populates="turn")

class BaseArtifact(Base):
    """Base class for all artifact tables."""
    __tablename__ = 'base_artifacts'
    __mapper_args__ = {
        'polymorphic_identity': 'base_artifact',
        'polymorphic_on': 'type'
    }

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(50))
    model_id: Mapped[str] = mapped_column(String(255))
    turn_id: Mapped[int] = mapped_column(Integer, ForeignKey('turns.id'))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    turn = relationship('Turn', back_populates='artifacts')

def create_artifact_table_for_model(model_cls: Type[Model]) -> type:
    """Create a new artifact table for a model class."""
    table_name = f"{model_to_table_name(model_cls)}_artifacts"
    
    # Create the artifact class
    class ModelArtifact(BaseArtifact):
        __tablename__ = table_name
        __mapper_args__ = {
            'polymorphic_identity': table_name,
        }
        
        # Add primary key that links to parent
        id: Mapped[int] = mapped_column(
            Integer, 
            ForeignKey('base_artifacts.id', ondelete='CASCADE'),
            primary_key=True
        )
        
        # Add reference to the model class
        model_class = model_cls
    
    return ModelArtifact

# Register event listener to automatically create artifact tables for versioned models
@event.listens_for(Table, 'after_create')
def create_versioned_artifacts(target, connection, **kw):
    """Create artifact tables for all versioned models after database creation."""
    if hasattr(target, 'model_class'):
        model_cls = target.model_class
        if hasattr(model_cls, 'Config') and hasattr(model_cls.Config, 'versioned') and model_cls.Config.versioned:
            artifact_cls = create_artifact_table_for_model(model_cls)
            if not artifact_cls.__table__.exists(connection):
                artifact_cls.__table__.create(connection)

# Type variable for generic artifact queries
MODEL = TypeVar('MODEL', bound=Model)

def get_artifact_class_for_model(model_cls: Type[MODEL]) -> Type[BaseArtifact]:
    """Get the artifact class for a model."""
    table_name = f"{model_to_table_name(model_cls)}_artifacts"
    for cls in BaseArtifact.__subclasses__():
        if cls.__tablename__ == table_name:
            return cls
    raise ValueError(f"No artifact class found for model {model_cls.__name__}")

class ArtifactQuery(Generic[MODEL]):
    """Query builder for artifacts."""
    def __init__(self, session_manager: SessionManager, model_cls: Type[MODEL]):
        self._session_manager = session_manager
        self.model_cls = model_cls
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._turn_filter: Optional[Dict[str, Union[int, str]]] = None
        self._time_filter: Optional[Dict[str, Union[datetime, Tuple[datetime, datetime]]]] = None
        self._field_filter: Optional[Dict[str, Any]] = None
        self._include_parent_branches: bool = True

    async def execute(self) -> List[MODEL]:
        """Execute the query and return model instances."""
        # Get the artifact class for this model
        artifact_cls = get_artifact_class_for_model(self.model_cls)
        
        # Build the base query
        # model_table = Table(model_to_table_name(self.model_cls), Base.metadata, autoload_with=self._session_manager._engine)
        model_table = metadata.tables.get(model_to_table_name(self.model_cls) + "_artifacts")
        if model_table is None:
            raise ValueError(f"Model table {model_to_table_name(self.model_cls)} not found")
        query = (
            select(model_table)
            .join(artifact_cls, artifact_cls.model_id == model_table.id)
            .join(Turn, artifact_cls.turn_id == Turn.id)
            .join(Branch, Turn.branch_id == Branch.id)
        )
        
        # Apply turn filters
        if self._turn_filter:
            if 'at' in self._turn_filter:
                turn_id = int(self._turn_filter['at'])
                query = query.where(Turn.id == turn_id)
            elif 'up_to' in self._turn_filter:
                turn_id = int(self._turn_filter['up_to'])
                query = query.where(Turn.id <= turn_id)
        
        # Apply time filters
        if self._time_filter:
            if 'at' in self._time_filter:
                timestamp = cast(datetime, self._time_filter['at'])
                query = query.where(Turn.created_at <= timestamp)
            elif 'between' in self._time_filter:
                time_range = cast(Tuple[datetime, datetime], self._time_filter['between'])
                query = query.where(Turn.created_at.between(*time_range))
        
        # Apply field filters
        if self._field_filter:
            for field, value in self._field_filter.items():
                query = query.where(getattr(self.model_cls, field) == value)
        
        # Apply limit and offset
        if self._limit is not None:
            query = query.limit(self._limit)
        if self._offset is not None:
            query = query.offset(self._offset)
        
        # Execute query
        async with self._session_manager.session() as session:
            result = await session.execute(query)
            return list(result.scalars().all())

    def limit(self, limit: int) -> 'ArtifactQuery[MODEL]':
        """Set the limit for the query."""
        if limit <= 0:
            raise ValueError("Limit must be positive")
        self._limit = limit
        return self

    def offset(self, offset: int) -> 'ArtifactQuery[MODEL]':
        """Set the offset for the query."""
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        self._offset = offset
        return self

    def at_turn(self, turn_id: int) -> "ArtifactQuery[MODEL]":
        """Get artifacts at a specific turn."""
        self._turn_filter = {"at": turn_id}
        return self
    
    def up_to_turn(self, turn_id: int) -> "ArtifactQuery[MODEL]":
        """Get artifacts up to and including a specific turn."""
        self._turn_filter = {"up_to": turn_id}
        return self
    
    def at_time(self, timestamp: datetime) -> "ArtifactQuery[MODEL]":
        """Get artifacts at a specific timestamp."""
        self._time_filter = {"at": timestamp}
        return self
    
    def between(self, start_time: datetime, end_time: datetime) -> "ArtifactQuery[MODEL]":
        """Get artifacts between timestamps."""
        self._time_filter = {"between": (start_time, end_time)}
        return self
    
    def latest_turn(self) -> "ArtifactQuery[MODEL]":
        """Get artifacts from the latest turn only."""
        self._turn_filter = {"type": "latest"}
        return self
    
    def exclude_parent_branches(self) -> "ArtifactQuery[MODEL]":
        """Only get artifacts from the current branch."""
        self._include_parent_branches = False
        return self
    
    def filter(self, predicate) -> "ArtifactQuery[MODEL]":
        """Filter artifacts by model fields."""
        self._field_filter = predicate
        return self
    
    async def first(self) -> Optional[MODEL]:
        """Get the first result."""
        self._limit = 1
        results = await self.execute()
        return results[0] if results else None
    
    async def last(self, n: Optional[int] = None) -> List[MODEL]:
        """Get the last n results."""
        if n is not None:
            self._limit = n
        # TODO: Implement actual query
        return []

class ArtifactLog:
    """
    Main class for managing versioned models and their artifacts.
    """
    def __init__(self):
        self._artifact_tables = {}
        self._session_manager = SessionManager()
        self._head = None
    
    async def register_model(self, model_cls: Type[Model]):
        """
        Register a versioned model and create its artifact table.
        """
        namespace = await model_cls.get_namespace()
        if namespace.name in self._artifact_tables:
            return self._artifact_tables[namespace.name]
        
        artifact_cls = create_artifact_table_for_model(model_cls)
        self._artifact_tables[namespace.name] = artifact_cls
        
        # Create the table in the database
        # async with self._session_manager.session() as session:
        #     async with session.begin():
        #         await session.run_sync(lambda conn: artifact_cls.__table__.create(conn, checkfirst=True))
        
        await self._session_manager.initialize_tables()
        return artifact_cls
    
    async def init_head(self, id: int | None = None) -> Head:
        """
        Initialize a new head with a main branch and initial turn.
        If id is provided, loads existing head instead of creating new one.
        """
        # First ensure base tables are created
        await self._session_manager.initialize_tables()
        
        async with self._session_manager.session() as session:
            if id is not None:
                # Load existing head
                stmt = (
                    select(Head)
                    .options(joinedload(Head.branch))
                    .options(joinedload(Head.turn))
                    .where(Head.id == id)
                )
                result = await session.execute(stmt)
                head = result.scalar_one_or_none()
                if head:
                    self._head = head
                    return head
                
            # Create new head
            branch = Branch(
                name="main", 
                path=Ltree("1")  # Root path
            )
            session.add(branch)
            await session.flush()
            
            # Create initial turn
            turn = Turn(
                branch=branch,
                index=1,
                status=TurnStatus.STAGED
            )
            session.add(turn)
            
            # Create head
            head = Head(
                id=id,
                branch=branch,
                turn=turn
            )
            session.add(head)
            await session.flush()
            
            # Update branch with head reference
            branch.head = head
            await session.flush()
            
            self._head = head
            return head
        
    @property
    def head(self) -> Head:
        """Get the current head."""
        if self._head is None:
            raise ValueError("No active head found")
        return self._head
    
    
    async def stage_artifact(self, model) -> BaseArtifact:
        """
        Stage a model instance as an artifact in the current turn.
        """
        config = getattr(model.__class__, 'Config', None)
        if not config or not getattr(config, 'versioned', False):
            raise ValueError(f"Model {model.__class__.__name__} is not versioned")
        
        # Get or create artifact table for this model
        artifact_cls = await self.register_model(model.__class__)
        
        # Get current head
        current_head = self.head
        if not current_head:
            raise ValueError("No active head found")
        
        # Create artifact
        artifact = artifact_cls(
            turn=current_head.turn,
            model_id=model.id
        )
        async with self._session_manager.session() as session:
            session.add(artifact)
            await session.flush()
        
        return artifact
    
    async def commit_turn(self, message: str | None = None) -> Turn:
        """
        Commit the current turn and create a new one.
        """
        if self._head is None:
            raise ValueError("No active head found")
        
        async with self._session_manager.session() as session:
            # Merge the current head into the session so that it's tracked
            current_head = await session.merge(self._head)
            
            # End current turn
            current_turn = current_head.turn
            current_turn.status = TurnStatus.COMMITTED
            current_turn.ended_at = datetime.now(timezone.utc).replace(tzinfo=None)
            current_turn.message = message
            
            session.add(current_turn)
            await session.flush()
            
            # Create new turn
            new_turn = Turn(
                branch=current_head.branch,
                index=current_turn.index + 1,
                status=TurnStatus.STAGED
            )
            session.add(new_turn)
            await session.flush()
            
            # Update head to reference the new turn
            current_head.turn = new_turn
            session.add(current_head)
            await session.flush()
            
            # Update the stored head
            self._head = current_head

        return new_turn
    
    
    async def branch_from(self, turn_id: int, name: str | None = None, check_out: bool = False) -> Branch:
        """
        Create a new branch from a turn.
        """
        # Get source turn
        stmt = select(Turn).where(Turn.id == turn_id)
        async with self._session_manager.session() as session:
            result = await session.execute(stmt)
            source_turn = result.scalar_one_or_none()
            if not source_turn:
                raise ValueError(f"Turn {turn_id} not found")
        
        # Get current head
        current_head = self.head
        if not current_head:
            raise ValueError("No active head found")
        
        # Create new branch
        new_branch = Branch(
            name=name,
            path=Ltree(f"{source_turn.branch.path}.{source_turn.id}")
        )
        async with self._session_manager.session() as session:
            session.add(new_branch)
            await session.flush()
        
        if check_out:
            # Update head to point to new branch
            current_head.branch = new_branch
            current_head.turn = source_turn
        
        await session.flush()
        return new_branch
    
    async def checkout_branch(self, branch_id: int) -> None:
        """
        Switch HEAD to a different branch.
        """
        # Get branch
        stmt = select(Branch).where(Branch.id == branch_id)
        async with self._session_manager.session() as session:
            result = await session.execute(stmt)
            branch = result.scalar_one_or_none()
            if not branch:
                raise ValueError(f"Branch {branch_id} not found")
        
        # Get current head
        current_head = self.head
        if not current_head:
            raise ValueError("No active head found")
        
        # Get or create staged turn
        stmt = (
            select(Turn)
            .where(Turn.branch_id == branch_id, Turn.status == TurnStatus.STAGED)
            .order_by(Turn.index.desc())
        )
        async with self._session_manager.session() as session:
            result = await session.execute(stmt)
            turn = result.scalar_one_or_none()
        
        if not turn:
            # Create new turn if none is staged
            turn = Turn(
                branch=branch,
                index=1,
                status=TurnStatus.STAGED
            )
            async with self._session_manager.session() as session:
                session.add(turn)
        
        # Update head
        current_head.branch = branch
        current_head.turn = turn
        await session.flush()
    
    async def revert_turn(self) -> Turn:
        """
        Revert the current turn and create a new one.
        """
        # Get current head
        current_head = self.head
        if not current_head:
            raise ValueError("No active head found")
        
        current_turn = current_head.turn
        current_turn.status = TurnStatus.REVERTED
        current_turn.ended_at = datetime.utcnow()
        
        # Create new turn
        new_turn = Turn(
            branch=current_head.branch,
            index=current_turn.index + 1,
            status=TurnStatus.STAGED
        )
        async with self._session_manager.session() as session:
            session.add(new_turn)
            await session.flush()
        
        # Update head
        current_head.turn = new_turn
        await session.flush()
        
        return new_turn
    
    async def revert_to_turn(self, turn_id: int) -> Turn:
        """
        Revert to a specific turn, marking all later turns as reverted.
        """
        # Get target turn
        stmt = select(Turn).where(Turn.id == turn_id)
        async with self._session_manager.session() as session:
            result = await session.execute(stmt)
            target_turn = result.scalar_one_or_none()
            if not target_turn:
                raise ValueError(f"Turn {turn_id} not found")
        
        # Get current head
        current_head = self.head
        if not current_head:
            raise ValueError("No active head found")
        
        # Mark all later turns as reverted
        stmt = (
            update(Turn)
            .where(
                Turn.branch_id == target_turn.branch_id,
                Turn.index > target_turn.index,
                Turn.status != TurnStatus.REVERTED
            )
            .values(
                status=TurnStatus.REVERTED,
                ended_at=datetime.utcnow()
            )
        )
        async with self._session_manager.session() as session:
            await session.execute(stmt)
        
        # Create new turn
        new_turn = Turn(
            branch=target_turn.branch,
            index=target_turn.index + 1,
            status=TurnStatus.STAGED
        )
        async with self._session_manager.session() as session:
            session.add(new_turn)
            await session.flush()
        
        # Update head
        current_head.branch = target_turn.branch
        current_head.turn = new_turn
        await session.flush()
        
        return new_turn
    
    def get_artifact(self, model_type: Type[MODEL]) -> ArtifactQuery[MODEL]:
        """
        Get a query builder for artifacts.
        """
        config = getattr(model_type, 'Config', None)
        if not config or not getattr(config, 'versioned', False):
            raise ValueError(f"Model {model_type.__name__} is not versioned")
        
        return ArtifactQuery(self._session_manager, model_type)
    
    async def get_turn(self, turn_id: int) -> Turn:
        """Get a specific turn."""
        stmt = select(Turn).where(Turn.id == turn_id)
        async with self._session_manager.session() as session:
            result = await session.execute(stmt)
            turn = result.scalar_one_or_none()
            if not turn:
                raise ValueError(f"Turn {turn_id} not found")
            return turn
    
    async def get_branch(self, branch_id: int) -> Branch:
        """Get a specific branch."""
        stmt = select(Branch).where(Branch.id == branch_id)
        async with self._session_manager.session() as session:
            result = await session.execute(stmt)
            branch = result.scalar_one_or_none()
            if not branch:
                raise ValueError(f"Branch {branch_id} not found")
            return branch