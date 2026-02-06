import pytest
from datetime import datetime, timedelta

import pytest_asyncio

from chatboard.model3.fields import ModelField, KeyField
from chatboard.model3.model3 import Model
from chatboard.model3.namespace_manager2 import NamespaceManager
from chatboard.model3.postgres2.pg_query_set import select


@pytest_asyncio.fixture()
async def setup_db():
    """Ensure we start with a clean DB schema for the tests."""
    # Drop all namespaces (tables)
    NamespaceManager.drop_all_tables()
    yield
    # Drop again after tests
    NamespaceManager.drop_all_tables()



@pytest.mark.asyncio
async def test_default_order_field_and_first_last(setup_db):
    # --- Define test model ---
    class Event(Model):
        id: int = KeyField(primary_key=True)
        name: str = ModelField()
        created_at: datetime = ModelField(order_by=True)

    # --- Initialize namespace ---
    await NamespaceManager.initialize_all()

    # --- Create events in random created_at order ---
    now = datetime.utcnow()
    e1 = await Event(name="First", created_at=now + timedelta(seconds=10)).save()
    e2 = await Event(name="Second", created_at=now).save()
    e3 = await Event(name="Third", created_at=now + timedelta(seconds=5)).save()

    # --- First should be earliest created_at ---
    first_event = await select(Event).first()
    assert first_event.name == "Second"
    assert first_event.created_at == e2.created_at

    # --- Last should be latest created_at ---
    last_event = await select(Event).last()
    assert last_event.name == "First"
    assert last_event.created_at == e1.created_at

    # --- get_default_order_field should return created_at ---
    ns = Event.get_namespace()
    assert ns.default_order_field == "created_at"

    # --- Offset test: get the 2nd earliest event ---
    second_event = await select(Event).order_by(ns.default_order_field).offset(1).limit(1).execute()
    assert len(second_event) == 1
    assert second_event[0].name == "Third"

    # --- Offset test: skip first two and get the 3rd earliest ---
    third_event = await select(Event).order_by(ns.default_order_field).offset(2).limit(1).execute()
    assert len(third_event) == 1
    assert third_event[0].name == "First"


@pytest.mark.asyncio
async def test_fallback_to_primary_key_if_no_order_by(setup_db):
    class Simple(Model):
        id: int = KeyField(primary_key=True, order_by=True)
        name: str = ModelField()

    await NamespaceManager.initialize_all()

    s1 = await Simple(name="A").save()
    s2 = await Simple(name="B").save()

    # PK ordering ascending
    first_simple = await select(Simple).first()
    assert first_simple.id == s1.id

    # PK ordering descending
    last_simple = await select(Simple).last()
    assert last_simple.id == s2.id

    ns = Simple.get_namespace()
    assert ns.default_order_field == "id"






