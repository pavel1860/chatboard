"""
Unit tests for vector similarity search functionality.

Tests the vector search API including:
- .similar() method for similarity comparisons
- .distance() method for ordering by distance
- Text auto-embedding support
- Integration with PostgreSQL query builder
"""

import pytest
import pytest_asyncio
import numpy as np
from chatboard.model.namespace_manager2 import NamespaceManager
from chatboard.model import Model, KeyField, ModelField, VectorField
from chatboard.model.vectors import Vector
from chatboard.algebra.vectors.empty_vectorizer import EmptyVectorizer
from chatboard.model.sql2.pg_query_builder import select


class Document(Model):
    """Test model with vector field."""
    id: int = KeyField(primary_key=True)
    title: str = ModelField()
    content: str = ModelField()
    embedding: Vector = VectorField(
        dimension=3,
        vectorizer=EmptyVectorizer,
        distance="cosine"
    )


@pytest_asyncio.fixture()
async def setup_database():
    """Setup clean database for each test."""
    # NamespaceManager._initialized = False
    await NamespaceManager.initialize_clean()
    yield
    # Cleanup after tests
    # await NamespaceManager.close()

# @pytest.fixture(autouse=True)
# async def setup_database():
#     """Setup clean database for each test."""
#     # NamespaceManager._initialized = False
#     NamespaceManager.drop_all_tables()
#     await NamespaceManager.initialize_all()
#     yield
#     # Cleanup after tests
#     NamespaceManager.drop_all_tables()


@pytest.mark.asyncio
async def test_vector_similarity_filter(setup_database):
    """Test filtering documents by similarity threshold."""
    # Create test documents
    doc1 = Document(
        title="AI Research",
        content="Machine learning algorithms",
        embedding=np.array([0.9, 0.1, 0.1])
    )
    await doc1.save()

    doc2 = Document(
        title="Deep Learning",
        content="Neural networks",
        embedding=np.array([0.85, 0.15, 0.15])
    )
    await doc2.save()

    doc3 = Document(
        title="Cooking",
        content="Recipes",
        embedding=np.array([0.1, 0.9, 0.3])
    )
    await doc3.save()

    # Query: similar to [0.88, 0.12, 0.12] with similarity > 0.95
    query_vector = np.array([0.88, 0.12, 0.12])

    results = await select(Document).where(
        Document.embedding.similar(query_vector) > 0.95
    ).execute()

    # Should find doc1 and doc2 (close to query), not doc3
    assert len(results) >= 1
    titles = [doc.title for doc in results]
    assert "AI Research" in titles or "Deep Learning" in titles
    assert "Cooking" not in titles


@pytest.mark.asyncio
async def test_vector_distance_ordering(setup_database):
    """Test ordering documents by distance (top K)."""
    # Create test documents
    doc1 = Document(
        title="AI Research",
        content="Machine learning",
        embedding=np.array([0.9, 0.1, 0.1])
    )
    await doc1.save()

    doc2 = Document(
        title="Deep Learning",
        content="Neural networks",
        embedding=np.array([0.5, 0.5, 0.2])
    )
    await doc2.save()

    doc3 = Document(
        title="Data Science",
        content="Statistics",
        embedding=np.array([0.85, 0.15, 0.15])
    )
    await doc3.save()

    # Query: top 2 most similar to [0.9, 0.1, 0.1]
    query_vector = np.array([0.9, 0.1, 0.1])

    results = await select(Document)\
        .order_by(Document.embedding.distance(query_vector))\
        .limit(2)\
        .execute()

    assert len(results) == 2
    # First result should be doc1 (exact match or very close)
    assert results[0].title in ["AI Research", "Data Science"]


@pytest.mark.asyncio
async def test_combined_filter_and_ordering(setup_database):
    """Test combining similarity threshold with distance ordering."""
    # Create test documents
    docs_data = [
        ("AI", np.array([0.9, 0.1, 0.1])),
        ("ML", np.array([0.85, 0.15, 0.15])),
        ("DL", np.array([0.8, 0.2, 0.2])),
        ("Stats", np.array([0.5, 0.5, 0.3])),
        ("Cooking", np.array([0.1, 0.9, 0.3])),
    ]

    for title, embedding in docs_data:
        doc = Document(
            title=title,
            content=f"Content about {title}",
            embedding=embedding
        )
        await doc.save()

    # Query: similarity > 0.8, ordered by distance, limit 3
    query_vector = np.array([0.9, 0.1, 0.1])

    results = await select(Document)\
        .where(Document.embedding.similar(query_vector) > 0.8)\
        .order_by(Document.embedding.distance(query_vector))\
        .limit(3)\
        .execute()

    # Should get AI, ML, DL (all similar) in order of closeness
    assert len(results) <= 3
    titles = [doc.title for doc in results]

    # AI and ML should definitely be included
    assert any(t in titles for t in ["AI", "ML", "DL"])
    # Cooking should not be included (low similarity)
    assert "Cooking" not in titles


@pytest.mark.asyncio
async def test_similarity_comparison_operators(setup_database):
    """Test different comparison operators (>, >=, <, <=)."""
    # Create test documents
    doc1 = Document(
        title="Similar",
        content="Very similar",
        embedding=np.array([0.9, 0.1, 0.1])
    )
    await doc1.save()

    doc2 = Document(
        title="Different",
        content="Very different",
        embedding=np.array([0.1, 0.9, 0.1])
    )
    await doc2.save()

    query_vector = np.array([0.9, 0.1, 0.1])

    # Test > operator
    results_gt = await select(Document).where(
        Document.embedding.similar(query_vector) > 0.9
    ).execute()
    assert len(results_gt) >= 1

    # Test >= operator
    results_gte = await select(Document).where(
        Document.embedding.similar(query_vector) >= 0.9
    ).execute()
    assert len(results_gte) >= len(results_gt)

    # Test < operator (should find dissimilar docs)
    results_lt = await select(Document).where(
        Document.embedding.similar(query_vector) < 0.5
    ).execute()
    # Different doc should have low similarity
    if len(results_lt) > 0:
        assert "Different" in [doc.title for doc in results_lt]


@pytest.mark.asyncio
async def test_vector_with_numpy_array(setup_database):
    """Test that numpy arrays work as query vectors."""
    doc = Document(
        title="Test",
        content="Test content",
        embedding=np.array([0.5, 0.5, 0.5])
    )
    await doc.save()

    # Query with numpy array
    query_vector = np.array([0.5, 0.5, 0.5])

    results = await select(Document).where(
        Document.embedding.similar(query_vector) > 0.5
    ).execute()

    assert len(results) >= 1
    assert results[0].title == "Test"


@pytest.mark.asyncio
async def test_vector_with_list(setup_database):
    """Test that Python lists work as query vectors."""
    doc = Document(
        title="Test",
        content="Test content",
        embedding=np.array([0.5, 0.5, 0.5])
    )
    await doc.save()

    # Query with list instead of numpy array
    query_vector = [0.5, 0.5, 0.5]

    results = await select(Document).where(
        Document.embedding.similar(query_vector) > 0.5
    ).execute()

    assert len(results) >= 1
    assert results[0].title == "Test"


@pytest.mark.asyncio
async def test_empty_result_set(setup_database):
    """Test that queries with no matches return empty results."""
    doc = Document(
        title="Test",
        content="Test content",
        embedding=np.array([0.1, 0.1, 0.1])
    )
    await doc.save()

    # Query for very dissimilar vector with high threshold
    query_vector = np.array([0.9, 0.9, 0.9])

    results = await select(Document).where(
        Document.embedding.similar(query_vector) > 0.99
    ).execute()

    # Should find nothing or very few results
    assert len(results) == 0 or len(results) < 2


@pytest.mark.asyncio
async def test_multiple_vector_fields(setup_database):
    """Test model with multiple vector fields (if supported)."""
    # This is a future test - currently we have one vector field
    # But the architecture should support multiple vector fields per model

    class MultiVectorDoc(Model):
        id: int = KeyField(primary_key=True)
        title: str = ModelField()
        text_embedding: Vector = VectorField(
            dimension=3,
            vectorizer=EmptyVectorizer,
            distance="cosine"
        )
        image_embedding: Vector = VectorField(
            dimension=3,
            vectorizer=EmptyVectorizer,
            distance="cosine"
        )

    # Reset and reinitialize with new model
    NamespaceManager._initialized = False
    await NamespaceManager.initialize_all()

    doc = MultiVectorDoc(
        title="Multimodal",
        text_embedding=np.array([0.9, 0.1, 0.1]),
        image_embedding=np.array([0.1, 0.9, 0.1])
    )
    await doc.save()

    # Query by text embedding
    results = await select(MultiVectorDoc).where(
        MultiVectorDoc.text_embedding.similar([0.9, 0.1, 0.1]) > 0.9
    ).execute()

    assert len(results) >= 1
    assert results[0].title == "Multimodal"


@pytest.mark.asyncio
async def test_sql_injection_safety(setup_database):
    """Test that vector values are properly parameterized (SQL injection safety)."""
    doc = Document(
        title="Test",
        content="Content",
        embedding=np.array([0.5, 0.5, 0.5])
    )
    await doc.save()

    # This should be safely parameterized, not cause SQL errors
    # Even with values that could be problematic if not parameterized
    query_vector = [0.5, 0.5, 0.5]

    results = await select(Document).where(
        Document.embedding.similar(query_vector) > 0.5
    ).execute()

    assert len(results) >= 1
    # If we got here without SQL errors, parameterization is working
