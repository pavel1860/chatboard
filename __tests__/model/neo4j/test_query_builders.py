import pytest

# Assuming your classes are imported like this:
from chatboard.model.neo4j.cypher import (
    MatchQuery, MergeQuery, NodePattern, RelPattern, PatternChain, CyParam, CyProperty, CyEq
)

def test_match_single_node():
    q = (
        MatchQuery()
        .match(NodePattern("n", ["User"], {"id": CyParam("id")}))
        .where(CyEq(CyProperty("n", "email"), CyParam("email")))
        .return_("n")
        .limit(1)
        .set_param(id="42", email="me@site.com")
    )
    cypher, params = q.compile()
    assert cypher == "MATCH (n:User {id: $id}) WHERE n.email = $email RETURN n LIMIT 1"
    assert params == {"id": "42", "email": "me@site.com"}

def test_merge_single_node_with_set():
    q = (
        MergeQuery()
        .merge(NodePattern("n", ["User"], {"id": CyParam("id")}))
        .set(n__name=CyParam("name"))
        .return_("n")
        .set_param(id="42", name="Alice")
    )
    cypher, params = q.compile()
    assert cypher == "MERGE (n:User {id: $id}) SET n.name = $name RETURN n"
    assert params == {"id": "42", "name": "Alice"}

def test_merge_relationship_pattern_chain():
    q = (
        MergeQuery()
        .merge(PatternChain(
            NodePattern("a", ["User"], {"id": CyParam("user_id")}),
            RelPattern("", "FRIEND", "right"),
            NodePattern("b", ["User"], {"id": CyParam("friend_id")}),
        ))
        .return_("a", "b")
        .set_param(user_id="1", friend_id="2")
    )
    cypher, params = q.compile()
    assert cypher == "MERGE (a:User {id: $user_id})-[:FRIEND]->(b:User {id: $friend_id}) RETURN a, b"
    assert params == {"user_id": "1", "friend_id": "2"}

def test_merge_multiple_patterns():
    q = (
        MergeQuery()
        .merge(
            NodePattern("n", ["User"], {"id": CyParam("id")}),
            NodePattern("m", ["Group"], {"id": CyParam("gid")})
        )
        .return_("n", "m")
        .set_param(id="7", gid="g8")
    )
    cypher, params = q.compile()
    assert cypher == "MERGE (n:User {id: $id}), (m:Group {id: $gid}) RETURN n, m"
    assert params == {"id": "7", "gid": "g8"}

def test_set_multiple_properties():
    q = (
        MergeQuery()
        .merge(NodePattern("u", ["User"], {"id": CyParam("id")}))
        .set(u__name=CyParam("name"), u__email=CyParam("email"))
        .return_("u")
        .set_param(id="5", name="Bob", email="bob@x.com")
    )
    cypher, params = q.compile()
    assert cypher == "MERGE (u:User {id: $id}) SET u.name = $name, u.email = $email RETURN u"
    assert params == {"id": "5", "name": "Bob", "email": "bob@x.com"}

def test_pattern_chain_string():
    pattern = PatternChain(
        NodePattern("x", ["Foo"], {"foo": CyParam("foo")}),
        RelPattern("", "BAR", "right"),
        NodePattern("y", ["Bar"], {"bar": CyParam("bar")}),
    )
    assert str(pattern) == "(x:Foo {foo: $foo})-[:BAR]->(y:Bar {bar: $bar})"

def test_relpattern_variants():
    assert str(RelPattern("r", "TYPE", "right")) == "-[r:TYPE]->"
    assert str(RelPattern("r", "TYPE", "left")) == "<-[r:TYPE]-"
    assert str(RelPattern("r", "TYPE", "none")) == "-[r:TYPE]-"








def test_match_single_node_minimal():
    q = (
        MatchQuery()
        .match(NodePattern("n", ["User"]))
        .return_("n")
    )
    cypher, params = q.compile()
    assert cypher == "MATCH (n:User) RETURN n"
    assert params == {}

def test_match_node_with_props():
    q = (
        MatchQuery()
        .match(NodePattern("n", ["User"], {"id": CyParam("id")}))
        .return_("n")
        .set_param(id="u100")
    )
    cypher, params = q.compile()
    assert cypher == "MATCH (n:User {id: $id}) RETURN n"
    assert params == {"id": "u100"}

def test_match_node_and_relationship_path():
    q = (
        MatchQuery()
        .match(PatternChain(
            NodePattern("a", ["User"], {"id": CyParam("user_id")}),
            RelPattern("", "FRIEND", "right"),
            NodePattern("b", ["User"])
        ))
        .return_("a", "b")
        .set_param(user_id="12")
    )
    cypher, params = q.compile()
    assert cypher == "MATCH (a:User {id: $user_id})-[:FRIEND]->(b:User) RETURN a, b"
    assert params == {"user_id": "12"}

def test_match_multiple_patterns():
    q = (
        MatchQuery()
        .match(
            NodePattern("n", ["User"], {"id": CyParam("id")}),
            NodePattern("m", ["Group"], {"id": CyParam("gid")})
        )
        .return_("n", "m")
        .set_param(id="x", gid="g8")
    )
    cypher, params = q.compile()
    assert cypher == "MATCH (n:User {id: $id}), (m:Group {id: $gid}) RETURN n, m"
    assert params == {"id": "x", "gid": "g8"}

def test_match_with_where_and_order_by():
    q = (
        MatchQuery()
        .match(NodePattern("u", ["User"]))
        .where(CyEq(CyProperty("u", "active"), CyParam("active")))
        .return_("u")
        .order("u.name")
        .set_param(active=True)
    )
    cypher, params = q.compile()
    assert cypher == "MATCH (u:User) WHERE u.active = $active RETURN u ORDER BY u.name ASC"
    assert params == {"active": True}

def test_match_with_limit_and_offset():
    q = (
        MatchQuery()
        .match(NodePattern("u", ["User"]))
        .return_("u")
        .order("u.name")
        .limit(10)
    )
    cypher, params = q.compile()
    assert cypher == "MATCH (u:User) RETURN u ORDER BY u.name ASC LIMIT 10"
    assert params == {}

def test_match_with_multiple_where_conditions():
    q = (
        MatchQuery()
        .match(NodePattern("u", ["User"]))
        .where(
            CyEq(CyProperty("u", "email"), CyParam("email")),
            CyEq(CyProperty("u", "active"), CyParam("active"))
        )
        .return_("u")
        .set_param(email="foo@bar.com", active=True)
    )
    cypher, params = q.compile()
    # Your where() could support multiple expressions or one at a time.
    # If you want to allow .where(expr1, expr2), modify the implementation.
    assert (
        cypher == "MATCH (u:User) WHERE u.email = $email AND u.active = $active RETURN u"
        or cypher == "MATCH (u:User) WHERE u.active = $active AND u.email = $email RETURN u"
    )
    assert params == {"email": "foo@bar.com", "active": True}
