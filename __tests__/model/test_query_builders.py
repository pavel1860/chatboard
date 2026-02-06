import pytest

from chatboard.model.fields import ModelField
from chatboard.model.model3 import Model
from chatboard.model.postgres2.pg_query_set import PgSelectQuerySet
from chatboard.model.sql.compiler import Compiler
from chatboard.model.sql.expressions import And, Like, In, Between, Eq, Gt, Value, Or, Not, Function, Coalesce, IsNull
from chatboard.model.sql.queries import Column, DeleteQuery, InsertQuery, SelectQuery, Subquery, Table, UpdateQuery
from __tests__.model.utils import assert_sql
import datetime as dt



def test_simple_select():
    users = Table("users", "u")
    u_id = Column("id", users)
    u_name = Column("name", users)

    query = SelectQuery()
    query.columns = [u_id, u_name]
    query.from_table = users
    query.where &= Eq(u_id, Value(42, inline=False))

    expected_sql = """
    SELECT u.id, u.name
    FROM users AS u
    WHERE (u.id = $1)
    """
    expected_params = [42]

    assert_sql(query, expected_sql, expected_params)



def test_simple_select_with_where_and_join():
    users = Table("users", alias="u")
    posts = Table("posts", alias="p")

    query = SelectQuery()
    query.from_table = users
    query.columns = [Column("id", users), Column("title", posts)]
    query.inner_join(
        posts,
        Eq(Column("id", users), Column("user_id", posts))
    )
    query.where &= Gt(Column("age", users), Value(21, inline=False))

    compiler = Compiler()
    sql, params = compiler.compile(query)

    expected_sql = (
        "SELECT u.id, p.title\n"
        "FROM users AS u\n"
        "INNER JOIN posts AS p ON (u.id = p.user_id)\n"
        "WHERE (u.age > $1)"
    )

    assert_sql(query, expected_sql, [21])

def test_select_with_multiple_conditions_and():
    users = Table("users", alias="u")

    query = SelectQuery()
    query.from_table = users
    query.columns = [Column("id", users)]
    query.where(
        Gt(Column("age", users), Value(21, inline=False)) &
        Eq(Column("is_active", users), Value(True, inline=False))
    )

    compiler = Compiler()
    sql, params = compiler.compile(query)

    expected_sql = (
        "SELECT u.id\n"
        "FROM users AS u\n"
        "WHERE ((u.age > $1) AND (u.is_active = $2))"
    )

    assert_sql(query, expected_sql, [21, True])


def test_select_with_condition_aggregation():
    users = Table("users", alias="u")

    query = SelectQuery()
    query.from_table = users
    query.columns = [Column("id", users)]
    query.where &= Gt(Column("age", users), Value(21, inline=False))         
    query.where &= Eq(Column("is_active", users), Value(True, inline=False))
    query.where |= Eq(Column("name", users), Value("John", inline=False))
    

    compiler = Compiler()
    sql, params = compiler.compile(query)

    expected_sql = (
        "SELECT u.id\n"
        "FROM users AS u\n"
        "WHERE (((u.age > $1) AND (u.is_active = $2)) OR (u.name = $3))"
    )

    assert_sql(query, expected_sql, [21, True, "John"])



def test_select_with_or_and_not():
    users = Table("users", alias="u")

    

    query = SelectQuery()
    query.from_table = users
    query.columns = [Column("id", users)]
    query.where &= Not(
        Or(
            Eq(Column("role", users), Value("admin", inline=False)),
            Eq(Column("role", users), Value("moderator", inline=False)),
        )
    )

    expected_sql = (
        "SELECT u.id\n"
        "FROM users AS u\n"
        "WHERE (NOT ((u.role = $1) OR (u.role = $2)))"
    )

    assert_sql(query, expected_sql, ["admin", "moderator"])





def test_select_with_coalesce_and_json_agg():        

    # Tables
    users = Table("users", "u")
    posts = Table("posts", "p")
    likes = Table("likes", "l")

    # Columns
    u_id = Column("id", users)
    u_name = Column("name", users)
    u_age = Column("age", users)
    u_created = Column("created_at", users)
    u_address = Column("address", users)

    p_id = Column("id", posts)
    p_user_id = Column("user_id", posts)
    p_created = Column("created_at", posts)
    p_title = Column("title", posts)
    p_content = Column("content", posts)

    l_id = Column("id", likes)
    l_created = Column("created_at", likes)
    l_post_id = Column("post_id", likes)

    # Subquery for likes
    likes_obj = Function(
        "jsonb_build_object",
        Value("id"), l_id,
        Value("created_at"), l_created,
        Value("post_id"), l_post_id
    )

    likes_subq = SelectQuery()
    likes_subq.columns = [Function("json_agg", likes_obj)]
    likes_subq.from_table = likes
    likes_subq.where &= Eq(l_post_id, p_id)

    likes_coalesced = Coalesce(likes_subq, Value("[]", inline=True))

    # Posts aggregation
    post_obj = Function(
        "jsonb_build_object",
        Value("id"), p_id,
        Value("created_at"), p_created,
        Value("title"), p_title,
        Value("content"), p_content,
        Value("user_id"), p_user_id,
        Value("likes"), likes_coalesced
    )

    posts_agg = Function(
        "json_agg", post_obj,
        distinct=True,
        filter_where=Not(IsNull(p_id))
    )

    posts_coalesced = Coalesce(posts_agg, Value("[]", inline=True), alias="posts")

    # Final query
    query = SelectQuery()
    query.columns = [u_id, u_created, u_name, u_age, u_address, posts_coalesced]
    query.from_table = users
    query.join(posts, Eq(u_id, p_user_id))
    query.join(likes, Eq(p_id, l_post_id))
    query.group_by = [u_id]
  
    
    expected_sql = """
    SELECT u.id, u.created_at, u.name, u.age, u.address, COALESCE(json_agg(DISTINCT jsonb_build_object('id', p.id, 'created_at', p.created_at, 'title', p.title, 'content', p.content, 'user_id', p.user_id, 'likes', COALESCE((SELECT json_agg(jsonb_build_object('id', l.id, 'created_at', l.created_at, 'post_id', l.post_id))
    FROM likes AS l
    WHERE (l.post_id = p.id)), '[]'))) FILTER (WHERE (NOT (p.id IS NULL))), '[]') AS posts
    FROM users AS u
    LEFT JOIN posts AS p ON (u.id = p.user_id) LEFT JOIN likes AS l ON (p.id = l.post_id)
    GROUP BY u.id
    """
    expected_params = []
    assert_sql(query, expected_sql, expected_params)









def test_where_clause():
    users = Table("users", "u")
    u_id = Column("id", users)
    u_name = Column("name", users)
    u_age = Column("age", users)
    u_address = Column("address", users)
    u_created = Column("created_at", users)

    # WHERE clause
    where_expr = And(
        Or(
            And(
                Gt(u_age, Value(21, inline=False)),
                Like(u_name, Value("J%", inline=False))
            ),
            Not(IsNull(u_address))
        ),
        In(u_id, [1, 2, 3]),
        Between(u_created, Value("2023-01-01", inline=False), Value("2023-12-31", inline=False))
    )

    # Build the query
    query = SelectQuery()
    query.columns = [u_id, u_name, u_age, u_address, u_created]
    query.from_table = users
    query.where &= where_expr

    # Compile
    compiler = Compiler()
    sql, params = compiler.compile(query)

    # Print for debugging (optional)
    
    # Basic assertions
    assert "WHERE" in sql
    assert "AND" in sql
    assert "OR" in sql
    assert "LIKE" in sql
    assert "BETWEEN" in sql
    assert "IN ($" in sql or "IN (" in sql  # Depending on your param rendering
    assert len(params) == 7
    assert params == [21, "J%", 1, 2, 3, "2023-01-01", "2023-12-31"][:len(params)]





def test_insert_query():
    users = Table("users", "u")
    u_id = Column("id", users)
    u_name = Column("name", users)
    u_email = Column("email", users)

    q = InsertQuery(users)
    q.columns = [u_id, u_name, u_email]
    q.values = [
        [Value(1, inline=False), Value("Alice", inline=False), Value("alice@example.com", inline=False)],
        [Value(2, inline=False), Value("Bob", inline=False), Value("bob@example.com", inline=False)]
    ]
    q.returning = [u_id]

    compiler = Compiler()
    sql, params = compiler.compile(q)

    print(sql)
    print("Params:", params)

    assert sql.startswith("INSERT INTO users")
    assert "VALUES" in sql
    assert "RETURNING" in sql
    assert params == [1, "Alice", "alice@example.com", 2, "Bob", "bob@example.com"]


def test_update_query():
    users = Table("users", "u")
    u_id = Column("id", users)
    u_name = Column("name", users)
    u_email = Column("email", users)

    q = UpdateQuery(users)
    q.set(
        [
            (u_name, Value("Alice Updated", inline=False)),
            (u_email, Value("alice@newmail.com", inline=False))
        ]
    )
    q.where &= Eq(u_id, Value(1, inline=False))
    q.returning = [u_id]

    compiler = Compiler()
    sql, params = compiler.compile(q)

    print(sql)
    print("Params:", params)

    assert sql.startswith("UPDATE users")
    assert "SET" in sql
    assert "WHERE" in sql
    assert "RETURNING" in sql
    assert params == ["Alice Updated", "alice@newmail.com", 1]



def test_delete_query():
    users = Table("users", "u")
    u_id = Column("id", users)
    u_name = Column("name", users)

    q = DeleteQuery(users)
    q.where &= Eq(u_name, Value("Bob", inline=False))
    q.returning = [u_id]

    compiler = Compiler()
    sql, params = compiler.compile(q)

    print(sql)
    print("Params:", params)

    assert sql.startswith("DELETE FROM users")
    assert "WHERE" in sql
    assert "RETURNING" in sql
    assert params == ["Bob"]









def test_sub_queries():
    posts = Table("posts", "p")
    p_user_id = Column("user_id", posts)

    # Subquery: SELECT user_id, COUNT(*) AS post_count FROM posts GROUP BY user_id
    inner_query = SelectQuery()
    inner_query.columns = [
        Column("user_id", posts),
        Function("COUNT", Value("*"), alias="post_count")
    ]
    inner_query.from_table = posts
    inner_query.group_by = [Column("user_id", posts)]

    subquery = Subquery(inner_query, alias="pc")

    # Outer query: SELECT p.user_id, pc.post_count ...
    outer_query = SelectQuery()
    outer_query.columns = [
        Column("user_id", posts),
        Column("post_count", subquery)
    ]
    outer_query.from_table = subquery
    outer_query.join(posts, Eq(Column("user_id", posts), Column("user_id", subquery)))

    expected_sql = """
    SELECT p.user_id, pc.post_count
    FROM (
        SELECT p.user_id, COUNT(*) AS post_count
        FROM posts AS p
        GROUP BY p.user_id
    ) AS pc
    LEFT JOIN posts AS p ON (p.user_id = pc.user_id)
    """



    assert_sql(outer_query, expected_sql, [])
    
    
    
    
    
    
    
def test_cte_support():
    posts = Table("posts", "p")
    users = Table("users", "u")

    p_user_id = Column("user_id", posts)
    p_created = Column("created_at", posts)
    u_id = Column("id", users)
    u_name = Column("name", users)

    # CTE query: recent posts
    recent_posts = SelectQuery()
    recent_posts.columns = [p_user_id]
    recent_posts.from_table = posts
    recent_posts.where &= Gt(p_created, Value("2024-01-01", inline=False))

    # Outer query: users + recent posts
    cte_alias = "recent_posts"
    cte_table = Table(cte_alias, "rp")

    rp_user_id = Column("user_id", cte_table)

    query = SelectQuery()
    query.ctes = [(cte_alias, recent_posts)]
    query.columns = [u_id, u_name]
    query.from_table = users
    query.join(cte_table, Eq(u_id, rp_user_id))

    expected_sql = """
    WITH recent_posts AS (
        SELECT p.user_id
        FROM posts AS p
        WHERE (p.created_at > $1)
    )
    SELECT u.id, u.name
    FROM users AS u
    LEFT JOIN recent_posts AS rp ON (u.id = rp.user_id)
    """
    expected_params = ["2024-01-01"]

    assert_sql(query, expected_sql, expected_params)





def test_multiple_ctes():
    posts = Table("posts", "p")
    users = Table("users", "u")
    user_counts_cte = Table("user_counts", "uc")
    top_users_cte = Table("top_users", "t")

    # Columns
    p_user_id = Column("user_id", posts)
    u_id = Column("id", users)
    u_name = Column("name", users)
    uc_user_id = Column("user_id", user_counts_cte)
    uc_post_count = Column("post_count", user_counts_cte)
    t_user_id = Column("user_id", top_users_cte)

    # First CTE: user_counts
    user_counts = SelectQuery()
    user_counts.columns = [
        p_user_id,
        Function("COUNT", Value("*"), alias="post_count")
    ]
    user_counts.from_table = posts
    user_counts.group_by = [p_user_id]

    # Second CTE: top_users
    top_users = SelectQuery()
    top_users.columns = [uc_user_id]
    top_users.from_table = user_counts_cte
    top_users.where &= Gt(uc_post_count, Value(10, inline=False))

    # Final query
    query = SelectQuery()
    query.ctes = [("user_counts", user_counts), ("top_users", top_users)]
    query.columns = [u_id, u_name]
    query.from_table = users
    query.join(top_users_cte, Eq(u_id, t_user_id))

    expected_sql = """
    WITH user_counts AS (
        SELECT p.user_id, COUNT(*) AS post_count
        FROM posts AS p
        GROUP BY p.user_id
    ),
    top_users AS (
        SELECT uc.user_id
        FROM user_counts AS uc
        WHERE (uc.post_count > $1)
    )
    SELECT u.id, u.name
    FROM users AS u
    LEFT JOIN top_users AS t ON (u.id = t.user_id)
    """
    expected_params = [10]

    assert_sql(query, expected_sql, expected_params)



class Post(Model):
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    updated_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    topic: str = ModelField()
    content: str = ModelField()
    views: int = ModelField()
    author: str = ModelField()
    uuid: str = ModelField()




def test_basic_eq():
    qs = PgSelectQuerySet(Post).select("*").where(lambda x: x.topic == "animals")
    sql = qs.render()
    assert "WHERE (p.topic = $1)" in sql
    assert qs._params == ["animals"]

def test_combined_and_or():
    qs = PgSelectQuerySet(Post).select("*").where(
        lambda x: (x.topic == "a") & (x.views > 5) | (x.author == "admin")
    )
    sql = qs.render()
    assert "WHERE (((p.topic = $1) AND (p.views > $2)) OR (p.author = $3))" in sql
    assert qs._params == ["a", 5, "admin"]

def test_negation():
    qs = PgSelectQuerySet(Post).select("*").where(lambda x: ~(x.topic == "a"))
    sql = qs.render()
    assert "WHERE (NOT (p.topic = $1))" in sql
    assert qs._params == ["a"]
