# neo4j_connection.py

from neo4j import AsyncGraphDatabase
import os


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")




from neo4j import AsyncGraphDatabase, RoutingControl

class Neo4jConnectionManager:
    _driver = None

    @classmethod
    def init(cls, uri: str, user: str, password: str):
        if cls._driver is None:
            cls._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    @classmethod
    def get_driver(cls):
        if cls._driver is None:
            cls.init(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            # raise RuntimeError("Neo4j driver not initialized! Call Neo4jConnectionManager.init() first.")
        return cls._driver

    @classmethod
    async def close(cls):
        if cls._driver is not None:
            await cls._driver.close()
            cls._driver = None

    # READ ALL RECORDS (returns list of dicts)
    @classmethod
    async def execute_read(cls, cypher: str, params: dict | None = None, database: str | None = None):
        driver = cls.get_driver()
        async with driver.session(database=database) as session:
            result = await session.run(cypher, parameters=params or {})
            # The recommended way to consume all results async:
            records = []
            async for record in result:
                records.append(record.data())
            return records

    # READ ONE RECORD (returns dict or None)
    @classmethod
    async def execute_read_single(cls, cypher: str, params: dict | None = None, database: str | None = None):
        driver = cls.get_driver()
        async with driver.session(database=database) as session:
            result = await session.run(cypher, parameters=params or {})
            record = await result.single()
            return record.data() if record else None

    # WRITE (returns list of dicts)
    @classmethod
    async def execute_write(cls, cypher: str, params: dict | None = None, database: str | None = None):
        driver = cls.get_driver()
        async with driver.session(database=database) as session:
            result = await session.run(cypher, parameters=params or {})
            # Most write queries don't return lots of results, but for completeness:
            records = []
            async for record in result:
                records.append(record.data())
            return records

    # WRITE ONE (returns dict or None)
    @classmethod
    async def execute_write_single(cls, cypher: str, params: dict | None = None, database: str | None = None):
        driver = cls.get_driver()
        async with driver.session(database=database) as session:
            result = await session.run(cypher, parameters=params or {})
            record = await result.single()
            return record.data() if record else None

    # Bonus: Use the newer driver.execute_query if you want ultra-short queries:
    @classmethod
    async def execute_query(cls, cypher: str, params: dict = None, database: str = None, routing=RoutingControl.WRITE, single=False):
        driver = cls.get_driver()
        # Neo4j 5.x+ only: this will use managed sessions/transactions for you
        result = await driver.execute_query(
            cypher,
            parameters_=params or {},
            routing_=routing,
            database_=database,
            result_transformer_=(lambda res: res.single() if single else res.data())
        )
        if single:
            return result.data() if result else None
        return result



# class Neo4jConnectionManager:
#     _driver = None

#     @classmethod
#     def init(cls, uri: str | None = None, user: str | None = None, password: str | None = None):
#         if uri is None:
#             uri = NEO4J_URI
#         if user is None:
#             user = NEO4J_USER
#         if password is None:
#             password = NEO4J_PASSWORD
#         if cls._driver is None:
#             cls._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

#     @classmethod
#     def get_driver(cls):
#         if not cls._driver:
#             cls.init()
#             # raise ValueError("Neo4j driver not initialized")
#         return cls._driver

    
#     @classmethod
#     async def execute_write(cls, cypher: str, params: dict | None = None):
#         driver = cls.get_driver()
#         async with driver.session() as session:
#             result = await session.run(cypher, **(params or {}))
#             r = await result.consume()
#             return result
        
#     @classmethod
#     async def execute_read(cls, cypher: str, params: dict | None = None):
#         driver = cls.get_driver()
#         async with driver.session() as session:
#             result = await session.run(cypher, **(params or {}))
#             r = await result.consume()
#             return result
        
        
   