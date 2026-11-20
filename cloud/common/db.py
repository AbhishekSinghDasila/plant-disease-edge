"""Small Postgres helper (psycopg 3, connection pool, JSON in/out)."""
import json
import logging
import time

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from common import settings

log = logging.getLogger("db")
_pool = None


def pool():
    global _pool
    if _pool is None:
        for _ in range(30):  # wait for postgres to accept connections
            try:
                _pool = ConnectionPool(settings.DATABASE_URL, min_size=1, max_size=5,
                                       kwargs={"row_factory": dict_row, "autocommit": True}, open=True)
                _pool.wait(timeout=5)
                break
            except Exception as e:
                log.warning("postgres not ready (%s)", e)
                _pool = None
                time.sleep(2)
    return _pool


def execute(sql, params=()):
    with pool().connection() as c:
        c.execute(sql, params)


def query(sql, params=()):
    with pool().connection() as c:
        return c.execute(sql, params).fetchall()


def jsonb(d):
    return Jsonb(json.loads(json.dumps(d, default=str)))
