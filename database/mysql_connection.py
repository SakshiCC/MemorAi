from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import mysql.connector
from mysql.connector import Error
from mysql.connector.connection import MySQLConnection
from mysql.connector.pooling import MySQLConnectionPool

from config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_POOL_NAME,
    DB_POOL_SIZE,
    DB_PORT,
    DB_USER,
)
from utils.logger import get_logger

logger = get_logger("database.connection", "database.log")

_pool: MySQLConnectionPool | None = None


def _server_connection() -> MySQLConnection:
    """Connect without selecting a database, used for first-time creation."""
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        autocommit=True,
    ) # type: ignore


def ensure_database_exists() -> None:
    connection = None
    cursor = None

    try:
        connection = _server_connection()
        cursor = connection.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        logger.info("Database '%s' is available.", DB_NAME)
    except Error:
        logger.exception("Could not create/check database '%s'.", DB_NAME)
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_pool() -> MySQLConnectionPool:
    global _pool

    if _pool is None:
        ensure_database_exists()
        _pool = MySQLConnectionPool(
            pool_name=DB_POOL_NAME,
            pool_size=DB_POOL_SIZE,
            pool_reset_session=True,
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset="utf8mb4",
            collation="utf8mb4_unicode_ci",
        )
        logger.info("MySQL connection pool initialized.")

    return _pool


def get_connection() -> MySQLConnection:
    return get_pool().get_connection() # type: ignore


@contextmanager
def database_connection() -> Iterator[MySQLConnection]:
    connection = get_connection()

    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    finally:
        if connection.is_connected():
            connection.close()
