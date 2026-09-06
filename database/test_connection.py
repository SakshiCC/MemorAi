from database.db_manager import initialize_database
from database.mysql_connection import get_connection


def main() -> None:
    initialize_database()

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT DATABASE(), VERSION()")
        database_name, version = cursor.fetchone()

        print("MySQL connection successful.")
        print(f"Database: {database_name}")
        print(f"MySQL version: {version}")
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
