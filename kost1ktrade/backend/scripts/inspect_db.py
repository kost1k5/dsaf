import os
import sys
from sqlalchemy import create_engine, inspect, func
from sqlalchemy.orm import sessionmaker

# Adjust path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.session import DATABASE_URL, SessionLocal
from src.database.models import Base # Import Base to get metadata

def inspect_database():
    """
    Connects to the database, lists all tables, and prints the row count for each.
    """
    engine = create_engine(DATABASE_URL)
    db = SessionLocal()
    inspector = inspect(engine)

    try:
        # Get table names from the metadata of the Base class
        table_names = Base.metadata.tables.keys()

        print("--- Database Inspection Results ---")
        if not table_names:
            print("No tables found in the database metadata.")
            return

        for table_name in sorted(table_names):
            # The model class is needed for the query, we can get it from the table object
            table_object = Base.metadata.tables[table_name]
            # A bit of a workaround to query without importing all models directly
            count = db.query(func.count()).select_from(table_object).scalar()
            print(f"Table: {table_name:<20} | Rows: {count}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        db.close()
        print("--- End of Report ---")

if __name__ == "__main__":
    inspect_database()
