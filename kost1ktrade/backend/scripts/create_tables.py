import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.session import engine
from src.database.models import Base

def main():
    """
    Creates all database tables defined in the SQLAlchemy models.
    Can also drop all tables first if --recreate flag is used.
    """
    parser = argparse.ArgumentParser(description="Manage database tables.")
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Drop all tables before creating them. WARNING: This will delete all existing data.'
    )
    args = parser.parse_args()

    if args.recreate:
        print("Dropping all tables...")
        try:
            Base.metadata.drop_all(bind=engine)
            print("Tables dropped successfully.")
        except Exception as e:
            print(f"An error occurred while dropping tables: {e}")
            return

    print("Connecting to the database and creating tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully.")
    except Exception as e:
        print(f"An error occurred while creating tables: {e}")
        print("Please check your DATABASE_URL in the .env file and ensure the database server is running.")

if __name__ == "__main__":
    main()
