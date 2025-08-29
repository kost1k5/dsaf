import sys
import os

# Add the project root to the Python path to allow for absolute imports
# This allows us to run this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.session import engine
from src.database.models import Base

def main():
    """
    Creates all database tables defined in the SQLAlchemy models.
    """
    print("Connecting to the database and creating tables...")
    try:
        # The `checkfirst=True` argument prevents errors if the tables already exist.
        Base.metadata.create_all(bind=engine, checkfirst=True)
        print("Tables created successfully (if they didn't exist).")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your DATABASE_URL in the .env file and ensure the database server is running.")

if __name__ == "__main__":
    main()
