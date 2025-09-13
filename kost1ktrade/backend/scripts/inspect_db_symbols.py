import sys
import os
from sqlalchemy.orm import Session
from sqlalchemy import distinct

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.session import SessionLocal
from src.database.models import Candle

def inspect_db_symbols():
    """
    Connects to the database and prints all unique symbol/interval pairs
    found in the 'candles' table to help debug data loading issues.
    """
    db: Session = SessionLocal()
    try:
        print("--- Inspecting Database for Available Symbols and Intervals ---")

        # Query for distinct symbol and interval pairs
        results = db.query(Candle.symbol, Candle.interval).distinct().all()

        if not results:
            print("\nCRITICAL: No candle data found in the database at all.")
            return

        print("\nFound the following symbol/interval pairs in the database:")
        for symbol, interval in results:
            print(f"  - Symbol: '{symbol}', Interval: '{interval}'")

        print("\n--- Inspection Complete ---")

    except Exception as e:
        print(f"\nAn error occurred while connecting to or querying the database: {e}")
    finally:
        db.close()

if __name__ == '__main__':
    inspect_db_symbols()
