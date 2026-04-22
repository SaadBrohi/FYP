# update_jobs_table.py
from db import get_connection, release_connection

def update_jobs_table():
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        print("Adding missing columns to jobs table...")

        # Add required_traits if it doesn't exist
        cur.execute("""
            ALTER TABLE jobs 
            ADD COLUMN IF NOT EXISTS required_traits JSONB DEFAULT '{}';
        """)

        # Add timestamps if they don't exist
        cur.execute("""
            ALTER TABLE jobs 
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
        """)

        cur.execute("""
            ALTER TABLE jobs 
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
        """)

        conn.commit()
        print("✅ Jobs table updated successfully with all required columns!")

        # Verify the new structure
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'jobs'
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        print("\nUpdated Table Structure:")
        for col in columns:
            print(f"   {col[0]:<20} → {col[1]}")

    except Exception as e:
        print(f"❌ Error updating jobs table: {e}")
    finally:
        if conn:
            release_connection(conn)

if __name__ == "__main__":
    update_jobs_table()