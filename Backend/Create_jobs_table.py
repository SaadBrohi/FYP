# create_jobs_table.py
import psycopg2
from db import get_connection, release_connection

def create_jobs_table():
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id SERIAL PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                skills TEXT[],
                required_traits JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Add index for better performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(title);")

        conn.commit()
        print("✅ Jobs table created successfully!")

    except Exception as e:
        print(f"❌ Error creating jobs table: {e}")
    finally:
        if conn:
            release_connection(conn)

if __name__ == "__main__":
    create_jobs_table()