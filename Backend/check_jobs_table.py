# check_jobs_table.py
from db import get_connection, release_connection

def check_jobs():
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'jobs'
            );
        """)
        table_exists = cur.fetchone()[0]
        print(f"✅ Table 'jobs' exists: {table_exists}")

        if table_exists:
            # Show table structure
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'jobs'
                ORDER BY ordinal_position;
            """)
            columns = cur.fetchall()
            print("\nTable Structure:")
            for col in columns:
                print(f"   {col[0]:<20} → {col[1]}")

            # Show how many jobs exist
            cur.execute("SELECT COUNT(*) FROM jobs;")
            count = cur.fetchone()[0]
            print(f"\nTotal jobs in table: {count}")

            # Show sample data (if any)
            cur.execute("SELECT job_id, title, skills FROM jobs LIMIT 3;")
            rows = cur.fetchall()
            if rows:
                print("\nSample Jobs:")
                for row in rows:
                    print(f"   ID: {row[0]}, Title: {row[1]}, Skills: {row[2]}")
            else:
                print("\nNo jobs found in the table yet.")

        cur.close()

    except Exception as e:
        print(f"❌ Error checking jobs table: {e}")
    finally:
        if conn:
            release_connection(conn)

if __name__ == "__main__":
    check_jobs()