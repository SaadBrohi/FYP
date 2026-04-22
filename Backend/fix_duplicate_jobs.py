# cleanup_duplicate_jobs.py
from db import get_connection, release_connection

def cleanup_duplicates():
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        print("Checking for duplicate job titles...")

        # Find duplicate titles
        cur.execute("""
            SELECT title, COUNT(*) as count 
            FROM jobs 
            GROUP BY title 
            HAVING COUNT(*) > 1;
        """)
        duplicates = cur.fetchall()

        if not duplicates:
            print("✅ No duplicate titles found.")
        else:
            print(f"Found {len(duplicates)} titles with duplicates:")
            for title, count in duplicates:
                print(f"   → '{title}' appears {count} times")

            # Keep only the one with smallest job_id for each duplicate title
            cur.execute("""
                DELETE FROM jobs a
                WHERE a.job_id NOT IN (
                    SELECT MIN(job_id)
                    FROM jobs b
                    WHERE a.title = b.title
                );
            """)
            deleted = cur.rowcount
            conn.commit()
            print(f"✅ Cleaned up {deleted} duplicate job entries.")

        # Now add the unique constraint
        cur.execute("""
            ALTER TABLE jobs 
            ADD CONSTRAINT unique_job_title UNIQUE (title);
        """)
        conn.commit()
        print("✅ Unique constraint on 'title' added successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if conn:
            release_connection(conn)

if __name__ == "__main__":
    cleanup_duplicates()