# reset_faiss.py
import psycopg2
from db import get_connection, release_connection   # use your existing db.py

conn = get_connection()
cur = conn.cursor()

try:
    cur.execute("TRUNCATE TABLE resume_faiss_map RESTART IDENTITY;")
    conn.commit()
    print("✅ resume_faiss_map table cleared successfully!")
except Exception as e:
    print("❌ Error:", e)
finally:
    cur.close()
    release_connection(conn)