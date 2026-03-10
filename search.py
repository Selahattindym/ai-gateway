from sentence_transformers import SentenceTransformer
import psycopg2

model = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(
    host="127.0.0.1",
    port=5433,
    database="aidb",
    user="aiuser",
    password="aipass"
)

cur = conn.cursor()

query = input("Soru: ")

query_vector = model.encode(query).tolist()
query_vector = "[" + ",".join(map(str, query_vector)) + "]"

cur.execute("""
SELECT content
FROM documents
ORDER BY embedding <-> %s::vector
LIMIT 3;
""", (query_vector,))

results = cur.fetchall()

for r in results:
    print("-", r[0])
