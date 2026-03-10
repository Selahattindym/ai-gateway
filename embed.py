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

text = "GitLab CI/CD pipeline automates build and deployment"

vector = model.encode(text).tolist()

cur.execute(
    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
    (text, vector)
)

conn.commit()

print("Saved to Vector DB")

