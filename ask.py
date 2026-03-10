from sentence_transformers import SentenceTransformer
import psycopg2
import ollama

model = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(
    host="127.0.0.1",
    port=5433,
    database="aidb",
    user="aiuser",
    password="aipass"
)

cur = conn.cursor()

question = input("Soru: ")

query_vector = model.encode(question).tolist()

cur.execute("""
SELECT content
FROM documents
ORDER BY embedding <-> %s::vector
LIMIT 3
""", (query_vector,))

results = cur.fetchall()

context = "\n".join([row[0] for row in results])

prompt = f"""
Bu bilgilerden yararlanarak soruya cevap ver.

Bilgi:
{context}

Soru:
{question}
"""

response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": prompt}]
)

print("\nCevap:\n")
print(response["message"]["content"])

