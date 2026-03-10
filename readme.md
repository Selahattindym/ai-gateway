# Vector DB + Ollama Mistral + LoRA Experiment

Bu proje Ubuntu sunucu üzerinde **Ollama ile Mistral modelini çalıştırarak**, metinleri **embedding vektörlerine dönüştürüp PostgreSQL (pgvector)** içerisinde saklama ve **LoRA davranış özelleştirmesi denemeleri** yapmak amacıyla oluşturulmuştur.

---

# Sistem Mimarisi

```
Text
 ↓
Embedding Model
 ↓
Vector
 ↓
PostgreSQL (pgvector)
 ↓
Similarity Search
 ↓
LLM (Mistral / Ollama)
```

---

# Gereksinimler

* Ubuntu Server
* Python 3
* PostgreSQL
* pgvector
* Ollama
* Sentence Transformers
* psycopg2

---

# 1 Ubuntu Güncelleme

```bash
sudo apt update
sudo apt upgrade -y
```

---

# 2 Python ve Pip Kurulumu

```bash
sudo apt install python3-pip -y
```

kontrol

```bash
python3 --version
pip3 --version
```

---

# 3 PostgreSQL Kurulumu

```bash
sudo apt install postgresql postgresql-contrib -y
```

servis kontrol

```bash
sudo systemctl status postgresql
```

postgres kullanıcısına geç

```bash
sudo -u postgres psql
```

---

# 4 Veritabanı ve Kullanıcı Oluşturma

```sql
CREATE DATABASE aidb;

CREATE USER aiuser WITH PASSWORD 'aipass';

ALTER ROLE aiuser SET client_encoding TO 'utf8';
ALTER ROLE aiuser SET default_transaction_isolation TO 'read committed';
ALTER ROLE aiuser SET timezone TO 'UTC';

GRANT ALL PRIVILEGES ON DATABASE aidb TO aiuser;
```

---

# 5 pgvector Kurulumu

```bash
sudo apt install postgresql-15-pgvector
```

psql içine gir

```bash
sudo -u postgres psql -d aidb
```

extension oluştur

```sql
CREATE EXTENSION vector;
```

---

# 6 Vector Tablo Oluşturma

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(384)
);
```

---

# 7 Python Kütüphaneleri

```bash
pip install sentence-transformers
pip install psycopg2-binary
```

---

# 8 Embedding Script (embed.py)

```python
from sentence_transformers import SentenceTransformer
import psycopg2

model = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(
    host="127.0.0.1",
    port="5432",
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

cur.close()
conn.close()
```

çalıştır

```bash
python3 embed.py
```

---

# 9 Similarity Search

```sql
SELECT content
FROM documents
ORDER BY embedding <-> '[...]'
LIMIT 5;
```

---

# 10 Ollama Kurulumu

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

kontrol

```bash
ollama --version
```

---

# 11 Mistral Modeli Çekme

```bash
ollama pull mistral
```

model çalıştırma

```bash
ollama run mistral
```

---

# 12 Modelfile Oluşturma

```bash
nano Modelfile
```

içerik

```
FROM mistral

SYSTEM You are an expert SEO specialist.
You analyze content structure, keywords and ranking potential.
Always give professional SEO suggestions.
```

---

# 13 Custom Model Oluşturma

```bash
ollama create seo-mistral -f Modelfile
```

çalıştırma

```bash
ollama run seo-mistral
```

---

# 14 LoRA Mantığı

LoRA (Low Rank Adaptation) modeli tamamen yeniden eğitmek yerine **ek düşük boyutlu ağırlıklar kullanarak modeli özelleştirmeye yarayan bir yöntemdir.**

Genel süreç:

```
Dataset
 ↓
LoRA Training
 ↓
LoRA Weights
 ↓
Merge
 ↓
Final Model
```

LoRA ağırlıkları ana model ile **merge edilerek tek model gibi kullanılabilir.**

---

# 15 Kullanılan Teknolojiler

* Ollama
* Mistral LLM
* PostgreSQL
* pgvector
* Python
* Sentence Transformers
* psycopg2

---

# Amaç

Bu proje aşağıdaki konuları öğrenmek amacıyla yapılmıştır:

* Vector Database mantığı
* Embedding üretimi
* Similarity Search
* LLM + Vector DB entegrasyonu
* LoRA ile model davranışı özelleştirme

---

# Not

Bu proje **GitLab Duo / AI Assistant mimarilerinin temel mantığını anlamak için yapılan deneysel bir çalışmadır.**
