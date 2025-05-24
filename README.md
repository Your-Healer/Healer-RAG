# Healer RAG - Medical Assistant

## Vector Database Options

The system now supports two vector database options:

### 1. Chroma (Default)

Chroma is a local vector database that stores embeddings in memory or on disk. It's easy to use and requires no additional setup, making it ideal for development and testing.

### 2. PostgreSQL with pgvector

For production environments or larger datasets, we've added support for PostgreSQL with the pgvector extension. This provides:

- Scalable vector storage
- SQL-based querying
- Persistence across sessions
- Support for larger document collections
- Better concurrency and multi-user support

## Setting Up pgvector

1. Install PostgreSQL and the pgvector extension:

```bash
# For Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

2. Create a database and enable the pgvector extension:

```sql
CREATE DATABASE healer_vector_db;
\c healer_vector_db
CREATE EXTENSION vector;
```

3. Configure database access in your `.env` file:

```
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DATABASE=healer_vector_db
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_password_here
```

4. Run the application with pgvector as the vector store:

```bash
python src/main.py --vector-store pgvector
```

## Switching Between Vector Stores

You can choose which vector store to use at runtime:

```bash
# Use Chroma (default)
python src/main.py --vector-store chroma

# Use PostgreSQL with pgvector
python src/main.py --vector-store pgvector
```

Or set the default in your `.env` file:

```
DEFAULT_VECTOR_STORE=pgvector
```

## Benefits of Using pgvector

- **Scalability**: Efficiently handles millions of vectors
- **Persistence**: Data remains available across restarts
- **Performance**: Optimized for high-volume retrieval operations
- **Integration**: Works with existing PostgreSQL infrastructure
- **Production-ready**: Suitable for deployment environments
