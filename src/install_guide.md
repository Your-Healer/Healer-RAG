# PostgreSQL with pgvector Installation Guide

## Installation Steps for Windows

### 1. Install PostgreSQL

1. Download PostgreSQL installer from https://www.postgresql.org/download/windows/
2. Run the installer and follow the setup wizard
3. Select at least these components:
   - PostgreSQL Server
   - pgAdmin 4 (for GUI management)
   - Command Line Tools
4. Choose a password for the postgres user and remember it
5. Complete the installation

### 2. Add PostgreSQL to your PATH

After installation, add the PostgreSQL bin directory to your system PATH:

1. Right-click on "This PC" or "My Computer" and select "Properties"
2. Click on "Advanced system settings"
3. Click on the "Environment Variables" button
4. Under "System variables", find the "Path" variable, select it and click "Edit"
5. Click "New" and add the path to your PostgreSQL bin directory (typically `C:\Program Files\PostgreSQL\15\bin`)
6. Click "OK" on all dialogs to save

### 3. Install pgvector Extension

1. Open Command Prompt as Administrator
2. Install the PostgreSQL extension build tools:

   ```
   pip install psycopg2-binary
   ```

3. Clone and build pgvector:

   ```
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   SET PG_CONFIG=C:\Program Files\PostgreSQL\15\bin\pg_config.exe
   cmake -B build .
   cmake --build build --config Release
   cmake --install build
   ```

4. Create a database and enable pgvector:
   ```
   psql -U postgres
   CREATE DATABASE healer_vector_db;
   \c healer_vector_db
   CREATE EXTENSION vector;
   \q
   ```

### 4. Set Environment Variables

Create a `.env` file in your project root with:

```
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DATABASE=healer_vector_db
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_password_here
```

### 5. Install Required Python Packages

```
pip install psycopg2-binary langchain-postgres pgvector
```

## Installation Steps for Linux/macOS

### 1. Install PostgreSQL

For Ubuntu/Debian:

```
sudo apt update
sudo apt install postgresql postgresql-contrib build-essential
```

For macOS with Homebrew:

```
brew install postgresql
```

### 2. Install pgvector

```
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 3. Create Database and Enable Extension

```
sudo -u postgres psql
CREATE DATABASE healer_vector_db;
\c healer_vector_db
CREATE EXTENSION vector;
\q
```

### 4. Set Environment Variables

Create a `.env` file in your project root with:

```
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DATABASE=healer_vector_db
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_password_here
```

### 5. Install Required Python Packages

```
pip install psycopg2-binary langchain-postgres pgvector
```

## Verifying Installation

Run the check_pgvector.py script to verify that everything is set up correctly:

```
python src/check_pgvector.py
```

If successful, you should see a message confirming the connection.
