import os
import sys
import argparse
from dotenv import load_dotenv
import psycopg2

def check_pg_connection(connection_params=None):
    """
    Simple PostgreSQL connection check that doesn't depend on langchain.
    
    Args:
        connection_params: Dictionary with connection parameters or None to use env vars
        
    Returns:
        tuple: (success, message)
    """
    # Use environment variables if params not provided
    if not connection_params:
        connection_params = {
            "host": os.environ.get("PGVECTOR_HOST", "localhost"),
            "port": os.environ.get("PGVECTOR_PORT", "5432"),
            "database": os.environ.get("PGVECTOR_DATABASE", "healer_vector_db"),
            "user": os.environ.get("PGVECTOR_USER", "postgres"),
            "password": os.environ.get("PGVECTOR_PASSWORD", ""),
        }
    
    try:
        # Connect to the database directly with psycopg2
        conn = psycopg2.connect(
            host=connection_params["host"],
            port=connection_params["port"],
            database=connection_params["database"],
            user=connection_params["user"],
            password=connection_params["password"]
        )
        
        # Create a cursor and check for pgvector extension
        cursor = conn.cursor()
        
        # Check PostgreSQL version
        cursor.execute("SELECT version();")
        pg_version = cursor.fetchone()[0]
        
        # Check if pgvector extension is installed
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        pgvector_installed = cursor.fetchone() is not None
        
        # Check if collection exists (if specified)
        collection_exists = False
        collection_name = connection_params.get("collection_name")
        if collection_name:
            cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s);",
                (f"langchain_{collection_name}",)
            )
            collection_exists = cursor.fetchone()[0]
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        # Return status
        if pgvector_installed:
            msg = f"✅ Connection successful. PostgreSQL: {pg_version.split(',')[0]}"
            msg += "\n✅ pgvector extension is installed."
            
            if collection_name:
                if collection_exists:
                    msg += f"\n✅ Collection '{collection_name}' exists."
                else:
                    msg += f"\n❌ Collection '{collection_name}' does not exist."
            
            return True, msg
        else:
            msg = f"✅ Connection successful. PostgreSQL: {pg_version.split(',')[0]}"
            msg += "\n❌ pgvector extension is NOT installed."
            return False, msg
            
    except Exception as e:
        return False, f"❌ Connection failed: {str(e)}"

def parse_connection_string(connection_string):
    """Parse a PostgreSQL connection string into component parts."""
    try:
        # Remove postgresql:// prefix
        if connection_string.startswith("postgresql://"):
            connection_string = connection_string[len("postgresql://"):]
        
        # Parse user:pass@host:port/dbname
        user_pass, host_port_db = connection_string.split("@")
        
        if ":" in user_pass:
            user, password = user_pass.split(":", 1)
        else:
            user = user_pass
            password = ""
        
        host_port, database = host_port_db.split("/", 1)
        
        if ":" in host_port:
            host, port = host_port.split(":")
        else:
            host = host_port
            port = "5432"
        
        return {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password
        }
    except Exception as e:
        print(f"Error parsing connection string: {str(e)}")
        return None

def main():
    """Check PostgreSQL and pgvector health."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Check PostgreSQL and pgvector connection health")
    parser.add_argument("--host", help="PostgreSQL host")
    parser.add_argument("--port", help="PostgreSQL port")
    parser.add_argument("--database", help="PostgreSQL database name")
    parser.add_argument("--user", help="PostgreSQL username")
    parser.add_argument("--password", help="PostgreSQL password")
    parser.add_argument("--connection-string", help="Full PostgreSQL connection string")
    parser.add_argument("--collection", help="Collection name to check", default="healer_medical_docs")
    parser.add_argument("--check-collection", action="store_true", help="Check if collection exists")
    
    args = parser.parse_args()
    
    # Determine connection parameters
    connection_params = None
    if args.connection_string:
        connection_params = parse_connection_string(args.connection_string)
    elif args.host or args.database or args.user:
        connection_params = {
            "host": args.host or os.environ.get("PGVECTOR_HOST", "localhost"),
            "port": args.port or os.environ.get("PGVECTOR_PORT", "5432"),
            "database": args.database or os.environ.get("PGVECTOR_DATABASE", "healer_vector_db"),
            "user": args.user or os.environ.get("PGVECTOR_USER", "postgres"),
            "password": args.password or os.environ.get("PGVECTOR_PASSWORD", ""),
        }
    
    # Add collection name if checking collection
    if args.check_collection:
        if connection_params is None:
            connection_params = {}
        connection_params["collection_name"] = args.collection
    
    print("Checking PostgreSQL connection...")
    print("Note: This will only check the database connection, not the LangChain PGVector integration.")
    print("For details on the API parameters, visit: https://python.langchain.com/docs/integrations/vectorstores/pgvector")
    
    success, message = check_pg_connection(connection_params)
    
    print(message)
    
    if success:
        print("\nPostgreSQL connection is healthy.")
        sys.exit(0)
    else:
        print("\nIf using Windows, ensure PostgreSQL is installed and try one of these solutions:")
        print("1. Install PostgreSQL with the official installer from postgresql.org")
        print("2. Add PostgreSQL bin directory to your PATH environment variable")
        print("3. Install psycopg2-binary package: pip install psycopg2-binary")
        sys.exit(1)

if __name__ == "__main__":
    main()
