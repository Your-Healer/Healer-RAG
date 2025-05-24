import os
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def check_chroma_db(db_path="./chroma_langchain_db"):
    """
    Check if a Chroma database exists and report information about it.
    
    Args:
        db_path (str): Path to the Chroma database directory
    """
    print(f"Checking Chroma database at: {os.path.abspath(db_path)}")
    
    # Check if directory exists
    if not os.path.exists(db_path):
        print("❌ Database directory does not exist.")
        return False
    
    # Check if it's a directory
    if not os.path.isdir(db_path):
        print("❌ Path exists but is not a directory.")
        return False
    
    # Check if directory has contents
    contents = os.listdir(db_path)
    if not contents:
        print("❌ Database directory exists but is empty.")
        return False
    
    print(f"✅ Database directory exists with {len(contents)} items:")
    for item in contents:
        item_path = os.path.join(db_path, item)
        if os.path.isdir(item_path):
            subitem_count = len(os.listdir(item_path))
            print(f"  - {item}/ (directory with {subitem_count} items)")
        else:
            size = os.path.getsize(item_path) / 1024  # Size in KB
            print(f"  - {item} ({size:.2f} KB)")
    
    # Try to connect to the database
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory=db_path
        )
        collection_count = db._collection.count()
        print(f"✅ Successfully connected to database. Contains {collection_count} documents.")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to database: {str(e)}")
        return False

def reset_database(db_path="./chroma_langchain_db", confirm=True):
    """
    Delete the database to start fresh.
    
    Args:
        db_path (str): Path to the database directory
        confirm (bool): Whether to ask for confirmation
    """
    if not os.path.exists(db_path):
        print(f"Database at {db_path} does not exist. Nothing to reset.")
        return
    
    if confirm:
        response = input(f"Are you sure you want to delete the database at {db_path}? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    try:
        shutil.rmtree(db_path)
        print(f"✅ Database at {db_path} successfully deleted.")
    except Exception as e:
        print(f"❌ Failed to delete database: {str(e)}")

if __name__ == "__main__":
    db_exists = check_chroma_db()
    
    if db_exists:
        print("\nOptions:")
        print("1. Keep the existing database")
        print("2. Reset the database to start fresh")
        
        choice = input("Enter your choice (1/2): ")
        if choice == "2":
            reset_database()
