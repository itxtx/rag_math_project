#!/usr/bin/env python3
"""
Clean all databases and caches for fresh testing
Usage: python scripts/clean_databases.py [--confirm]
"""

import os
import sys
import shutil
import weaviate
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import config
from src.data_ingestion import vector_store_manager

def print_status(message, status="INFO"):
    """Print colored status messages"""
    colors = {
        "INFO": "\033[0;34m",  # Blue
        "SUCCESS": "\033[0;32m",  # Green
        "WARNING": "\033[1;33m",  # Yellow
        "ERROR": "\033[0;31m",  # Red
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}[{status}] {message}{reset}")

def clean_weaviate_database():
    """Clean all data from Weaviate"""
    print_status("Cleaning Weaviate database...", "INFO")
    
    try:
        # Connect to Weaviate
        client = vector_store_manager.get_weaviate_client()
        
        # Check if class exists
        schema = client.schema.get()
        class_exists = any(c['class'] == vector_store_manager.WEAVIATE_CLASS_NAME 
                          for c in schema.get('classes', []))
        
        if class_exists:
            # Get object count before deletion
            result = client.query.aggregate(vector_store_manager.WEAVIATE_CLASS_NAME)\
                .with_meta_count()\
                .do()
            
            count = 0
            if result and 'data' in result:
                aggregates = result['data']['Aggregate'].get(vector_store_manager.WEAVIATE_CLASS_NAME, [])
                if aggregates:
                    count = aggregates[0]['meta']['count']
            
            print_status(f"Found {count} objects in Weaviate", "INFO")
            
            if count > 0:
                # Delete all objects in the class
                client.batch.delete_objects(
                    class_name=vector_store_manager.WEAVIATE_CLASS_NAME,
                    where={
                        "path": ["chunk_id"],
                        "operator": "Like",
                        "valueText": "*"  # Match all
                    }
                )
                print_status(f"Deleted all objects from {vector_store_manager.WEAVIATE_CLASS_NAME}", "SUCCESS")
            
            # Optionally delete and recreate the entire class
            print_status("Deleting and recreating Weaviate class...", "INFO")
            client.schema.delete_class(vector_store_manager.WEAVIATE_CLASS_NAME)
            print_status("Class deleted", "SUCCESS")
            
            # Recreate the schema
            vector_store_manager.create_weaviate_schema(client)
            print_status("Class recreated with fresh schema", "SUCCESS")
            
        else:
            print_status("Weaviate class doesn't exist, creating fresh schema...", "WARNING")
            vector_store_manager.create_weaviate_schema(client)
            print_status("Fresh schema created", "SUCCESS")
            
    except Exception as e:
        print_status(f"Error cleaning Weaviate: {e}", "ERROR")
        return False
    
    return True

def clean_graph_database():
    """Clean graph database files"""
    print_status("Cleaning graph database...", "INFO")
    
    graph_dir = Path("data/graph_db")
    files_to_delete = [
        graph_dir / "knowledge_graph.graphml",
    ]
    
    deleted_count = 0
    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()
            print_status(f"Deleted: {file_path}", "SUCCESS")
            deleted_count += 1
    
    if deleted_count == 0:
        print_status("No graph files found to delete", "WARNING")
    
    return True

def clean_embeddings():
    """Clean embeddings files and cache"""
    print_status("Cleaning embeddings...", "INFO")
    
    embeddings_dir = Path("data/embeddings")
    files_to_delete = [
        embeddings_dir / "initial_text_embeddings.pkl",
        embeddings_dir / "gnn_embeddings.pkl",
    ]
    
    # Delete specific files
    deleted_count = 0
    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()
            print_status(f"Deleted: {file_path}", "SUCCESS")
            deleted_count += 1
    
    # Clean cache directory
    cache_dir = embeddings_dir / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        print_status("Cleaned embeddings cache", "SUCCESS")
    
    if deleted_count == 0:
        print_status("No embedding files found to delete", "WARNING")
    
    return True

def clean_processed_log():
    """Clean the processed documents log"""
    print_status("Cleaning processed documents log...", "INFO")
    
    log_file = Path(config.PROCESSED_DOCS_LOG_FILE)
    if log_file.exists():
        # Backup the log file first
        backup_path = log_file.with_suffix('.txt.backup')
        shutil.copy2(log_file, backup_path)
        print_status(f"Backed up log to: {backup_path}", "INFO")
        
        # Clear the log
        log_file.unlink()
        print_status("Cleared processed documents log", "SUCCESS")
    else:
        print_status("No processed documents log found", "WARNING")
    
    return True

def clean_learner_profiles():
    """Clean learner profiles database"""
    print_status("Cleaning learner profiles...", "INFO")
    
    db_path = Path("data/learner_profiles.sqlite3")
    if db_path.exists():
        # Backup first
        backup_path = db_path.with_suffix('.sqlite3.backup')
        shutil.copy2(db_path, backup_path)
        print_status(f"Backed up database to: {backup_path}", "INFO")
        
        # Delete the database
        db_path.unlink()
        print_status("Deleted learner profiles database", "SUCCESS")
    else:
        print_status("No learner profiles database found", "WARNING")
    
    return True

def clean_temp_files():
    """Clean temporary LaTeX processing files"""
    print_status("Cleaning temporary files...", "INFO")
    
    temp_dirs = [
        Path("data/latex_temp/logs"),
        Path("data/performance_logs"),
    ]
    
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            print_status(f"Cleaned: {temp_dir}", "SUCCESS")
    
    # Clean Python cache
    for root, dirs, files in os.walk("src"):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"))
    
    print_status("Cleaned Python cache files", "SUCCESS")
    
    return True

def get_database_stats():
    """Get current database statistics"""
    print_status("Current Database Statistics", "INFO")
    print("=" * 50)
    
    # Weaviate stats
    try:
        client = vector_store_manager.get_weaviate_client()
        result = client.query.aggregate(vector_store_manager.WEAVIATE_CLASS_NAME)\
            .with_meta_count()\
            .do()
        
        count = 0
        if result and 'data' in result:
            aggregates = result['data']['Aggregate'].get(vector_store_manager.WEAVIATE_CLASS_NAME, [])
            if aggregates:
                count = aggregates[0]['meta']['count']
        
        print(f"Weaviate objects: {count}")
    except:
        print("Weaviate: Not accessible or empty")
    
    # Graph database stats
    graph_file = Path("data/graph_db/knowledge_graph.graphml")
    if graph_file.exists():
        size_mb = graph_file.stat().st_size / (1024 * 1024)
        print(f"Graph database: {size_mb:.2f} MB")
    else:
        print("Graph database: Not found")
    
    # Embeddings stats
    embeddings_files = [
        ("Initial embeddings", "data/embeddings/initial_text_embeddings.pkl"),
        ("GNN embeddings", "data/embeddings/gnn_embeddings.pkl"),
    ]
    
    for name, path in embeddings_files:
        file_path = Path(path)
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"{name}: {size_mb:.2f} MB")
        else:
            print(f"{name}: Not found")
    
    # Processed documents
    log_file = Path(config.PROCESSED_DOCS_LOG_FILE)
    if log_file.exists():
        with open(log_file, 'r') as f:
            doc_count = len(f.readlines())
        print(f"Processed documents: {doc_count}")
    else:
        print("Processed documents: 0")
    
    print("=" * 50)

def main():
    """Main cleaning function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean RAG Math databases for testing")
    parser.add_argument("--confirm", action="store_true", 
                       help="Skip confirmation prompt")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only show statistics, don't clean")
    parser.add_argument("--keep-profiles", action="store_true",
                       help="Keep learner profiles")
    parser.add_argument("--keep-log", action="store_true",
                       help="Keep processed documents log")
    
    args = parser.parse_args()
    
    # Show current stats
    get_database_stats()
    
    if args.stats_only:
        return
    
    # Confirmation
    if not args.confirm:
        print("\n" + "="*50)
        print("⚠️  WARNING: This will delete all data in the databases!")
        print("This includes:")
        print("  - All Weaviate vector embeddings")
        print("  - The knowledge graph")
        print("  - All computed embeddings")
        if not args.keep_profiles:
            print("  - All learner profiles")
        if not args.keep_log:
            print("  - The processed documents log")
        print("="*50)
        
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            print_status("Cleaning cancelled", "WARNING")
            return
    
    print("\n" + "="*50)
    print_status("Starting database cleanup...", "INFO")
    
    # Clean each component
    success = True
    
    # Clean Weaviate
    if not clean_weaviate_database():
        success = False
    
    # Clean graph database
    if not clean_graph_database():
        success = False
    
    # Clean embeddings
    if not clean_embeddings():
        success = False
    
    # Clean processed log (optional)
    if not args.keep_log:
        if not clean_processed_log():
            success = False
    
    # Clean learner profiles (optional)
    if not args.keep_profiles:
        if not clean_learner_profiles():
            success = False
    
    # Clean temp files
    if not clean_temp_files():
        success = False
    
    print("\n" + "="*50)
    if success:
        print_status("Database cleanup completed successfully!", "SUCCESS")
        print_status("You can now run 'make ingest' to process documents fresh", "INFO")
    else:
        print_status("Some errors occurred during cleanup", "ERROR")
        print_status("Check the error messages above", "WARNING")

if __name__ == "__main__":
    main()