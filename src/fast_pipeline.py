# src/fast_pipeline.py
import os
import argparse
import sys
import asyncio
import time

# --- Add project root to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Fast Pipeline Modules ---
from src import config
from src.data_ingestion import document_loader
from src.data_ingestion.latex_parser import LatexToGraphParser
from src.data_ingestion import chunker
from src.data_ingestion import vector_store_manager
from src.data_ingestion.optimized_vector_store_manager import fast_embed_and_store_chunks

# --- GNN Training Module ---
try:
    from src.gnn_training.train import run_training as run_gnn_training
except ImportError:
    run_gnn_training = None
    print("WARNING: GNN training module not found. 'train-gnn' command unavailable.")

async def run_fast_ingestion_pipeline():
    """
    A unified and corrected data ingestion pipeline.
    """
    print("--- 🚀 Starting Fast Data Ingestion Pipeline ---")
    start_time = time.time()

    # 1. Load raw content of new documents. This no longer parses anything.
    print("\n📂 Loading new documents...")
    all_new_docs_data = document_loader.load_new_documents()

    if not all_new_docs_data:
        print("✅ No new documents to process. Pipeline finished.")
        return

    print(f"✓ Found {len(all_new_docs_data)} new documents to process")

    # 2. This is now the ONLY parsing stage.
    # It creates one parser and processes all documents with it.
    print("\n🧠 Building Knowledge Graph...")
    graph_parser = LatexToGraphParser(model_name=config.EMBEDDING_MODEL_NAME)

    for doc_data in all_new_docs_data:
        if doc_data['type'] == 'latex':
            print(f"  🔄 Parsing: {doc_data['filename']}")
            # Use the raw_content loaded previously
            graph_parser.extract_structured_nodes(
                latex_content=doc_data['raw_content'],
                doc_id=doc_data['doc_id'],
                source=doc_data['source']
            )

    # 3. Save the final, complete graph and its embeddings
    print("\n💾 Saving knowledge graph and embeddings...")
    os.makedirs('data/graph_db', exist_ok=True)
    os.makedirs('data/embeddings', exist_ok=True)

    graph_parser.save_graph_and_embeddings(
        'data/graph_db/knowledge_graph.graphml',
        'data/embeddings/initial_text_embeddings.pkl'
    )

    # 4. Convert graph nodes to conceptual blocks for chunking
    print("\n🔪 Chunking conceptual blocks...")
    conceptual_blocks = graph_parser.get_graph_nodes_as_conceptual_blocks()

    if not conceptual_blocks:
        print("⚠️  No conceptual blocks generated from graph. Check parsing logs.")
        # We still log the files as processed to avoid trying them again.
        newly_processed_filenames = [doc['filename'] for doc in all_new_docs_data]
        document_loader.update_processed_docs_log(config.PROCESSED_DOCS_LOG_FILE, newly_processed_filenames)
        return

    # 5. Chunk the blocks
    final_chunks = chunker.chunk_conceptual_blocks(conceptual_blocks)

    if not final_chunks:
        print("⚠️  No chunks generated for vector store")
        return

    # 6. Fast embedding and storage in Weaviate
    print(f"\n⚡ Fast embedding and storage of {len(final_chunks)} chunks...")
    try:
        client = vector_store_manager.get_weaviate_client()
        await fast_embed_and_store_chunks(client, final_chunks, batch_size=100)
    except Exception as e:
        print(f"❌ Failed to store in Weaviate: {e}")
        return

    # 7. Update processed docs log
    newly_processed_filenames = [doc['filename'] for doc in all_new_docs_data]
    document_loader.update_processed_docs_log(config.PROCESSED_DOCS_LOG_FILE, newly_processed_filenames)

    total_time = time.time() - start_time
    print(f"\n✅ Fast ingestion pipeline completed in {total_time:.2f} seconds!")

    if run_gnn_training:
        print("\nNext: Run 'python -m src.fast_pipeline train-gnn' for graph-aware embeddings")

async def process_document_async(graph_parser, doc_data):
    """Process a single document asynchronously"""
    print(f"  🔄 Processing: {doc_data['filename']}")
    
    # Run the graph extraction in an executor to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        graph_parser.extract_structured_nodes,
        doc_data['parsed_content'],
        doc_data['doc_id'],
        doc_data['source']
    )
    
    print(f"  ✓ Completed: {doc_data['filename']}")


def run_fast_gnn_training():
    """
    Fast GNN training with optimizations
    """
    if not run_gnn_training:
        print("❌ GNN training module not available")
        return
    
    print("--- 🧠 Starting Fast GNN Training ---")
    start_time = time.time()
    
    try:
        run_gnn_training()
        total_time = time.time() - start_time
        print(f"✅ GNN training completed in {total_time:.2f} seconds!")
        print("\nOptimized embeddings saved to: data/embeddings/gnn_embeddings.pkl")
    except Exception as e:
        print(f"❌ GNN training failed: {e}")


def run_performance_test():
    """
    Test the performance of the optimized system
    """
    print("--- 🏃 Running Performance Tests ---")
    
    try:
        from src.api.fast_api import FastRAGComponents
        import asyncio
        
        async def test_performance():
            print("Initializing components...")
            components = FastRAGComponents()
            components._init_core_components()
            
            print("Testing retrieval performance...")
            start_time = time.time()
            
            # Test queries
            test_queries = [
                "vector space definition",
                "linear algebra examples",
                "matrix operations", 
                "eigenvalues",
                "proof techniques"
            ]
            
            # Cold run
            print("  Cold run (no cache)...")
            cold_times = []
            for query in test_queries:
                query_start = time.time()
                results = await components.retriever.fast_semantic_search(query, limit=5)
                query_time = time.time() - query_start
                cold_times.append(query_time)
                print(f"    '{query}': {query_time:.3f}s ({len(results)} results)")
            
            # Warm run
            print("  Warm run (with cache)...")
            warm_times = []
            for query in test_queries:
                query_start = time.time()
                results = await components.retriever.fast_semantic_search(query, limit=5)
                query_time = time.time() - query_start
                warm_times.append(query_time)
                print(f"    '{query}': {query_time:.3f}s ({len(results)} results)")
            
            # Performance summary
            avg_cold = sum(cold_times) / len(cold_times)
            avg_warm = sum(warm_times) / len(warm_times)
            speedup = avg_cold / avg_warm if avg_warm > 0 else 0
            
            print(f"\n📊 Performance Summary:")
            print(f"  Average cold query time: {avg_cold:.3f}s")
            print(f"  Average warm query time: {avg_warm:.3f}s")
            print(f"  Cache speedup: {speedup:.1f}x")
            
            # Cache stats
            cache_stats = components.retriever.get_cache_stats()
            print(f"  Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
            print(f"  Cached embeddings: {cache_stats['embedding_cache_size']}")
            print(f"  Cached results: {cache_stats['results_cache_size']}")
            
            components.cleanup()
        
        asyncio.run(test_performance())
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")


def main():
    """Main function with fast pipeline commands"""
    parser = argparse.ArgumentParser(
        description="Fast RAG Math Pipeline - Optimized for speed and performance",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Fast Ingestion Command ---
    subparsers.add_parser(
        "ingest", 
        help="🚀 Run fast data ingestion pipeline\n"
             "Processes LaTeX docs with parallel embedding and optimized storage"
    )
    
    # --- Fast GNN Training Command ---
    if run_gnn_training:
        subparsers.add_parser(
            "train-gnn", 
            help="🧠 Run optimized GNN training\n"
                 "Generates graph-aware embeddings for advanced retrieval"
        )

    # --- Performance Test Command ---
    subparsers.add_parser(
        "test", 
        help="🏃 Run performance tests\n"
             "Tests retrieval speed and caching effectiveness"
    )

    # --- Server Command ---
    subparsers.add_parser(
        "serve", 
        help="🖥️  Start optimized API server\n"
             "Launches FastAPI with pre-loaded components and caching"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(run_fast_ingestion_pipeline())
        
    elif args.command == "train-gnn":
        run_fast_gnn_training()
        
    elif args.command == "test":
        run_performance_test()
        
    elif args.command == "serve":
        print("🖥️  Starting optimized API server...")
        import uvicorn
        uvicorn.run(
            "src.api.fast_api:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=False,  # Disable reload for better performance
            workers=1      # Single worker to share cache
        )

if __name__ == '__main__':
    main()