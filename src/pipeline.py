# src/fast_pipeline.py - Improved version
import os
import argparse
import sys
import asyncio
import time
import logging
import pickle
import networkx as nx
from typing import Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
from src.data_ingestion.vector_store_manager import fast_embed_and_store_chunks

# --- GNN Training Module ---
try:
    from src.gnn_training.train import run_training as run_gnn_training
except ImportError:
    run_gnn_training = None
    logger.warning("GNN training module not found. 'train-gnn' command unavailable.")

class FastPipeline:
    """Encapsulates the fast pipeline logic with better error handling and resource management."""
    
    def __init__(self):
        self.graph_parser = None
        self.start_time = None
        self.processed_count = 0
        self.error_count = 0
        self.total_chunks_processed = 0
        self.processed_doc_filenames = []
        
    async def run_ingestion(self) -> bool:
        """
        A unified and corrected data ingestion pipeline.
        Returns True if successful, False otherwise.
        """
        logger.info("🚀 Starting ingestion pipeline...")
        self.start_time = time.time()
        
        try:
            # 1. Load raw content of new documents
            logger.info("📂 Loading new documents...")
            all_new_docs_data = document_loader.load_new_documents()

            if not all_new_docs_data:
                logger.info("✅ No new documents to process. Pipeline finished.")
                return True

            logger.info(f"✓ Found {len(all_new_docs_data)} new documents to process")

            # 2. Parse documents and build knowledge graph
            logger.info("🧠 Building Knowledge Graph...")
            self.graph_parser = LatexToGraphParser(model_name=config.EMBEDDING_MODEL_NAME)
            
            # Use ProcessPoolExecutor for CPU-bound parsing
            max_workers = min(multiprocessing.cpu_count(), len(all_new_docs_data))
            
            for doc_data in all_new_docs_data:
                if doc_data['type'] == 'latex':
                    try:
                        logger.info(f"  🔄 Parsing: {doc_data['filename']}")
                        self.graph_parser.extract_structured_nodes(
                            latex_content=doc_data['raw_content'],
                            doc_id=doc_data['doc_id'],
                            source=doc_data['source']
                        )
                        self.processed_count += 1
                    except Exception as e:
                        logger.error(f"  ❌ Failed to parse {doc_data['filename']}: {e}")
                        self.error_count += 1
                        continue

            # 3. Save graph and embeddings
            logger.info("💾 Saving knowledge graph and embeddings...")
            os.makedirs('data/graph_db', exist_ok=True)
            os.makedirs('data/embeddings', exist_ok=True)

            self.graph_parser.save_graph_and_embeddings(
                'data/graph_db/knowledge_graph.graphml',
                'data/embeddings/initial_text_embeddings.pkl'
            )

            # 4. Convert graph nodes to conceptual blocks
            logger.info("🔪 Chunking conceptual blocks...")
            conceptual_blocks = self.graph_parser.get_graph_nodes_as_conceptual_blocks()

            if not conceptual_blocks:
                logger.warning("⚠️  No conceptual blocks generated from graph.")
                # Still log processed files
                self._update_processed_log(all_new_docs_data)
                return self.error_count == 0

            # 5. Chunk the blocks
            final_chunks = chunker.chunk_conceptual_blocks(conceptual_blocks)

            if not final_chunks:
                logger.warning("⚠️  No chunks generated for vector store")
                return False

            # 6. Fast embedding and storage
            logger.info(f"⚡ Fast embedding and storage of {len(final_chunks)} chunks...")
            try:
                client = vector_store_manager.get_weaviate_client()
                await fast_embed_and_store_chunks(client, final_chunks, batch_size=100)
            except Exception as e:
                logger.error(f"❌ Failed to store in Weaviate: {e}")
                return False

            # 7. Update processed docs log
            self._update_processed_log(all_new_docs_data)

            total_time = time.time() - self.start_time
            logger.info(f"✅ Fast ingestion pipeline completed in {total_time:.2f} seconds!")
            logger.info(f"   Processed: {self.processed_count} documents")
            logger.info(f"   Errors: {self.error_count} documents")

            if run_gnn_training:
                logger.info("Next: Run 'python -m src.fast_pipeline train-gnn' for graph-aware embeddings")

            return self.error_count == 0

        except Exception as e:
            logger.error(f"❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def _update_processed_log(self, docs_data: List[Dict]):
        """Update the processed documents log."""
        newly_processed_filenames = [doc['filename'] for doc in docs_data]
        document_loader.update_processed_docs_log(
            config.PROCESSED_DOCS_LOG_FILE, 
            newly_processed_filenames
        )

async def run_fast_ingestion_pipeline():
    """Wrapper for backward compatibility."""
    pipeline = FastPipeline()
    success = await pipeline.run_ingestion()
    if not success:
        sys.exit(1)

def run_fast_gnn_training():
    """
    Fast GNN training with optimizations
    """
    if not run_gnn_training:
        logger.error("❌ GNN training module not available")
        sys.exit(1)
    
    logger.info("🧠 Starting Fast GNN Training")
    start_time = time.time()
    
    try:
        # Check if required files exist
        if not os.path.exists('data/graph_db/knowledge_graph.graphml'):
            logger.error("❌ Knowledge graph not found. Run 'make ingest' first.")
            sys.exit(1)
            
        if not os.path.exists('data/embeddings/initial_text_embeddings.pkl'):
            logger.error("❌ Initial embeddings not found. Run 'make ingest' first.")
            sys.exit(1)
        
        run_gnn_training()
        total_time = time.time() - start_time
        logger.info(f"✅ GNN training completed in {total_time:.2f} seconds!")
        logger.info("Optimized embeddings saved to: data/embeddings/gnn_embeddings.pkl")
    except Exception as e:
        logger.error(f"❌ GNN training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

async def run_performance_test():
    """
    Test the performance of the optimized system
    """
    logger.info("🏃 Running Performance Tests")
    
    try:
        from src.api.fast_api import FastRAGComponents
        
        logger.info("Initializing components...")
        components = FastRAGComponents()
        components._init_core_components()
        
        logger.info("Testing retrieval performance...")
        
        # Test queries
        test_queries = [
            "vector space definition",
            "linear algebra examples",
            "matrix operations", 
            "eigenvalues",
            "proof techniques"
        ]
        
        # Run tests
        cold_times = []
        warm_times = []
        
        # Cold run
        logger.info("  Cold run (no cache)...")
        for query in test_queries:
            start = time.time()
            results = await components.retriever.fast_semantic_search(query, limit=5)
            elapsed = time.time() - start
            cold_times.append(elapsed)
            logger.info(f"    '{query}': {elapsed:.3f}s ({len(results)} results)")
        
        # Warm run
        logger.info("  Warm run (with cache)...")
        for query in test_queries:
            start = time.time()
            results = await components.retriever.fast_semantic_search(query, limit=5)
            elapsed = time.time() - start
            warm_times.append(elapsed)
            logger.info(f"    '{query}': {elapsed:.3f}s ({len(results)} results)")
        
        # Performance summary
        avg_cold = sum(cold_times) / len(cold_times) if cold_times else 0
        avg_warm = sum(warm_times) / len(warm_times) if warm_times else 0
        speedup = avg_cold / avg_warm if avg_warm > 0 else 0
        
        logger.info("📊 Performance Summary:")
        logger.info(f"  Average cold query time: {avg_cold:.3f}s")
        logger.info(f"  Average warm query time: {avg_warm:.3f}s")
        logger.info(f"  Cache speedup: {speedup:.1f}x")
        
        # Cache stats
        cache_stats = components.retriever.get_cache_stats()
        logger.info(f"  Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        logger.info(f"  Cached embeddings: {cache_stats['embedding_cache_size']}")
        logger.info(f"  Cached results: {cache_stats['results_cache_size']}")
        
        components.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main function with fast pipeline commands"""
    parser = argparse.ArgumentParser(
        description="Fast RAG Math Pipeline - Optimized for speed and performance",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Fast Ingestion Command
    subparsers.add_parser(
        "ingest", 
        help="🚀 Run fast data ingestion pipeline\n"
             "Processes LaTeX docs with parallel embedding and optimized storage"
    )
    
    # Fast GNN Training Command
    if run_gnn_training:
        subparsers.add_parser(
            "train-gnn", 
            help="🧠 Run optimized GNN training\n"
                 "Generates graph-aware embeddings for advanced retrieval"
        )

    # Performance Test Command
    subparsers.add_parser(
        "test", 
        help="🏃 Run performance tests\n"
             "Tests retrieval speed and caching effectiveness"
    )

    # Server Command
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
        asyncio.run(run_performance_test())
        
    elif args.command == "serve":
        logger.info("🖥️  Starting optimized API server...")
        import uvicorn
        uvicorn.run(
            "src.api.fast_api:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=False,  # Disable reload for better performance
            workers=1,     # Single worker to share cache
            log_level="info"
        )

if __name__ == '__main__':
    main()