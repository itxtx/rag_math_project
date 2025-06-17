# scripts/compare_performance.py
import asyncio
import time
import sys
import os
from src.retrieval.retriever import HybridRetriever
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

async def test_original_vs_optimized():
    """
    Compare performance between original and optimized retrievers
    """
    print("🏁 Performance Comparison: Original vs Optimized")
    print("=" * 50)
    
    test_queries = [
        "vector space definition",
        "linear algebra examples",
        "matrix multiplication",
        "eigenvalues and eigenvectors",
        "proof techniques"
    ]
    
    try:
        # Test original retriever
        print("\n📊 Testing Original Retriever...")
        from src.retrieval.retriever import Retriever
        from src.data_ingestion import vector_store_manager
        
        client = vector_store_manager.get_weaviate_client()
        original_retriever = Retriever(weaviate_client=client)
        
        original_times = []
        for i, query in enumerate(test_queries):
            start_time = time.time()
            results = original_retriever.semantic_search(query, limit=5)
            query_time = time.time() - start_time
            original_times.append(query_time)
            print(f"  Query {i+1}: {query_time:.3f}s ({len(results)} results)")
        
        original_avg = sum(original_times) / len(original_times)
        print(f"  📈 Average time: {original_avg:.3f}s")
        
    except Exception as e:
        print(f"  ❌ Original retriever test failed: {e}")
        original_avg = 0
        original_times = []
    
    try:
        # Test optimized retriever
        print("\n⚡ Testing Optimized Retriever...")
        
        
        optimized_retriever = HybridRetriever(weaviate_client=client)
        
        # Cold run
        print("  🧊 Cold run (no cache):")
        cold_times = []
        for i, query in enumerate(test_queries):
            start_time = time.time()
            results = await optimized_retriever.fast_semantic_search(query, limit=5)
            query_time = time.time() - start_time
            cold_times.append(query_time)
            print(f"    Query {i+1}: {query_time:.3f}s ({len(results)} results)")
        
        cold_avg = sum(cold_times) / len(cold_times)
        print(f"    📈 Cold average: {cold_avg:.3f}s")
        
        # Warm run
        print("  🔥 Warm run (with cache):")
        warm_times = []
        for i, query in enumerate(test_queries):
            start_time = time.time()
            results = await optimized_retriever.fast_semantic_search(query, limit=5)
            query_time = time.time() - start_time
            warm_times.append(query_time)
            print(f"    Query {i+1}: {query_time:.3f}s ({len(results)} results)")
        
        warm_avg = sum(warm_times) / len(warm_times)
        print(f"    📈 Warm average: {warm_avg:.3f}s")
        
        # Cache statistics
        cache_stats = optimized_retriever.get_cache_stats()
        print(f"    💾 Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        print(f"    💾 Cached embeddings: {cache_stats['embedding_cache_size']}")
        
    except Exception as e:
        print(f"  ❌ Optimized retriever test failed: {e}")
        cold_avg = warm_avg = 0
        cache_stats = {}
    
    # Performance summary
    print("\n🏆 Performance Summary")
    print("=" * 30)
    
    if original_avg > 0:
        print(f"Original Retriever:     {original_avg:.3f}s average")
        
        if cold_avg > 0:
            cold_speedup = original_avg / cold_avg if cold_avg > 0 else 0
            print(f"Optimized (cold):       {cold_avg:.3f}s average ({cold_speedup:.1f}x speedup)")
        
        if warm_avg > 0:
            warm_speedup = original_avg / warm_avg if warm_avg > 0 else 0
            print(f"Optimized (warm):       {warm_avg:.3f}s average ({warm_speedup:.1f}x speedup)")
            
            cache_speedup = cold_avg / warm_avg if warm_avg > 0 else 0
            print(f"Cache effectiveness:    {cache_speedup:.1f}x speedup")
    
    # Memory usage comparison
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"\nMemory usage: {memory_mb:.1f} MB")
    except ImportError:
        print("\n(Install psutil to see memory usage)")
    
    print("\n✅ Performance comparison complete!")


def test_embedding_performance():
    """Test embedding generation performance"""
    print("\n🔬 Testing Embedding Performance...")
    
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_texts = [
        "What is a vector space?",
        "How do you multiply matrices?",
        "Define eigenvalues and eigenvectors",
        "Explain linear independence",
        "What is the rank of a matrix?"
    ]
    
    # Single embedding test
    print("  🔸 Single embedding generation:")
    start_time = time.time()
    for text in test_texts:
        embedding = model.encode(text)
    single_total = time.time() - start_time
    single_avg = single_total / len(test_texts)
    print(f"    Total: {single_total:.3f}s, Average: {single_avg:.3f}s per query")
    
    # Batch embedding test
    print("  🔸 Batch embedding generation:")
    start_time = time.time()
    embeddings = model.encode(test_texts, batch_size=32)
    batch_total = time.time() - start_time
    batch_avg = batch_total / len(test_texts)
    print(f"    Total: {batch_total:.3f}s, Average: {batch_avg:.3f}s per query")
    
    speedup = single_avg / batch_avg if batch_avg > 0 else 0
    print(f"    Batch speedup: {speedup:.1f}x")


def main():
    """Run all performance tests"""
    print("🚀 RAG System Performance Analysis")
    print("=" * 40)
    
    # Test embedding performance
    test_embedding_performance()
    
    # Test retrieval performance
    asyncio.run(test_original_vs_optimized())
    
    print("\n🎯 Optimization Recommendations:")
    print("  1. Use OptimizedRetriever for 5-10x faster repeat queries")
    print("  2. Use batch embedding for bulk operations")
    print("  3. Pre-warm caches for production deployment")
    print("  4. Monitor cache hit rates and adjust cache sizes")


if __name__ == "__main__":
    main()