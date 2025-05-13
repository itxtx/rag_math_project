from src.data_ingestion import document_loader
from src import config # To initialize paths etc.
# import weaviate # We'll use this later

def main():
    print("Starting RAG System Prototype...")

    # 1. Load and Parse Documents
    print("\nStep 1: Loading and Parsing Documents...")
    parsed_docs = document_loader.load_and_parse_documents()
    
    if not parsed_docs:
        print("No documents were processed. Exiting.")
        return

    print(f"\nSuccessfully parsed {len(parsed_docs)} documents.")
    for i, doc_info in enumerate(parsed_docs):
        print(f"  Document {i+1}:")
        print(f"    Source: {doc_info['source']}")
        print(f"    Type: {doc_info['type']}")
        # print(f"    Content: {doc_info['content'][:200]}...") # Potentially long
    
    # TODO: Step 2: Chunk Documents
    # chunks = chunk_documents(parsed_docs)
    print("\nStep 2: Chunking Documents (Not Implemented Yet)")

    # TODO: Step 3: Initialize Weaviate Client
    # client = weaviate.Client(config.WEAVIATE_URL)
    print("\nStep 3: Initialize Weaviate Client (Not Implemented Yet)")

    # TODO: Step 4: Embed and Store Chunks
    # embed_and_store(chunks, client)
    print("\nStep 4: Embed and Store Chunks (Not Implemented Yet)")

    # TODO: Step 5: Implement Querying
    # results = query_system("What is the meaning of life?", client)
    print("\nStep 5: Implement Querying (Not Implemented Yet)")

    print("\nPrototype run finished.")

if __name__ == "__main__":
    main()