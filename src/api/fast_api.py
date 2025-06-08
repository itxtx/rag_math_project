# src/api/fast_api.py
import os
import asyncio
import time
from fastapi import FastAPI, HTTPException, Depends, Request
from typing import Dict, Any, Optional, List as PyList
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import datetime
import uuid

from src import config
from src.api.models import (
    LearnerInteractionStartRequest, 
    QuestionResponse, 
    AnswerSubmissionRequest, AnswerSubmissionResponse,
    EvaluationResult, ErrorResponse, TopicResponse
)
from src.adaptive_engine.srs_scheduler import SRSScheduler
from src.data_ingestion import vector_store_manager
from src.learner_model.profile_manager import LearnerProfileManager
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.evaluation.answer_evaluator import AnswerEvaluator
from src.learner_model.knowledge_tracker import KnowledgeTracker
from src.interaction.answer_handler import AnswerHandler
from src.adaptive_engine.question_selector import QuestionSelector

# Import our optimized components
from src.retrieval.optimized_retriever import OptimizedRetriever

class FastRAGComponents:
    """
    Optimized RAG components with pre-loading and async initialization
    """
    
    def __init__(self):
        print("🚀 Initializing Fast RAG Components...")
        self.initialization_error = None
        self.startup_time = time.time()
        
        # Component cache
        self._profile_manager = None
        self._retriever = None
        self._question_generator = None
        self._answer_evaluator = None
        
        # Pre-warm flag
        self._prewarmed = False
    
    def _init_core_components(self):
        """Initialize core components synchronously"""
        try:
            # Weaviate connection
            print("  📡 Connecting to Weaviate...")
            self.weaviate_client = vector_store_manager.get_weaviate_client()
            vector_store_manager.create_weaviate_schema(self.weaviate_client)
            
            # Use optimized retriever
            print("  🔍 Initializing optimized retriever...")
            self._retriever = OptimizedRetriever(weaviate_client=self.weaviate_client)
            
            # Profile manager
            print("  👤 Initializing profile manager...")
            self._profile_manager = LearnerProfileManager()
            
            # SRS Scheduler
            print("  📅 Initializing SRS scheduler...")
            self.srs_scheduler = SRSScheduler()
            
            print("  ✓ Core components initialized")
            
        except Exception as e:
            print(f"  ❌ Error initializing core components: {e}")
            self.initialization_error = e
            raise
    
    def _init_llm_components(self):
        """Initialize LLM components (slower, done lazily)"""
        if not self._question_generator:
            print("  🤖 Initializing question generator...")
            self._question_generator = RAGQuestionGenerator()
        
        if not self._answer_evaluator:
            print("  📝 Initializing answer evaluator...")
            self._answer_evaluator = AnswerEvaluator()
    
    def _init_dependent_components(self):
        """Initialize components that depend on others"""
        print("  🔗 Initializing dependent components...")
        
        self.knowledge_tracker = KnowledgeTracker(
            profile_manager=self._profile_manager,
            srs_scheduler=self.srs_scheduler
        )
        
        self.question_selector = QuestionSelector(
            profile_manager=self._profile_manager,
            retriever=self._retriever,
            question_generator=self._question_generator
        )
        
        self.answer_handler = AnswerHandler(
            evaluator=self._answer_evaluator,
            tracker=self.knowledge_tracker
        )
        
        print("  ✓ All components initialized")
    
    async def ensure_ready(self):
        """Ensure all components are ready (lazy initialization)"""
        if self.initialization_error:
            raise Exception(f"Components failed to initialize: {self.initialization_error}")
        
        # Initialize LLM components if not already done
        if not self._question_generator or not self._answer_evaluator:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_llm_components)
            await loop.run_in_executor(None, self._init_dependent_components)
        
        # Pre-warm if not already done
        if not self._prewarmed:
            await self._prewarm_system()
            self._prewarmed = True
    
    async def _prewarm_system(self):
        """Pre-warm caches with common operations"""
        print("  🔥 Pre-warming system caches...")
        
        try:
            # Pre-warm embedding cache with common queries
            common_queries = [
                "vector space",
                "linear algebra", 
                "matrix",
                "definition",
                "theorem"
            ]
            
            # Pre-warm one at a time to avoid memory issues
            for query in common_queries:
                try:
                    print(f"  Pre-warming cache for query: {query}")
                    await self._retriever.fast_semantic_search(query, limit=1)
                except Exception as e:
                    print(f"  ⚠️  Failed to pre-warm cache for query '{query}': {e}")
                    continue
            
            # Load curriculum map
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.question_selector._load_curriculum_map
                )
            except Exception as e:
                print(f"  ⚠️  Failed to load curriculum map: {e}")
            
            print(f"  ✓ System pre-warmed in {time.time() - self.startup_time:.2f}s")
            
        except Exception as e:
            print(f"  ⚠️  Pre-warming failed (non-critical): {e}")
            # Don't raise the error - pre-warming is non-critical
            pass
    
    @property
    def profile_manager(self):
        return self._profile_manager
    
    @property 
    def retriever(self):
        return self._retriever
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {
            "startup_time": time.time() - self.startup_time,
            "prewarmed": self._prewarmed,
            "components_ready": self.initialization_error is None
        }
        
        if self._retriever:
            stats.update(self._retriever.get_cache_stats())
        
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        if self._profile_manager:
            self._profile_manager.close_db()
        print("  🧹 Resources cleaned up")

    async def _get_context_for_concept(self, concept_id: str, limit: int = 2) -> str:
        """Get context for a concept with proper LaTeX formatting"""
        try:
            chunks = await self.retriever.get_chunks_by_parent_block(concept_id, limit)
            if not chunks:
                return "No context available for this concept."
                
            # Get the topic from the first chunk
            topic = chunks[0].get('doc_id', 'Unknown Topic')
            topic = topic.replace('_', ' ').title()
            
            # Build context with proper LaTeX formatting
            context_parts = []
            for chunk in chunks:
                if chunk.get('text'):
                    # Clean up any malformed LaTeX
                    text = chunk['text'].strip()
                    if not text.startswith('$'):
                        text = f"${text}$"
                    context_parts.append(text)
            
            if not context_parts:
                return "No valid context available for this concept."
                
            context = "\n\n".join(context_parts)
            
            return f"""--- Context for New Concept ---
Topic: {topic}
Please review the following information before answering the question:
--------------------------------------------------------------------
{context}
--------------------------------------------------------------------"""
            
        except Exception as e:
            print(f"Error getting context for concept {concept_id}: {e}")
            return "Error retrieving context for this concept."

# Global state
app_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app"""
    print("🚀 Fast API startup...")
    
    # Initialize components
    print("🚀 Initializing Fast RAG Components...")
    components = FastRAGComponents()
    components._init_core_components()
    components._init_llm_components()
    components._init_dependent_components()
    
    # Store components in app state
    app_state["components"] = components
    app_state["profile_manager"] = components._profile_manager
    app_state["question_selector"] = components.question_selector
    app_state["question_generator"] = components._question_generator
    app_state["answer_evaluator"] = components._answer_evaluator
    app_state["active_interactions"] = {}
    
    print("✅ Fast startup complete! Core components ready.")
    print("   LLM components will be loaded on first request.")
    
    yield
    
    print("🛑 Shutting down...")
    if "components" in app_state:
        await app_state["components"].cleanup()
    print("  🧹 Resources cleaned up")

# Create optimized app
app = FastAPI(
    title="Fast Adaptive RAG Learning System",
    description="High-performance RAG API with optimized retrieval",
    version="0.2.0",
    lifespan=lifespan
)

# CORS
origins = [
    "http://localhost", "http://localhost:3000", "http://localhost:8080", "http://localhost:5173", 
    "http://127.0.0.1:3000", "http://127.0.0.1:8000", "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

async def get_fast_components() -> FastRAGComponents:
    """Get components and ensure they're ready"""
    components = app_state.get("components")
    if not components:
        raise HTTPException(status_code=503, detail="System not ready")
    
    await components.ensure_ready()
    return components

# Performance tracking middleware
@app.middleware("http")
async def track_performance(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    return response

# Optimized endpoints
@app.get("/api/v1/topics", response_model=PyList[TopicResponse])
async def list_topics_fast(components: FastRAGComponents = Depends(get_fast_components)):
    """Get list of available topics"""
    try:
        print("Starting topics endpoint...")
        # Get all documents without semantic search
        print("Fetching all documents...")
        results = await components.retriever.get_all_documents(limit=1000)
        print(f"Found {len(results)} documents")
        
        # Group by doc_id and create topic responses
        topics = {}
        for result in results:
            doc_id = result.get('doc_id')
            if not doc_id:
                print(f"Skipping result with no doc_id: {result}")
                continue
            if doc_id in topics:
                print(f"Skipping duplicate doc_id: {doc_id}")
                continue
                
            print(f"Processing document: {doc_id}")
            
            # Get document title and description
            title = result.get('title')
            if not title:
                # Try to get title from filename
                filename = result.get('filename', '')
                if filename:
                    # Remove .tex extension and convert underscores to spaces
                    title = os.path.splitext(filename)[0].replace('_', ' ').title()
                    print(f"Generated title from filename for {doc_id}: {title}")
                else:
                    # Use doc_id as fallback
                    title = doc_id.replace('_', ' ').title()
                    print(f"Using doc_id as title for {doc_id}: {title}")
            
            topic = TopicResponse(
                doc_id=doc_id,
                title=title,
                description=result.get('description', ''),
                concepts=[c for c in [result.get('concept_name')] if c]  # Only include non-None concepts
            )
            print(f"Created topic response for {doc_id}: {topic}")
            topics[doc_id] = topic
        
        topic_list = list(topics.values())
        print(f"Returning {len(topic_list)} topics")
        return topic_list
        
    except Exception as e:
        print(f"Error listing topics: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list topics: {str(e)}"
        )

@app.post("/api/v1/interaction/start", response_model=QuestionResponse)
async def start_interaction(
    request: Request,
    learner_id: int = 1,
    topic_id: Optional[str] = None
):
    """Start a new interaction with a question"""
    try:
        print(f"🎯 Fast interaction start for learner: {learner_id}")
        
        # Get or create learner profile
        profile = await app_state["profile_manager"].get_or_create_profile(learner_id)
        
        # Select topic if not provided
        if not topic_id:
            topics = await list_topics_fast()
            if not topics:
                raise HTTPException(status_code=404, detail="No topics available")
            topic_id = topics[0].doc_id
        
        # Get next question
        question_data = await app_state["question_selector"].select_next_question(
            learner_id=learner_id,
            topic_id=topic_id
        )
        
        if not question_data:
            raise HTTPException(status_code=404, detail="No questions available for this topic")
        
        # Get context for the concept
        context = await app_state["components"]._get_context_for_concept(question_data["concept_id"])
        
        # Generate question
        question = await app_state["question_generator"].generate_question(
            context=context,
            difficulty=question_data["difficulty"],
            question_type=question_data["type"],
            style=question_data["style"]
        )
        
        if not question:
            raise HTTPException(status_code=500, detail="Failed to generate question")
        
        # Create interaction
        interaction_id = str(uuid.uuid4())
        app_state["active_interactions"][interaction_id] = {
            "learner_id": learner_id,
            "topic_id": topic_id,
            "concept_id": question_data["concept_id"],
            "question": question,
            "start_time": datetime.now(),
            "status": "active"
        }
        
        return QuestionResponse(
            interaction_id=interaction_id,
            question=question,
            concept_id=question_data["concept_id"],
            concept_name=question_data.get("concept_name", "Unknown Concept"),
            topic_id=topic_id,
            difficulty=question_data["difficulty"],
            question_type=question_data["type"]
        )
        
    except Exception as e:
        print(f"❌ Error in fast interaction start: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/interaction/submit_answer", response_model=AnswerSubmissionResponse)
async def submit_answer_fast(
    request: AnswerSubmissionRequest,
    components: FastRAGComponents = Depends(get_fast_components)
):
    """Fast answer submission with async evaluation"""
    print(f"📝 Fast answer submission for learner: {request.learner_id}")
    
    try:
        # Run answer handling asynchronously
        handler_response = await components.answer_handler.submit_answer(
            learner_id=request.learner_id,
            question_id=request.question_id,
            doc_id=request.doc_id,
            question_text=request.question_text,
            retrieved_context=request.context_for_evaluation,
            learner_answer=request.learner_answer
        )
        
        eval_result = EvaluationResult(
            accuracy_score=handler_response.get("accuracy_score", 0.0),
            feedback=handler_response.get("feedback", "Evaluation feedback not available."),
            correct_answer=handler_response.get("correct_answer_suggestion")
        )
        
        return AnswerSubmissionResponse(
            learner_id=request.learner_id,
            question_id=request.question_id,
            doc_id=request.doc_id,
            evaluation=eval_result
        )
        
    except Exception as e:
        print(f"❌ Error in fast answer submission: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/api/v1/health")
async def health_check_fast():
    """Health check with performance metrics"""
    components = app_state.get("components")
    
    health_data = {
        "status": "healthy" if components and not components.initialization_error else "degraded",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    
    if components:
        health_data.update(components.get_performance_stats())
    
    return health_data

@app.get("/api/v1/performance")
async def get_performance_stats():
    """Get detailed performance statistics"""
    components = app_state.get("components")
    if not components:
        return {"error": "Components not initialized"}
    
    return components.get_performance_stats()

@app.post("/api/v1/clear_cache")
async def clear_cache():
    """Clear all caches (useful for testing)"""
    components = app_state.get("components")
    if components and components._retriever:
        components._retriever.clear_cache()
        return {"message": "Cache cleared successfully"}
    return {"message": "No cache to clear"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Fast RAG API Server...")
    uvicorn.run("src.api.fast_api:app", host="0.0.0.0", port=8000, reload=True)