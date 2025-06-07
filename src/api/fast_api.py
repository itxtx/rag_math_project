# src/api/fast_api.py
import os
import asyncio
import time
from fastapi import FastAPI, HTTPException, Depends
from typing import Dict, Any, Optional, List as PyList
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import datetime

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
            
            # Pre-warm in background
            prewarm_tasks = [
                self._retriever.fast_semantic_search(query, limit=1)
                for query in common_queries
            ]
            
            await asyncio.gather(*prewarm_tasks, return_exceptions=True)
            
            # Load curriculum map
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.question_selector._load_curriculum_map
            )
            
            print(f"  ✓ System pre-warmed in {time.time() - self.startup_time:.2f}s")
            
        except Exception as e:
            print(f"  ⚠️  Pre-warming failed (non-critical): {e}")
    
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

# Global state
app_state: Dict[str, Any] = {}

@asynccontextmanager
async def fast_lifespan(app_instance: FastAPI):
    """Fast startup lifespan manager"""
    print("🚀 Fast API startup...")
    
    # Load environment
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
    
    # Initialize components (core only, LLM components loaded on-demand)
    try:
        components = FastRAGComponents()
        components._init_core_components()
        app_state["rag_components"] = components
        
        print(f"✅ Fast startup complete! Core components ready.")
        print("   LLM components will be loaded on first request.")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        app_state["rag_components"] = None
    
    yield
    
    # Cleanup
    print("🛑 Shutting down...")
    if app_state.get("rag_components"):
        app_state["rag_components"].cleanup()

# Create optimized app
app = FastAPI(
    title="Fast Adaptive RAG Learning System",
    description="High-performance RAG API with optimized retrieval",
    version="0.2.0",
    lifespan=fast_lifespan
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
    components = app_state.get("rag_components")
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
    """Fast topic listing with caching"""
    try:
        # Use cached curriculum map
        topics = components.question_selector.get_available_topics()
        return [TopicResponse(**topic) for topic in topics]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing topics: {str(e)}")

@app.post("/api/v1/interaction/start", response_model=QuestionResponse)
async def start_interaction_fast(
    request: LearnerInteractionStartRequest,
    components: FastRAGComponents = Depends(get_fast_components)
):
    """Fast interaction start with async question selection"""
    print(f"🎯 Fast interaction start for learner: {request.learner_id}")
    
    try:
        # Run question selection asynchronously
        next_question_info = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: asyncio.run(components.question_selector.select_next_question(
                learner_id=request.learner_id,
                target_doc_id=request.topic_id
            ))
        )
        
        if not next_question_info or "error" in next_question_info:
            detail_msg = next_question_info.get("error", "Could not select question") if next_question_info else "Could not select question"
            raise HTTPException(status_code=404, detail=detail_msg)
        
        return QuestionResponse(
            question_id=next_question_info["concept_id"],
            doc_id=next_question_info.get("doc_id", "unknown_doc"),
            concept_name=next_question_info["concept_name"],
            question_text=next_question_info["question_text"],
            context_for_evaluation=next_question_info["context_for_evaluation"],
            is_new_concept_context_presented=next_question_info.get("is_new_concept_context_presented", False)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in fast interaction start: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

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
    components = app_state.get("rag_components")
    
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
    components = app_state.get("rag_components")
    if not components:
        return {"error": "Components not initialized"}
    
    return components.get_performance_stats()

@app.post("/api/v1/clear_cache")
async def clear_cache():
    """Clear all caches (useful for testing)"""
    components = app_state.get("rag_components")
    if components and components._retriever:
        components._retriever.clear_cache()
        return {"message": "Cache cleared successfully"}
    return {"message": "No cache to clear"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Fast RAG API Server...")
    uvicorn.run("src.api.fast_api:app", host="0.0.0.0", port=8000, reload=True)