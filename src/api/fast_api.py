# src/api/fast_api.py - Updated with fixed RL integration
import os
import asyncio
import time
from fastapi import FastAPI, HTTPException, Depends, Request, Security, Header
from fastapi.security.api_key import APIKeyHeader
from typing import Dict, Any, Optional, List as PyList
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import datetime
import re
import logging

logger = logging.getLogger(__name__)

from datetime import datetime
from src.rl_engine.rl_question_selector import RLQuestionSelector
from src.rl_engine.reward_system import VoteType
from src.rl_engine.environment import ProfileManagerAdapter
from src.api.feedback_models import InteractionFeedbackRequest, InteractionFeedbackResponse, FeedbackRating

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
from src.retrieval.hybrid_retriever import HybridRetriever

# Import our optimized components
from src.retrieval.retriever import HybridRetriever

class FastRAGComponents:
    """
    Optimized RAG components with pre-loading and async initialization
    Now includes RL integration
    """
    
    def __init__(self):
        print("🚀 Initializing Fast RAG Components with RL...")
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
            self._retriever = HybridRetriever(weaviate_client=self.weaviate_client)
            
            # FIXED: Use improved profile manager
            print("  👤 Initializing improved profile manager...")
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
        
        # FIXED: Initialize RL Question Selector with proper async handling
        self.rl_question_selector = RLQuestionSelector(
            profile_manager=self._profile_manager,
            retriever=self._retriever,
            question_generator=self._question_generator
        )
        
        # Keep old question selector as fallback
        self.question_selector = QuestionSelector(
            profile_manager=ProfileManagerAdapter(self._profile_manager),
            retriever=self._retriever,
            question_generator=self._question_generator
        )
        
        self.answer_handler = AnswerHandler(
            evaluator=self._answer_evaluator,
            tracker=self.knowledge_tracker
        )
        
        print("  ✓ All components initialized")
    
    async def ensure_ready(self):
        """Ensure all components are ready (pre-initialized during startup)"""
        if self.initialization_error:
            raise Exception(f"Components failed to initialize: {self.initialization_error}")
        
        # FIXED: Initialize RL components with graceful error handling
        if hasattr(self, 'rl_question_selector') and not self.rl_question_selector.is_initialized:
            try:
                print("  🤖 Initializing RL Question Selector...")
                await self.rl_question_selector.initialize()
                print("  ✓ RL Question Selector initialized")
            except Exception as e:
                print(f"  ⚠️  Failed to initialize RL Question Selector (non-critical): {e}")
                # Don't fail the entire system if RL fails to initialize
                # The system can still work with rule-based selection
        
        # LLM components are now pre-initialized during startup
        # Only pre-warm if not already done
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
            
            # Load curriculum map for rule-based selector
            try:
                await self.question_selector.initialize()
                print("  ✓ Rule-based question selector initialized")
            except Exception as e:
                print(f"  ⚠️  Failed to initialize rule-based selector: {e}")
            
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
            "components_ready": self.initialization_error is None,
            "rl_initialized": self.rl_question_selector.is_initialized if hasattr(self, 'rl_question_selector') else False
        }
        
        if self._retriever:
            stats.update(self._retriever.get_cache_stats())
        
        # Add RL stats if available
        if hasattr(self, 'rl_question_selector') and self.rl_question_selector.is_initialized:
            try:
                rl_stats = self.rl_question_selector.get_stats()
                stats['rl_stats'] = rl_stats
            except Exception as e:
                logger.warning(f"Failed to get RL stats: {e}")
        
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        if self._profile_manager:
            self._profile_manager.close_db()
        if hasattr(self, 'weaviate_client') and self.weaviate_client is not None:
            try:
                self.weaviate_client.close()
                print("  Weaviate client connection closed.")
            except Exception as e:
                print(f"  Warning: Failed to close Weaviate client: {e}")
        print("  🧹 Resources cleaned up")

    async def _get_context_for_concept(self, concept_id: str, limit: int = 2) -> str:
        """Get context for a concept with proper LaTeX formatting"""
        try:
            chunks = await self.retriever.get_chunks_for_parent_block(concept_id, limit)
            if not chunks:
                return "No context available for this concept."
                
            # Get the topic from the first chunk
            topic = chunks[0].get('doc_id', 'Unknown Topic')
            topic = topic.replace('_', ' ').title()
            
            # Build context with proper LaTeX formatting
            context_parts = []
            for chunk in chunks:
                if chunk.get('chunk_text'):
                    # Clean up any malformed LaTeX and repeated text
                    text = chunk['chunk_text'].strip()
                    # Remove repeated "Definition N" patterns
                    text = re.sub(r'(Definition \d+)\s+\1\s+\1', r'\1', text)
                    # Ensure proper LaTeX math mode formatting
                    text = re.sub(r'\$([^$]+)\$', r'$$\1$$', text)  # Convert inline math to display math
                    context_parts.append(text)
            
            if not context_parts:
                return "No valid context available for this concept."
                
            context = "\n\n".join(context_parts)
            
            # Extract the definition title if present
            title_match = re.search(r'Definition\s+\d+\s+\(([^)]+)\)', context)
            title = title_match.group(1) if title_match else "Concept"
            
            return f"""# {title}

{context}"""
            
        except Exception as e:
            print(f"Error getting context for concept {concept_id}: {e}")
            return "Error retrieving context for this concept."

# API Key security
API_KEY = os.getenv("API_KEY", "test-api-key-12345")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=403, detail="Could not validate credentials"
        )

# Global state
app_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app"""
    print("🚀 Fast API startup...")
    
    # Initialize components
    print("🚀 Initializing Fast RAG Components...")
    components = None
    try:
        components = FastRAGComponents()
        components._init_core_components()
        
        # Pre-initialize LLM components during startup
        print("🤖 Pre-initializing LLM components...")
        components._init_llm_components()
        components._init_dependent_components()
        
        # Store components in app state
        app_state["components"] = components
        app_state["profile_manager"] = components._profile_manager
        app_state["question_selector"] = components.question_selector
        app_state["question_generator"] = components._question_generator
        app_state["answer_evaluator"] = components._answer_evaluator
        app_state["active_interactions"] = {}
        
        # FIXED: Add debugging to verify components are stored
        print(f"✅ Components stored in app_state: {components is not None}")
        print(f"✅ App state keys: {list(app_state.keys())}")
        print(f"✅ Components initialization_error: {components.initialization_error}")
        
        print("✅ Fast startup complete! All components ready.")
        print("   LLM components pre-loaded for fast first requests.")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        import traceback
        traceback.print_exc()
        # Initialize with None values to prevent KeyError
        app_state["components"] = None
        app_state["profile_manager"] = None
        app_state["question_selector"] = None
        app_state["question_generator"] = None
        app_state["answer_evaluator"] = None
        app_state["active_interactions"] = {}
    
    yield
    
    print("🛑 Shutting down...")
    try:
        if "components" in app_state and app_state["components"] is not None:
            app_state["components"].cleanup()
        else:
            print("  No components to cleanup")
    except Exception as e:
        print(f"  Warning: Error during cleanup: {e}")
    print("  🧹 Resources cleaned up")

# Create optimized app
app = FastAPI(
    title="Fast Adaptive RAG Learning System with RL",
    description="High-performance RAG API with RL-powered question selection",
    version="0.3.0",
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
        raise HTTPException(status_code=503, detail="System not ready - components not initialized")
    
    try:
        await components.ensure_ready()
        return components
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"System not ready - {str(e)}")

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
async def list_topics_fast():
    """Get list of available topics"""
    try:
        print("Starting topics endpoint...")
        components = await get_fast_components()
        
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
    learner_id: Optional[str] = None,
    topic_id: Optional[str] = None,
    use_rl: bool = True  # FIXED: Default to True for RL system
):
    """Start a new interaction with a question using RL or rule-based selection"""
    try:
        # Parse request body only if it's not empty
        body = {}
        try:
            body_bytes = await request.body()
            if body_bytes:
                body = await request.json()
        except Exception as e:
            print(f"Warning: Could not parse request body as JSON: {e}")
            body = {}
        
        # Use query parameters if provided, otherwise use body
        learner_id = learner_id or body.get("learner_id", "1")
        topic_id = topic_id or body.get("topic_id")
        use_rl = body.get("use_rl", use_rl)
        
        print(f"🎯 Starting interaction for learner: {learner_id} (RL: {use_rl})")
        
        components = await get_fast_components()
        
        # Get or create learner profile
        profile = await components.profile_manager.get_profile(str(learner_id))
        if not profile:
            await components.profile_manager.create_profile(str(learner_id))
            profile = await components.profile_manager.get_profile(str(learner_id))
        
        # Select topic if not provided
        if not topic_id:
            topics = await list_topics_fast()
            if not topics:
                raise HTTPException(status_code=404, detail="No topics available")
            topic_id = topics[0].doc_id
        
        # Choose question selector based on use_rl parameter and availability
        if use_rl and hasattr(components, 'rl_question_selector') and components.rl_question_selector.is_initialized:
            question_selector = components.rl_question_selector
            print(f"Using RL question selector")
        else:
            question_selector = components.question_selector
            print(f"Using rule-based question selector")
            
            # Initialize rule-based selector if not done yet
            if not hasattr(question_selector, 'curriculum_map') or not question_selector.curriculum_map:
                await question_selector.initialize()
        
        # Get next question using the selected engine
        try:
            question_data = await question_selector.select_next_question(
                learner_id=str(learner_id),
                target_doc_id=topic_id
            )
        except Exception as selector_error:
            print(f"❌ Error in question selector: {selector_error}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Question selection failed: {str(selector_error)}")
        
        if not question_data:
            raise HTTPException(status_code=404, detail="No questions available for this topic")
        
        if "error" in question_data:
            raise HTTPException(status_code=500, detail=question_data["error"])
        
        # For rule-based selector, enhance the response
        if not use_rl or not components.rl_question_selector.is_initialized:
            # Get context for the concept if not already provided
            if not question_data.get("context_for_evaluation"):
                context = await components._get_context_for_concept(question_data["concept_id"])
                question_data["context_for_evaluation"] = context
        
        return QuestionResponse(
            question_id=question_data["concept_id"],
            doc_id=question_data.get("doc_id", topic_id or "unknown_doc"),
            concept_name=question_data.get("concept_name", "Unknown Concept"),
            question_text=question_data["question_text"],
            context_for_evaluation=question_data.get("context_for_evaluation", ""),
            is_new_concept_context_presented=question_data.get("is_new_concept_context_presented", False),
            # Add RL-specific fields if available
            interaction_id=question_data.get("interaction_id"),
            rl_metadata=question_data.get("rl_metadata") or {
                "selected_by_rl": question_data.get("selected_by_rl", False),
                "rl_action": question_data.get("rl_action"),
                "rl_confidence": question_data.get("rl_confidence"),
                "valid_actions_count": question_data.get("valid_actions_count"),
                "total_actions_count": question_data.get("total_actions_count")
            }
        )
        
    except Exception as e:
        print(f"❌ Error in interaction start: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/interaction/next_question", response_model=QuestionResponse)
async def get_next_question(
    learner_id: str = "1",
    topic_id: Optional[str] = None,
    use_rl: bool = True,
    components: FastRAGComponents = Depends(get_fast_components)
):
    """Get the next question for a learner (same as start_interaction but with a different endpoint name)"""
    try:
        print(f"🎯 Getting next question for learner: {learner_id} (RL: {use_rl})")
        
        # Get or create learner profile
        profile = await components.profile_manager.get_profile(str(learner_id))
        if not profile:
            await components.profile_manager.create_profile(str(learner_id))
            profile = await components.profile_manager.get_profile(str(learner_id))
        
        # Select topic if not provided
        if not topic_id:
            topics = await list_topics_fast()
            if not topics:
                raise HTTPException(status_code=404, detail="No topics available")
            topic_id = topics[0].doc_id
        
        # Choose question selector based on use_rl parameter and availability
        if use_rl and hasattr(components, 'rl_question_selector') and components.rl_question_selector.is_initialized:
            question_selector = components.rl_question_selector
            print(f"Using RL question selector")
        else:
            question_selector = components.question_selector
            print(f"Using rule-based question selector")
            
            # Initialize rule-based selector if not done yet
            if not hasattr(question_selector, 'curriculum_map') or not question_selector.curriculum_map:
                await question_selector.initialize()
        
        # Get next question using the selected engine
        try:
            question_data = await question_selector.select_next_question(
                learner_id=str(learner_id),
                target_doc_id=topic_id
            )
        except Exception as selector_error:
            print(f"❌ Error in question selector: {selector_error}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Question selection failed: {str(selector_error)}")
        
        if not question_data:
            raise HTTPException(status_code=404, detail="No questions available for this topic")
        
        if "error" in question_data:
            raise HTTPException(status_code=500, detail=question_data["error"])
        
        # For rule-based selector, enhance the response
        if not use_rl or not components.rl_question_selector.is_initialized:
            # Get context for the concept if not already provided
            if not question_data.get("context_for_evaluation"):
                context = await components._get_context_for_concept(question_data["concept_id"])
                question_data["context_for_evaluation"] = context
        
        return QuestionResponse(
            question_id=question_data["concept_id"],
            doc_id=question_data.get("doc_id", topic_id or "unknown_doc"),
            concept_name=question_data.get("concept_name", "Unknown Concept"),
            question_text=question_data["question_text"],
            context_for_evaluation=question_data.get("context_for_evaluation", ""),
            is_new_concept_context_presented=question_data.get("is_new_concept_context_presented", False),
            # Add RL-specific fields if available
            interaction_id=question_data.get("interaction_id"),
            rl_metadata=question_data.get("rl_metadata") or {
                "selected_by_rl": question_data.get("selected_by_rl", False),
                "rl_action": question_data.get("rl_action"),
                "rl_confidence": question_data.get("rl_confidence"),
                "valid_actions_count": question_data.get("valid_actions_count"),
                "total_actions_count": question_data.get("total_actions_count")
            }
        )
        
    except Exception as e:
        print(f"❌ Error in getting next question: {e}")
        import traceback
        traceback.print_exc()
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

@app.post("/api/v1/interaction/{interaction_id}/rate", response_model=InteractionFeedbackResponse)
async def rate_interaction(
    interaction_id: str,
    request: InteractionFeedbackRequest,
    components: FastRAGComponents = Depends(get_fast_components)
):
    """
    Rate an interaction with engagement feedback
    This triggers the reward calculation for the RL system
    """
    try:
        logger.info(f"Received rating for interaction {interaction_id}: {request.rating.value}")
        
        # Convert rating to VoteType
        vote_type = VoteType.UPVOTE if request.rating == FeedbackRating.UP else VoteType.DOWNVOTE
        
        # Check if RL system is available and this was an RL interaction
        if not components.rl_question_selector.is_initialized:
            return InteractionFeedbackResponse(
                interaction_id=interaction_id,
                learner_id="unknown",
                rating=request.rating,
                reward_calculated=False,
                message="RL system not available for this interaction"
            )
        
        # Get interaction data to find learner and concept
        interaction_data = components.rl_question_selector.reward_manager.interaction_tracker.get_interaction_data(interaction_id)
        
        if not interaction_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Interaction {interaction_id} not found"
            )
        
        learner_id = interaction_data['learner_id']
        concept_id = interaction_data['concept_id']
        
        # Process the feedback
        success = await components.rl_question_selector.process_feedback(
            interaction_id=interaction_id,
            learner_id=learner_id,
            concept_id=concept_id,
            vote_type=vote_type
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to process feedback"
            )
        
        # Get the completed interaction data with reward
        completed_data = components.rl_question_selector.reward_manager.interaction_tracker.get_interaction_data(interaction_id)
        
        reward_components = None
        reward_total = None
        
        if completed_data and completed_data.get('completed'):
            reward_components = completed_data['reward_components'].to_dict()
            reward_total = reward_components['total_reward']
        
        return InteractionFeedbackResponse(
            interaction_id=interaction_id,
            learner_id=learner_id,
            rating=request.rating,
            reward_calculated=success,
            reward_total=reward_total,
            reward_components=reward_components,
            message="Feedback processed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing interaction rating: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing feedback: {str(e)}"
        )

@app.get("/api/v1/rl/stats")
async def get_rl_stats(components: FastRAGComponents = Depends(get_fast_components)):
    """Get RL system statistics"""
    try:
        if not components.rl_question_selector.is_initialized:
            return {"error": "RL system not initialized", "initialized": False}
        
        stats = components.rl_question_selector.get_stats()
        return {
            "rl_system": stats,
            "timestamp": datetime.now().isoformat(),
            "initialized": True
        }
        
    except Exception as e:
        logger.error(f"Error getting RL stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/rl/training/enable")
async def enable_rl_training(components: FastRAGComponents = Depends(get_fast_components)):
    """Enable RL training mode"""
    try:
        if not components.rl_question_selector.is_initialized:
            raise HTTPException(status_code=503, detail="RL system not initialized")
        
        components.rl_question_selector.enable_training_mode()
        return {"message": "RL training mode enabled", "training_mode": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/rl/training/disable")
async def disable_rl_training(components: FastRAGComponents = Depends(get_fast_components)):
    """Disable RL training mode"""
    try:
        if not components.rl_question_selector.is_initialized:
            raise HTTPException(status_code=503, detail="RL system not initialized")
        
        components.rl_question_selector.disable_training_mode()
        return {"message": "RL training mode disabled", "training_mode": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/rl/model/save")
async def save_rl_model(components: FastRAGComponents = Depends(get_fast_components)):
    """Save the current RL model"""
    try:
        if not components.rl_question_selector.is_initialized:
            raise HTTPException(status_code=503, detail="RL system not initialized")
        
        components.rl_question_selector.save_model()
        return {"message": "RL model saved successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/learner/{learner_id}/session")
async def get_learner_session(
    learner_id: str,
    components: FastRAGComponents = Depends(get_fast_components)
):
    """Get detailed session information for a learner"""
    try:
        session_info = components.rl_question_selector.get_learner_session_info(learner_id)
        return session_info
    except Exception as e:
        logger.error(f"Error getting learner session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/learner/{learner_id}/session/reset")
async def reset_learner_session(
    learner_id: str,
    components: FastRAGComponents = Depends(get_fast_components)
):
    """Reset session data for a learner"""
    try:
        success = await components.rl_question_selector.reset_learner_session(learner_id)
        if success:
            return {"message": f"Session reset for learner {learner_id}", "success": True}
        else:
            return {"message": f"Failed to reset session for learner {learner_id}", "success": False}
    except Exception as e:
        logger.error(f"Error resetting learner session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check_fast():
    """Health check with performance metrics"""
    components = app_state.get("components")
    
    health_data = {
        "status": "healthy" if components and not components.initialization_error else "degraded",
        "timestamp": datetime.now().isoformat()
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

@app.get("/api/v1/test")
async def test_endpoint():
    """Simple test endpoint to verify the system is working"""
    return {
        "status": "working",
        "message": "Backend is responding correctly",
        "timestamp": datetime.now().isoformat(),
        "version": "0.3.0 with RL integration"
    }

@app.get("/api/v1/debug/database")
async def debug_database():
    """Debug endpoint to check database content"""
    try:
        components = await get_fast_components()
        
        # Get a sample of documents to see their structure
        sample_docs = await components.retriever.get_all_documents(limit=10)
        
        # Extract unique parent_block_ids
        parent_block_ids = set()
        concept_names = set()
        for doc in sample_docs:
            if doc.get('parent_block_id'):
                parent_block_ids.add(doc.get('parent_block_id'))
            if doc.get('concept_name'):
                concept_names.add(doc.get('concept_name'))
        
        return {
            "total_docs_sampled": len(sample_docs),
            "sample_docs": sample_docs[:3],  # Show first 3
            "unique_parent_block_ids": list(parent_block_ids)[:10],  # Show first 10
            "unique_concept_names": list(concept_names)[:10],  # Show first 10
            "curriculum_map_size": len(components.question_selector.curriculum_map) if hasattr(components.question_selector, 'curriculum_map') else 0
        }
    except Exception as e:
        return {"error": str(e), "type": str(type(e))}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Fast RAG API Server with RL...")
    uvicorn.run("src.api.fast_api:app", host="0.0.0.0", port=8000, reload=True)