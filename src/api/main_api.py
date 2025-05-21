import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import uvicorn


# Assuming your project root is the parent of 'src' and is in PYTHONPATH
# or you run this with `python -m src.api.main_api` from project root.
from src import config
from src.api.models import (
    LearnerInteractionStartRequest, QuestionResponse,
    AnswerSubmissionRequest, AnswerSubmissionResponse,
    EvaluationResult, ErrorResponse
)

# Import RAG system components
from src.data_ingestion import vector_store_manager # For Weaviate client
from src.learner_model.profile_manager import LearnerProfileManager
from src.retrieval.retriever import Retriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.evaluation.answer_evaluator import AnswerEvaluator
from src.learner_model.knowledge_tracker import KnowledgeTracker
from src.interaction.answer_handler import AnswerHandler
from src.adaptive_engine.question_selector import QuestionSelector

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Adaptive RAG Learning System API",
    description="API for interacting with the RAG-based adaptive learning system.",
    version="0.1.0"
)

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",  # Allow your frontend to access the backend
    "http://127.0.0.1:3000",  # Another common localhost address
    # You can add other origins if your frontend will be deployed elsewhere
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Global RAG System Components ---
# These will be initialized once when the application starts.
# This is a simplified approach for a single-process application.
# For production, consider dependency injection frameworks or more robust lifecycle management.

class RAGSystemComponents:
    def __init__(self):
        print("Initializing RAG System Components for API...")
        try:
            self.weaviate_client = vector_store_manager.get_weaviate_client()
            # Ensure schema exists on startup
            vector_store_manager.create_weaviate_schema(self.weaviate_client)
            print("Weaviate client and schema ensured.")

            self.profile_manager_singleton = LearnerProfileManager() # Uses default DB path from config
            print("LearnerProfileManager initialized.")

            self.retriever = Retriever(
                weaviate_client=self.weaviate_client,
                weaviate_class_name=vector_store_manager.WEAVIATE_CLASS_NAME
            )
            print("Retriever initialized.")

            self.rag_question_generator = RAGQuestionGenerator()
            print("RAGQuestionGenerator initialized.")

            self.question_selector = QuestionSelector(
                profile_manager=self.profile_manager_singleton,
                retriever=self.retriever,
                question_generator=self.rag_question_generator
            )
            print("QuestionSelector initialized.")

            self.answer_evaluator = AnswerEvaluator()
            print("AnswerEvaluator initialized.")

            self.knowledge_tracker = KnowledgeTracker(
                profile_manager=self.profile_manager_singleton
            )
            print("KnowledgeTracker initialized.")

            self.answer_handler = AnswerHandler(
                evaluator=self.answer_evaluator,
                tracker=self.knowledge_tracker
            )
            print("AnswerHandler initialized.")
            print("RAG System Components successfully initialized for API.")

        except Exception as e:
            print(f"FATAL ERROR: Could not initialize RAG system components: {e}")
            import traceback
            traceback.print_exc()
            # In a real app, you might want to prevent startup or enter a degraded mode.
            # For now, some components might be None if initialization fails.
            # FastAPI will likely fail to start if these are used in Depends and are None.
            raise RuntimeError(f"Failed to initialize RAG components: {e}") from e

# Global instance of components
# This simple global instance is okay for demonstration.
# For more complex apps, use FastAPI's dependency injection with `Depends`.
# We will use a simple dependency function for now.
rag_components_instance: Optional[RAGSystemComponents] = None

@app.on_event("startup")
async def startup_event():
    """Load .env and initialize RAG components on application startup."""
    global rag_components_instance
    # Load .env (assuming it's in project root)
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        print(f"API Startup: Found .env file at {dotenv_path}, loading.")
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
        print("API Startup: Environment variables from .env potentially loaded.")
    else:
        print(f"API Startup: .env file not found at {dotenv_path}.")
    
    # Initialize components
    try:
        rag_components_instance = RAGSystemComponents()
    except RuntimeError as e:
        print(f"API Startup: ERROR - {e}")
        # Allow app to start but endpoints might fail if components are needed.
        # A better approach for production would be to ensure components load or exit.
        pass


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    global rag_components_instance
    if rag_components_instance and rag_components_instance.profile_manager_singleton:
        rag_components_instance.profile_manager_singleton.close_db()
    print("API Shutdown: Resources cleaned up.")


# --- Dependency for getting RAG components ---
async def get_rag_components() -> RAGSystemComponents:
    if rag_components_instance is None:
        # This should ideally not happen if startup_event worked.
        # But as a fallback, try to initialize. This is not ideal for production.
        print("WARNING: RAG components not initialized at startup, attempting ad-hoc initialization.")
        try:
            return RAGSystemComponents() # This would create a new instance, not the global one.
                                         # Better to raise an error if not initialized.
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"RAG system components not available: {e}")
    return rag_components_instance


# --- API Endpoints ---

@app.post("/api/v1/interaction/start", 
            response_model=QuestionResponse,
            summary="Start Learner Interaction",
            description="Retrieves the next question for a given learner based on their profile and the curriculum.",
            responses={503: {"model": ErrorResponse, "description": "Service unavailable or RAG components failed to initialize."}})
async def start_learner_interaction(
    request: LearnerInteractionStartRequest,
    components: RAGSystemComponents = Depends(get_rag_components)
):
    """
    Endpoint to get the next question for a learner.
    """
    print(f"API: Received request to start interaction for learner_id: {request.learner_id}")
    try:
        next_question_info = await components.question_selector.select_next_question(request.learner_id)
        if not next_question_info:
            raise HTTPException(status_code=404, detail="Could not select a next question for the learner. Knowledge base might be empty or learner has mastered all available concepts.")
        
        return QuestionResponse(
            question_id=next_question_info["concept_id"], # concept_id is used as question_id
            concept_name=next_question_info["concept_name"],
            question_text=next_question_info["question_text"],
            context_for_evaluation=next_question_info["context_for_evaluation"]
        )
    except HTTPException as http_exc: # Re-raise FastAPI's HTTPException
        raise http_exc
    except Exception as e:
        print(f"API Error in /interaction/start: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error selecting question: {str(e)}")


@app.post("/api/v1/interaction/submit_answer", 
            response_model=AnswerSubmissionResponse,
            summary="Submit Learner's Answer",
            description="Allows a learner to submit an answer to a question for evaluation and profile update.",
            responses={503: {"model": ErrorResponse, "description": "Service unavailable or RAG components failed to initialize."}})
async def submit_learner_answer(
    request: AnswerSubmissionRequest,
    components: RAGSystemComponents = Depends(get_rag_components)
):
    """
    Endpoint for learner to submit an answer.
    """
    print(f"API: Received answer submission from learner_id: {request.learner_id} for question_id: {request.question_id}")
    try:
        handler_response = await components.answer_handler.submit_answer(
            learner_id=request.learner_id,
            question_id=request.question_id, # This is the concept_id
            question_text=request.question_text,
            retrieved_context=request.context_for_evaluation,
            learner_answer=request.learner_answer
        )
        
        # The handler_response contains accuracy_score, feedback, and now 'correct answer'.
        # IMPORTANT: Ensure your EvaluationResult Pydantic model (in src/api/models.py)
        # has a field named 'correct_answer' (e.g., correct_answer: Optional[str] = None)
        eval_result = EvaluationResult(
            accuracy_score=handler_response.get("accuracy_score", 0.0),
            feedback=handler_response.get("feedback", "Evaluation feedback not available."),
            # Changed from "correct_answer" to "correct answer" to match LLM response
            correct_answer=handler_response.get("correct answer")
        )
        
        return AnswerSubmissionResponse(
            learner_id=request.learner_id,
            question_id=request.question_id,
            evaluation=eval_result
        )
    except Exception as e:
        print(f"API Error in /interaction/submit_answer: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error submitting answer: {str(e)}")

@app.get("/api/v1/health", summary="Health Check", description="Simple health check endpoint.")
async def health_check():
    return {"status": "healthy", "timestamp": asyncio.to_thread(os.times)}


if __name__ == "__main__":
    # This allows running the API directly using Uvicorn for development.
    # Install Uvicorn: pip install uvicorn[standard]
    # Run from project root: python -m src.api.main_api
    # Or if this file is run directly: uvicorn src.api.main_api:app --reload

    print("Starting FastAPI server with Uvicorn...")
    # Ensure .env is loaded if running this file directly for Uvicorn
    dotenv_path_main = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path_main):
        print(f"Uvicorn Main: Found .env file at {dotenv_path_main}, loading.")
        from dotenv import load_dotenv
        load_dotenv(dotenv_path_main)
    
    uvicorn.run("src.api.main_api:app", host="0.0.0.0", port=8000, reload=True)
