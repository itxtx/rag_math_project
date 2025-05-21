# src/api/main_api.py
import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Query 
from typing import Dict, Any, Optional, List as PyList 
from pydantic import BaseModel 
from contextlib import asynccontextmanager # For lifespan events

from src import config
from src.api.models import ( 
    LearnerInteractionStartRequest, 
    QuestionResponse, 
    AnswerSubmissionRequest, AnswerSubmissionResponse,
    EvaluationResult, ErrorResponse
)

# Import RAG system components
from src.data_ingestion import vector_store_manager 
from src.learner_model.profile_manager import LearnerProfileManager
from src.retrieval.retriever import Retriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.evaluation.answer_evaluator import AnswerEvaluator
from src.learner_model.knowledge_tracker import KnowledgeTracker
from src.interaction.answer_handler import AnswerHandler
from src.adaptive_engine.question_selector import QuestionSelector

# --- Global RAG System Components ---
# This class will hold our initialized components.
class RAGSystemComponents:
    def __init__(self):
        print("Initializing RAG System Components for API...")
        try:
            self.weaviate_client = vector_store_manager.get_weaviate_client()
            vector_store_manager.create_weaviate_schema(self.weaviate_client)
            print("Weaviate client and schema ensured.")

            self.profile_manager_singleton = LearnerProfileManager() 
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
            # Store the exception to be re-raised by the dependency if needed
            self.initialization_error = e
            # raise RuntimeError(f"Failed to initialize RAG components: {e}") from e
        else:
            self.initialization_error = None


# This dictionary will hold our components. It's populated by the lifespan manager.
# Using a dictionary allows for a simple way to pass state managed by lifespan.
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # Code to run on startup
    print("API Lifespan: Startup sequence initiated.")
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        print(f"API Lifespan: Found .env file at {dotenv_path}, loading.")
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
        print("API Lifespan: Environment variables from .env potentially loaded.")
    else:
        print(f"API Lifespan: .env file not found at {dotenv_path}.")
    
    try:
        app_state["rag_components"] = RAGSystemComponents()
        if app_state["rag_components"].initialization_error:
             print(f"API Lifespan: ERROR during RAG component initialization - {app_state['rag_components'].initialization_error}")
             # Decide if the app should still start or raise an error to prevent it.
             # For now, it will start but get_rag_components will raise 503.
    except Exception as e:
        print(f"API Lifespan: CRITICAL ERROR during startup RAG component initialization - {e}")
        app_state["rag_components"] = None # Ensure it's None if init fails catastrophically

    print("API Lifespan: Startup sequence complete.")
    yield
    # Code to run on shutdown
    print("API Lifespan: Shutdown sequence initiated.")
    components = app_state.get("rag_components")
    if components and components.profile_manager_singleton:
        components.profile_manager_singleton.close_db()
    print("API Lifespan: Resources cleaned up. Shutdown sequence complete.")


# --- FastAPI App Initialization with Lifespan ---
app = FastAPI(
    title="Adaptive RAG Learning System API",
    description="API for interacting with the RAG-based adaptive learning system.",
    version="0.1.0",
    lifespan=lifespan # Register the lifespan context manager
)


# --- Dependency for getting RAG components ---
async def get_rag_components() -> RAGSystemComponents:
    components = app_state.get("rag_components")
    if components is None or components.initialization_error:
        error_detail = "RAG system components not available or failed to initialize."
        if components and components.initialization_error:
            error_detail += f" Error: {components.initialization_error}"
        raise HTTPException(status_code=503, detail=error_detail)
    return components

# --- API Endpoints ---

class TopicResponse(BaseModel): 
    topic_id: str
    source_file: str

@app.get("/api/v1/topics", 
           response_model=PyList[TopicResponse], 
           summary="List Available Topics",
           description="Retrieves a list of available top-level topics (documents) from the knowledge base.",
           responses={503: {"model": ErrorResponse, "description": "Service unavailable."}})
async def list_available_topics(
    components: RAGSystemComponents = Depends(get_rag_components)
):
    try:
        topics = components.question_selector.get_available_topics()
        if not topics: # Attempt to reload if curriculum map is empty
            print("API /topics: Curriculum map was empty, attempting reload.")
            components.question_selector._load_curriculum_map() 
            topics = components.question_selector.get_available_topics()
        return [TopicResponse(**topic) for topic in topics]
    except Exception as e:
        print(f"API Error in /topics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error listing topics: {str(e)}")


@app.post("/api/v1/interaction/start", 
            response_model=QuestionResponse, 
            summary="Start Learner Interaction (Adaptive or by Topic)",
            responses={
                404: {"model": ErrorResponse, "description": "No suitable question found."},
                503: {"model": ErrorResponse, "description": "Service unavailable."}
            })
async def start_learner_interaction(
    request: LearnerInteractionStartRequest, 
    components: RAGSystemComponents = Depends(get_rag_components)
):
    print(f"API: Received request to start interaction for learner_id: {request.learner_id}, topic_id: {request.topic_id}")
    try:
        next_question_info = await components.question_selector.select_next_question(
            learner_id=request.learner_id,
            target_doc_id=request.topic_id 
        )
        if not next_question_info:
            detail_msg = "Could not select a next question."
            if request.topic_id:
                detail_msg += f" No suitable questions found for topic '{request.topic_id}' or learner has mastered it."
            else:
                detail_msg += " Knowledge base might be empty or learner has mastered all available concepts."
            raise HTTPException(status_code=404, detail=detail_msg)
        
        return QuestionResponse(
            question_id=next_question_info["concept_id"], 
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
            responses={503: {"model": ErrorResponse, "description": "Service unavailable."}})
async def submit_learner_answer(
    request: AnswerSubmissionRequest, 
    components: RAGSystemComponents = Depends(get_rag_components)
):
    print(f"API: Received answer submission from learner_id: {request.learner_id} for question_id: {request.question_id}")
    try:
        handler_response = await components.answer_handler.submit_answer(
            learner_id=request.learner_id,
            question_id=request.question_id,
            question_text=request.question_text,
            retrieved_context=request.context_for_evaluation,
            learner_answer=request.learner_answer
        )
        eval_result = EvaluationResult( 
            accuracy_score=handler_response.get("accuracy_score", 0.0),
            feedback=handler_response.get("feedback", "Evaluation feedback not available."),
            correct_answer=handler_response.get("correct_answer") 
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

@app.get("/api/v1/health", summary="Health Check")
async def health_check():
    import datetime 
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    import datetime 
    print("Starting FastAPI server with Uvicorn...")
    dotenv_path_main = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path_main):
        print(f"Uvicorn Main: Found .env file at {dotenv_path_main}, loading.")
        from dotenv import load_dotenv
        load_dotenv(dotenv_path_main)
    
    uvicorn.run("src.api.main_api:app", host="0.0.0.0", port=8000, reload=True)
