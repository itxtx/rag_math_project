import pytest
from pydantic import ValidationError
from src.api.models import (
    TopicResponse,
    LearnerInteractionStartRequest,
    QuestionResponse,
    AnswerSubmissionRequest,
    EvaluationResult,
    AnswerSubmissionResponse,
    ErrorResponse
)

# --- TopicResponse Tests ---

def test_topic_response_valid():
    """Test valid TopicResponse creation."""
    data = {
        "doc_id": "test_doc_123",
        "title": "Test Topic",
        "description": "A test topic description",
        "concepts": ["concept1", "concept2", "concept3"]
    }
    
    topic = TopicResponse(**data)
    assert topic.doc_id == "test_doc_123"
    assert topic.title == "Test Topic"
    assert topic.description == "A test topic description"
    assert topic.concepts == ["concept1", "concept2", "concept3"]

def test_topic_response_minimal():
    """Test TopicResponse with minimal required fields."""
    data = {
        "doc_id": "test_doc_123"
    }
    
    topic = TopicResponse(**data)
    assert topic.doc_id == "test_doc_123"
    assert topic.title is None
    assert topic.description is None
    assert topic.concepts == []

def test_topic_response_missing_required_field():
    """Test TopicResponse with missing required doc_id field."""
    data = {
        "title": "Test Topic",
        "description": "A test topic description",
        "concepts": ["concept1"]
    }
    
    with pytest.raises(ValidationError) as exc_info:
        TopicResponse(**data)
    
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("doc_id",)
    assert errors[0]["type"] == "missing"

def test_topic_response_empty_concepts():
    """Test TopicResponse with empty concepts list."""
    data = {
        "doc_id": "test_doc_123",
        "concepts": []
    }
    
    topic = TopicResponse(**data)
    assert topic.concepts == []

def test_topic_response_invalid_concepts_type():
    """Test TopicResponse with invalid concepts type."""
    data = {
        "doc_id": "test_doc_123",
        "concepts": "not_a_list"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        TopicResponse(**data)
    
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("concepts",)

# --- LearnerInteractionStartRequest Tests ---

def test_learner_interaction_start_request_valid():
    """Test valid LearnerInteractionStartRequest creation."""
    data = {
        "learner_id": "learner_123",
        "topic_id": "topic_456"
    }
    
    request = LearnerInteractionStartRequest(**data)
    assert request.learner_id == "learner_123"
    assert request.topic_id == "topic_456"

def test_learner_interaction_start_request_without_topic():
    """Test LearnerInteractionStartRequest without optional topic_id."""
    data = {
        "learner_id": "learner_123"
    }
    
    request = LearnerInteractionStartRequest(**data)
    assert request.learner_id == "learner_123"
    assert request.topic_id is None

def test_learner_interaction_start_request_missing_learner_id():
    """Test LearnerInteractionStartRequest with missing required learner_id."""
    data = {
        "topic_id": "topic_456"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        LearnerInteractionStartRequest(**data)
    
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("learner_id",)
    assert errors[0]["type"] == "missing"

def test_learner_interaction_start_request_empty_learner_id():
    """Test LearnerInteractionStartRequest with empty learner_id."""
    data = {
        "learner_id": "",
        "topic_id": "topic_456"
    }
    
    # Empty string should be valid for string fields
    request = LearnerInteractionStartRequest(**data)
    assert request.learner_id == ""

# --- QuestionResponse Tests ---

def test_question_response_valid():
    """Test valid QuestionResponse creation."""
    data = {
        "question_id": "question_123",
        "doc_id": "doc_456",
        "concept_name": "Derivatives",
        "question_text": "What is the derivative of x^2?",
        "context_for_evaluation": "The derivative of x^2 is 2x.",
        "is_new_concept_context_presented": True
    }
    
    question = QuestionResponse(**data)
    assert question.question_id == "question_123"
    assert question.doc_id == "doc_456"
    assert question.concept_name == "Derivatives"
    assert question.question_text == "What is the derivative of x^2?"
    assert question.context_for_evaluation == "The derivative of x^2 is 2x."
    assert question.is_new_concept_context_presented is True

def test_question_response_with_defaults():
    """Test QuestionResponse with default values."""
    data = {
        "question_id": "question_123",
        "doc_id": "doc_456",
        "concept_name": "Derivatives",
        "question_text": "What is the derivative of x^2?",
        "context_for_evaluation": "The derivative of x^2 is 2x."
    }
    
    question = QuestionResponse(**data)
    assert question.is_new_concept_context_presented is False

def test_question_response_missing_required_fields():
    """Test QuestionResponse with missing required fields."""
    data = {
        "question_id": "question_123",
        "doc_id": "doc_456"
        # Missing concept_name, question_text, context_for_evaluation
    }
    
    with pytest.raises(ValidationError) as exc_info:
        QuestionResponse(**data)
    
    errors = exc_info.value.errors()
    assert len(errors) == 3
    error_fields = [error["loc"][0] for error in errors]
    assert "concept_name" in error_fields
    assert "question_text" in error_fields
    assert "context_for_evaluation" in error_fields

def test_question_response_empty_strings():
    """Test QuestionResponse with empty string values."""
    data = {
        "question_id": "",
        "doc_id": "",
        "concept_name": "",
        "question_text": "",
        "context_for_evaluation": ""
    }
    
    # Empty strings should be valid for string fields
    question = QuestionResponse(**data)
    assert question.question_id == ""
    assert question.doc_id == ""
    assert question.concept_name == ""
    assert question.question_text == ""
    assert question.context_for_evaluation == ""

# --- AnswerSubmissionRequest Tests ---

def test_answer_submission_request_valid():
    """Test valid AnswerSubmissionRequest creation."""
    data = {
        "learner_id": "learner_123",
        "question_id": "question_456",
        "doc_id": "doc_789",
        "question_text": "What is the derivative of x^2?",
        "context_for_evaluation": "The derivative of x^2 is 2x.",
        "learner_answer": "2x"
    }
    
    request = AnswerSubmissionRequest(**data)
    assert request.learner_id == "learner_123"
    assert request.question_id == "question_456"
    assert request.doc_id == "doc_789"
    assert request.question_text == "What is the derivative of x^2?"
    assert request.context_for_evaluation == "The derivative of x^2 is 2x."
    assert request.learner_answer == "2x"

def test_answer_submission_request_missing_fields():
    """Test AnswerSubmissionRequest with missing required fields."""
    data = {
        "learner_id": "learner_123",
        "question_id": "question_456"
        # Missing doc_id, question_text, context_for_evaluation, learner_answer
    }
    
    with pytest.raises(ValidationError) as exc_info:
        AnswerSubmissionRequest(**data)
    
    errors = exc_info.value.errors()
    assert len(errors) == 4
    error_fields = [error["loc"][0] for error in errors]
    assert "doc_id" in error_fields
    assert "question_text" in error_fields
    assert "context_for_evaluation" in error_fields
    assert "learner_answer" in error_fields

def test_answer_submission_request_empty_answer():
    """Test AnswerSubmissionRequest with empty learner answer."""
    data = {
        "learner_id": "learner_123",
        "question_id": "question_456",
        "doc_id": "doc_789",
        "question_text": "What is the derivative of x^2?",
        "context_for_evaluation": "The derivative of x^2 is 2x.",
        "learner_answer": ""
    }
    
    # Empty answer should be valid
    request = AnswerSubmissionRequest(**data)
    assert request.learner_answer == ""

def test_answer_submission_request_long_answer():
    """Test AnswerSubmissionRequest with very long learner answer."""
    long_answer = "x" * 10000  # Very long answer
    
    data = {
        "learner_id": "learner_123",
        "question_id": "question_456",
        "doc_id": "doc_789",
        "question_text": "What is the derivative of x^2?",
        "context_for_evaluation": "The derivative of x^2 is 2x.",
        "learner_answer": long_answer
    }
    
    # Long answer should be valid
    request = AnswerSubmissionRequest(**data)
    assert request.learner_answer == long_answer

# --- EvaluationResult Tests ---

def test_evaluation_result_valid():
    """Test valid EvaluationResult creation."""
    data = {
        "accuracy_score": 0.85,
        "feedback": "Good answer! The derivative of x^2 is indeed 2x.",
        "correct_answer": "2x"
    }
    
    evaluation = EvaluationResult(**data)
    assert evaluation.accuracy_score == 0.85
    assert evaluation.feedback == "Good answer! The derivative of x^2 is indeed 2x."
    assert evaluation.correct_answer == "2x"

def test_evaluation_result_without_correct_answer():
    """Test EvaluationResult without optional correct_answer."""
    data = {
        "accuracy_score": 0.85,
        "feedback": "Good answer!"
    }
    
    evaluation = EvaluationResult(**data)
    assert evaluation.accuracy_score == 0.85
    assert evaluation.feedback == "Good answer!"
    assert evaluation.correct_answer is None

def test_evaluation_result_missing_required_fields():
    """Test EvaluationResult with missing required fields."""
    data = {
        "accuracy_score": 0.85
        # Missing feedback
    }
    
    with pytest.raises(ValidationError) as exc_info:
        EvaluationResult(**data)
    
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("feedback",)
    assert errors[0]["type"] == "missing"

def test_evaluation_result_invalid_accuracy_score():
    """Test EvaluationResult with invalid accuracy_score values."""
    # Test score below 0
    with pytest.raises(ValidationError) as exc_info:
        EvaluationResult(accuracy_score=-0.1, feedback="Test")
    
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("accuracy_score",)
    
    # Test score above 1
    with pytest.raises(ValidationError) as exc_info:
        EvaluationResult(accuracy_score=1.1, feedback="Test")
    
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("accuracy_score",)

def test_evaluation_result_boundary_values():
    """Test EvaluationResult with boundary values for accuracy_score."""
    # Test minimum valid score
    evaluation = EvaluationResult(accuracy_score=0.0, feedback="Test")
    assert evaluation.accuracy_score == 0.0
    
    # Test maximum valid score
    evaluation = EvaluationResult(accuracy_score=1.0, feedback="Test")
    assert evaluation.accuracy_score == 1.0

def test_evaluation_result_wrong_accuracy_score_type():
    """Test EvaluationResult with wrong type for accuracy_score."""
    with pytest.raises(ValidationError):
        EvaluationResult(
            feedback="Good effort",
            is_correct=True,
            accuracy_score="not_a_float",
            correct_answer="The correct answer."
        )

# --- AnswerSubmissionResponse Tests ---

def test_answer_submission_response_valid():
    """Test valid AnswerSubmissionResponse creation."""
    evaluation_data = {
        "accuracy_score": 0.85,
        "feedback": "Good answer!",
        "correct_answer": "2x"
    }
    
    data = {
        "learner_id": "learner_123",
        "question_id": "question_456",
        "doc_id": "doc_789",
        "evaluation": evaluation_data
    }
    
    response = AnswerSubmissionResponse(**data)
    assert response.learner_id == "learner_123"
    assert response.question_id == "question_456"
    assert response.doc_id == "doc_789"
    assert response.evaluation.accuracy_score == 0.85
    assert response.evaluation.feedback == "Good answer!"
    assert response.evaluation.correct_answer == "2x"

def test_answer_submission_response_missing_fields():
    """Test AnswerSubmissionResponse with missing required fields."""
    evaluation_data = {
        "accuracy_score": 0.85,
        "feedback": "Good answer!"
    }
    
    data = {
        "learner_id": "learner_123",
        "evaluation": evaluation_data
        # Missing question_id, doc_id
    }
    
    with pytest.raises(ValidationError) as exc_info:
        AnswerSubmissionResponse(**data)
    
    errors = exc_info.value.errors()
    assert len(errors) == 2
    error_fields = [error["loc"][0] for error in errors]
    assert "question_id" in error_fields
    assert "doc_id" in error_fields

def test_answer_submission_response_invalid_evaluation():
    """Test AnswerSubmissionResponse with invalid evaluation data."""
    data = {
        "learner_id": "learner_123",
        "question_id": "question_456",
        "doc_id": "doc_789",
        "evaluation": {
            "accuracy_score": 1.5,  # Invalid score > 1
            "feedback": "Good answer!"
        }
    }
    
    with pytest.raises(ValidationError) as exc_info:
        AnswerSubmissionResponse(**data)
    
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("evaluation", "accuracy_score")

# --- ErrorResponse Tests ---

def test_error_response_valid():
    """Test valid ErrorResponse creation."""
    data = {
        "detail": "An error occurred while processing the request."
    }
    
    error = ErrorResponse(**data)
    assert error.detail == "An error occurred while processing the request."

def test_error_response_missing_detail():
    """Test ErrorResponse with missing required detail field."""
    data = {}
    
    with pytest.raises(ValidationError) as exc_info:
        ErrorResponse(**data)
    
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("detail",)
    assert errors[0]["type"] == "missing"

def test_error_response_empty_detail():
    """Test ErrorResponse with empty detail string."""
    data = {
        "detail": ""
    }
    
    error = ErrorResponse(**data)
    assert error.detail == ""

# --- Model Serialization Tests ---

def test_topic_response_serialization():
    """Test TopicResponse serialization to dict."""
    data = {
        "doc_id": "test_doc_123",
        "title": "Test Topic",
        "description": "A test topic description",
        "concepts": ["concept1", "concept2"]
    }
    
    topic = TopicResponse(**data)
    serialized = topic.model_dump()
    
    assert serialized["doc_id"] == "test_doc_123"
    assert serialized["title"] == "Test Topic"
    assert serialized["description"] == "A test topic description"
    assert serialized["concepts"] == ["concept1", "concept2"]

def test_question_response_serialization():
    """Test QuestionResponse serialization to dict."""
    data = {
        "question_id": "question_123",
        "doc_id": "doc_456",
        "concept_name": "Derivatives",
        "question_text": "What is the derivative of x^2?",
        "context_for_evaluation": "The derivative of x^2 is 2x.",
        "is_new_concept_context_presented": True
    }
    
    question = QuestionResponse(**data)
    serialized = question.model_dump()
    
    assert serialized["question_id"] == "question_123"
    assert serialized["doc_id"] == "doc_456"
    assert serialized["concept_name"] == "Derivatives"
    assert serialized["question_text"] == "What is the derivative of x^2?"
    assert serialized["context_for_evaluation"] == "The derivative of x^2 is 2x."
    assert serialized["is_new_concept_context_presented"] is True

def test_evaluation_result_serialization():
    """Test EvaluationResult serialization to dict."""
    data = {
        "accuracy_score": 0.85,
        "feedback": "Good answer!",
        "correct_answer": "2x"
    }
    
    evaluation = EvaluationResult(**data)
    serialized = evaluation.model_dump()
    
    assert serialized["accuracy_score"] == 0.85
    assert serialized["feedback"] == "Good answer!"
    assert serialized["correct_answer"] == "2x"

# --- Edge Cases and Boundary Tests ---

def test_models_with_unicode_characters():
    """Test models with unicode characters in strings."""
    data = {
        "doc_id": "test_doc_123",
        "title": "Tópico de Teste 🧪",
        "description": "A test topic with emojis 🎯 and unicode éñ",
        "concepts": ["concepto", "concepté"]
    }
    
    topic = TopicResponse(**data)
    assert topic.title == "Tópico de Teste 🧪"
    assert topic.description == "A test topic with emojis 🎯 and unicode éñ"
    assert topic.concepts == ["concepto", "concepté"]

def test_models_with_special_characters():
    """Test models with special characters in strings."""
    data = {
        "learner_id": "learner_123",
        "question_id": "question_456",
        "doc_id": "doc_789",
        "question_text": "What is f'(x) when f(x) = x² + 3x + 1?",
        "context_for_evaluation": "The derivative f'(x) = 2x + 3",
        "learner_answer": "f'(x) = 2x + 3"
    }
    
    request = AnswerSubmissionRequest(**data)
    assert request.question_text == "What is f'(x) when f(x) = x² + 3x + 1?"
    assert request.context_for_evaluation == "The derivative f'(x) = 2x + 3"
    assert request.learner_answer == "f'(x) = 2x + 3"

def test_models_with_whitespace():
    """Test models with leading/trailing whitespace."""
    data = {
        "doc_id": "  test_doc_123  ",
        "title": "  Test Topic  ",
        "description": "  A test topic description  ",
        "concepts": ["  concept1  ", "  concept2  "]
    }
    
    topic = TopicResponse(**data)
    assert topic.doc_id == "  test_doc_123  "  # Whitespace preserved
    assert topic.title == "  Test Topic  "
    assert topic.description == "  A test topic description  "
    assert topic.concepts == ["  concept1  ", "  concept2  "]

def test_models_with_none_values():
    """Test models with None values for optional fields."""
    data = {
        "doc_id": "test_doc_123",
        "title": None,
        "description": None,
        "concepts": []
    }
    
    topic = TopicResponse(**data)
    assert topic.title is None
    assert topic.description is None
    assert topic.concepts == []

def test_models_with_very_long_strings():
    """Test models with very long string values."""
    long_string = "x" * 10000
    
    data = {
        "learner_id": "learner_123",
        "question_id": "question_456",
        "doc_id": "doc_789",
        "question_text": long_string,
        "context_for_evaluation": long_string,
        "learner_answer": long_string
    }
    
    request = AnswerSubmissionRequest(**data)
    assert request.question_text == long_string
    assert request.context_for_evaluation == long_string
    assert request.learner_answer == long_string 