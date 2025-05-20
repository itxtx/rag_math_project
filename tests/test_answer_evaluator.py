# tests/evaluation/test_answer_evaluator.py
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import json
import asyncio
import aiohttp # For ClientResponseError

from src.evaluation.answer_evaluator import AnswerEvaluator
from src import config # For default model name, API key if used

# Helper to run async test methods
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

class TestAnswerEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = AnswerEvaluator(llm_api_key="test_dummy_key") 
        self.question = "What is the capital of France?"
        self.context = "France is a country in Western Europe. Its capital is Paris, known for the Eiffel Tower."
        self.learner_answer_correct = "Paris is the capital."
        self.learner_answer_partial = "I think it's Paris."
        self.learner_answer_incorrect = "Berlin."

    def test_build_evaluation_prompt(self):
        """Test the structure and content of the evaluation prompt."""
        prompt = self.evaluator._build_evaluation_prompt(
            self.question, self.context, self.learner_answer_correct
        )
        self.assertIn(self.question, prompt)
        self.assertIn(self.context, prompt)
        self.assertIn(self.learner_answer_correct, prompt)
        self.assertIn('"accuracy_score"', prompt)
        self.assertIn('"feedback"', prompt)
        self.assertIn("JSON format", prompt)

    @async_test 
    async def test_call_llm_api_for_evaluation_success(self):
        """Test successful LLM API call and JSON parsing."""
        with patch('src.evaluation.answer_evaluator.aiohttp.ClientSession') as mock_session_cls:
            mock_response_json_str = json.dumps({"accuracy_score": 0.9, "feedback": "Very good."})
            mock_llm_api_response = {
                "candidates": [{"content": {"parts": [{"text": mock_response_json_str}]}}]
            }

            mock_resp = AsyncMock(spec=aiohttp.ClientResponse)
            mock_resp.status = 200
            # .json() should return the parsed outer response
            # .text() should return the raw string of the outer response
            mock_resp.text = AsyncMock(return_value=json.dumps(mock_llm_api_response))
            
            mock_post_cm = AsyncMock() 
            mock_post_cm.__aenter__.return_value = mock_resp
            
            mock_session_instance = AsyncMock()
            mock_session_instance.post.return_value = mock_post_cm
            
            mock_session_cls.return_value = mock_session_instance 

            prompt = "Test prompt"
            result = await self.evaluator._call_llm_api_for_evaluation(prompt)

            self.assertIsNotNone(result)
            self.assertEqual(result.get("accuracy_score"), 0.9)
            self.assertEqual(result.get("feedback"), "Very good.")
            mock_session_instance.post.assert_called_once()

    @async_test 
    async def test_call_llm_api_for_evaluation_invalid_json_from_llm(self):
        """Test when LLM returns text that is not valid JSON."""
        with patch('src.evaluation.answer_evaluator.aiohttp.ClientSession') as mock_session_cls:
            invalid_json_str = "This is not JSON. Score: good. Feedback: ok."
            mock_llm_api_response = {
                "candidates": [{"content": {"parts": [{"text": invalid_json_str}]}}]
            }
            mock_resp = AsyncMock(spec=aiohttp.ClientResponse)
            mock_resp.status = 200
            mock_resp.text = AsyncMock(return_value=json.dumps(mock_llm_api_response))
            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_resp
            mock_session_instance = AsyncMock()
            mock_session_instance.post.return_value = mock_post_cm
            mock_session_cls.return_value = mock_session_instance

            result = await self.evaluator._call_llm_api_for_evaluation("Test prompt")
            self.assertIsNotNone(result)
            self.assertEqual(result.get("accuracy_score"), 0.0)
            self.assertIn("Error: LLM response was not valid JSON.", result.get("feedback", ""))

    @async_test 
    async def test_call_llm_api_http_error(self):
        """Test handling of HTTP errors from the LLM API."""
        with patch('src.evaluation.answer_evaluator.aiohttp.ClientSession') as mock_session_cls:
            # Mock request_info for ClientResponseError
            mock_request_info = MagicMock()
            mock_request_info.url = 'http://fakeurl.com'
            mock_request_info.method = 'POST'
            mock_request_info.headers = {}
            mock_request_info.real_url = 'http://fakeurl.com'


            mock_resp = AsyncMock(spec=aiohttp.ClientResponse)
            mock_resp.status = 403 
            mock_resp.text = AsyncMock(return_value='{"error": "API key invalid"}')
            
            # Simulate raise_for_status behavior correctly
            mock_resp.raise_for_status = MagicMock(
                side_effect=aiohttp.ClientResponseError(
                    request_info=mock_request_info, # Pass the mock request_info
                    history=(), 
                    status=403, 
                    message="Forbidden from mock" # Custom message for clarity
                    # headers=None # Optional
                )
            )
            
            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_resp
            mock_session_instance = AsyncMock()
            mock_session_instance.post.return_value = mock_post_cm
            mock_session_cls.return_value = mock_session_instance

            result = await self.evaluator._call_llm_api_for_evaluation("Test prompt")
            self.assertIsNone(result, "Should return None on HTTP error")

    @async_test 
    async def test_evaluate_answer_flow(self):
        """Test the main evaluate_answer method and score clamping."""
        with patch('src.evaluation.answer_evaluator.AnswerEvaluator._call_llm_api_for_evaluation', new_callable=AsyncMock) as mock_call_llm:
            mock_call_llm.return_value = {"accuracy_score": 0.85, "feedback": "Good job."}
            result = await self.evaluator.evaluate_answer(
                self.question, self.context, self.learner_answer_correct
            )
            self.assertEqual(result["accuracy_score"], 0.85)
            self.assertEqual(result["feedback"], "Good job.")
            mock_call_llm.assert_called_once()
            
            mock_call_llm.reset_mock()
            mock_call_llm.return_value = {"accuracy_score": 1.5, "feedback": "Too high."}
            result = await self.evaluator.evaluate_answer(
                self.question, self.context, self.learner_answer_correct
            )
            self.assertEqual(result["accuracy_score"], 1.0) 

            mock_call_llm.reset_mock()
            mock_call_llm.return_value = {"accuracy_score": -0.5, "feedback": "Too low."}
            result = await self.evaluator.evaluate_answer(
                self.question, self.context, self.learner_answer_correct
            )
            self.assertEqual(result["accuracy_score"], 0.0)

            mock_call_llm.reset_mock()
            mock_call_llm.return_value = None
            result = await self.evaluator.evaluate_answer(
                self.question, self.context, self.learner_answer_correct
            )
            self.assertEqual(result["accuracy_score"], 0.0)
            self.assertIn("Could not evaluate", result["feedback"])

            result_missing_input = await self.evaluator.evaluate_answer("", "", "")
            self.assertEqual(result_missing_input["accuracy_score"], 0.0)
            self.assertIn("Missing input", result_missing_input["feedback"])

if __name__ == '__main__':
    print("Please run these tests using `python -m unittest discover tests` or `python -m unittest tests.evaluation.test_answer_evaluator`")
