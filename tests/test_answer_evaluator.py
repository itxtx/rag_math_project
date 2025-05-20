# tests/evaluation/test_answer_evaluator.py
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import json
import asyncio
import aiohttp # For ClientResponseError

from src.evaluation.answer_evaluator import AnswerEvaluator
from src import config # For default model name, API key if used

class TestAnswerEvaluator(unittest.TestCase):

    def setUp(self):
        # Initialize with no API key for testing, relying on mocks or default model behavior
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

    @patch('src.evaluation.answer_evaluator.aiohttp.ClientSession')
    async def test_call_llm_api_for_evaluation_success(self, mock_session_cls):
        """Test successful LLM API call and JSON parsing."""
        mock_response_json_str = json.dumps({"accuracy_score": 0.9, "feedback": "Very good."})
        mock_llm_api_response = {
            "candidates": [{"content": {"parts": [{"text": mock_response_json_str}]}}]
        }

        # Configure the mock session and response
        mock_resp = AsyncMock(spec=aiohttp.ClientResponse)
        mock_resp.status = 200
        mock_resp.json.return_value = mock_llm_api_response # Outer JSON
        mock_resp.text.return_value = json.dumps(mock_llm_api_response) # For raise_for_status and error logging
        
        mock_post_cm = AsyncMock() # Context manager for session.post
        mock_post_cm.__aenter__.return_value = mock_resp
        
        mock_session_instance = AsyncMock()
        mock_session_instance.post.return_value = mock_post_cm
        
        mock_session_cls.return_value = mock_session_instance # session() returns our mock_session_instance

        prompt = "Test prompt"
        result = await self.evaluator._call_llm_api_for_evaluation(prompt)

        self.assertIsNotNone(result)
        self.assertEqual(result.get("accuracy_score"), 0.9)
        self.assertEqual(result.get("feedback"), "Very good.")
        mock_session_instance.post.assert_called_once()

    @patch('src.evaluation.answer_evaluator.aiohttp.ClientSession')
    async def test_call_llm_api_for_evaluation_invalid_json_from_llm(self, mock_session_cls):
        """Test when LLM returns text that is not valid JSON."""
        invalid_json_str = "This is not JSON. Score: good. Feedback: ok."
        mock_llm_api_response = {
            "candidates": [{"content": {"parts": [{"text": invalid_json_str}]}}]
        }
        mock_resp = AsyncMock(spec=aiohttp.ClientResponse)
        mock_resp.status = 200
        mock_resp.json.return_value = mock_llm_api_response
        mock_resp.text.return_value = json.dumps(mock_llm_api_response)
        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__.return_value = mock_resp
        mock_session_instance = AsyncMock()
        mock_session_instance.post.return_value = mock_post_cm
        mock_session_cls.return_value = mock_session_instance

        result = await self.evaluator._call_llm_api_for_evaluation("Test prompt")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("accuracy_score"), 0.0)
        self.assertIn("Error: LLM response was not valid JSON.", result.get("feedback", ""))

    @patch('src.evaluation.answer_evaluator.aiohttp.ClientSession')
    async def test_call_llm_api_http_error(self, mock_session_cls):
        """Test handling of HTTP errors from the LLM API."""
        mock_resp = AsyncMock(spec=aiohttp.ClientResponse)
        mock_resp.status = 403 # Simulate Forbidden
        mock_resp.text.return_value = '{"error": "API key invalid"}'
        # Simulate raise_for_status behavior
        mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(), 
            history=(), 
            status=403, 
            message="Forbidden"
        ))
        
        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__.return_value = mock_resp
        mock_session_instance = AsyncMock()
        mock_session_instance.post.return_value = mock_post_cm
        mock_session_cls.return_value = mock_session_instance

        result = await self.evaluator._call_llm_api_for_evaluation("Test prompt")
        self.assertIsNone(result, "Should return None on HTTP error")

    @patch('src.evaluation.answer_evaluator.AnswerEvaluator._call_llm_api_for_evaluation', new_callable=AsyncMock)
    async def test_evaluate_answer_flow(self, mock_call_llm):
        """Test the main evaluate_answer method and score clamping."""
        # Case 1: LLM returns valid score and feedback
        mock_call_llm.return_value = {"accuracy_score": 0.85, "feedback": "Good job."}
        result = await self.evaluator.evaluate_answer(
            self.question, self.context, self.learner_answer_correct
        )
        self.assertEqual(result["accuracy_score"], 0.85)
        self.assertEqual(result["feedback"], "Good job.")
        mock_call_llm.assert_called_once()
        
        # Case 2: LLM returns score out of bounds (too high)
        mock_call_llm.reset_mock()
        mock_call_llm.return_value = {"accuracy_score": 1.5, "feedback": "Too high."}
        result = await self.evaluator.evaluate_answer(
            self.question, self.context, self.learner_answer_correct
        )
        self.assertEqual(result["accuracy_score"], 1.0) # Should be clamped
        self.assertEqual(result["feedback"], "Too high.")

        # Case 3: LLM returns score out of bounds (too low)
        mock_call_llm.reset_mock()
        mock_call_llm.return_value = {"accuracy_score": -0.5, "feedback": "Too low."}
        result = await self.evaluator.evaluate_answer(
            self.question, self.context, self.learner_answer_correct
        )
        self.assertEqual(result["accuracy_score"], 0.0) # Should be clamped

        # Case 4: LLM call fails or returns malformed data
        mock_call_llm.reset_mock()
        mock_call_llm.return_value = None
        result = await self.evaluator.evaluate_answer(
            self.question, self.context, self.learner_answer_correct
        )
        self.assertEqual(result["accuracy_score"], 0.0)
        self.assertIn("Could not evaluate", result["feedback"])

        # Case 5: Missing input
        result_missing_input = await self.evaluator.evaluate_answer("", "", "")
        self.assertEqual(result_missing_input["accuracy_score"], 0.0)
        self.assertIn("Missing input", result_missing_input["feedback"])


# To run async tests directly:
if __name__ == '__main__':
    # Need to wrap in an event loop runner if using unittest.main() directly for async tests
    # or run with `python -m unittest tests.evaluation.test_answer_evaluator`
    
    # This is a simple way to run async tests if the file is executed directly.
    # For a full test suite, `python -m unittest discover` is preferred.
    async def main_async_tests():
        suite = unittest.TestSuite()
        # Need to use TestLoader for async tests if not using a specialized runner
        # For simplicity, we'll assume `python -m unittest ...` handles this.
        # If running this file directly, you might need a custom runner for async tests.
        # This example focuses on the test definitions.
        loader = unittest.TestLoader()
        suite.addTest(loader.loadTestsFromTestCase(TestAnswerEvaluator))
        # runner = unittest.TextTestRunner()
        # runner.run(suite) # This won't work directly for async methods without an async test runner
        print("Async tests in TestAnswerEvaluator are designed to be run with `python -m unittest discover` or similar.")
        print("Running a simple direct call for demonstration of one test:")
        
        # Example of running one async test directly (less ideal than unittest framework)
        test_instance = TestAnswerEvaluator()
        test_instance.setUp()
        await test_instance.test_call_llm_api_for_evaluation_success()
        print("One async test executed directly.")


    if __name__ == '__main__': # Double check for direct execution
        # asyncio.run(main_async_tests()) # This is complex to set up correctly with unittest
        # It's better to rely on `python -m unittest discover tests`
        print("Please run these tests using `python -m unittest discover tests` or `python -m unittest tests.evaluation.test_answer_evaluator`")
        # Fallback for simple direct run of one test (for quick check)
        async def run_one():
            evaluator_test = TestAnswerEvaluator()
            evaluator_test.setUp()
            await evaluator_test.test_evaluate_answer_flow() # Pick one async test
            print("Direct execution of one async test completed.")
        # asyncio.run(run_one()) # Uncomment to try direct run of one test
