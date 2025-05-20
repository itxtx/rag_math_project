# src/evaluation/answer_evaluator.py
import json
import asyncio
import os
from typing import Dict, Any, Optional
import aiohttp

from src import config # For API keys or model names, if any

class AnswerEvaluator:
    """
    Evaluates a learner's answer against a given question and context using an LLM.
    """

    def __init__(self, llm_api_key: Optional[str] = None, llm_model_name: Optional[str] = None):
        """
        Initializes the AnswerEvaluator.

        Args:
            llm_api_key: API key for the LLM service.
            llm_model_name: The specific LLM model to use for evaluation.
        """
        self.api_key = llm_api_key
        
        if llm_model_name:
            self.model_name = llm_model_name
        elif hasattr(config, 'EVALUATION_LLM_MODEL'): # Specific config for evaluation model
            self.model_name = config.EVALUATION_LLM_MODEL
        elif hasattr(config, 'QUESTION_GENERATION_LLM_MODEL'): # Fallback to question gen model
            self.model_name = config.QUESTION_GENERATION_LLM_MODEL
        else:
            self.model_name = "gemini-2.0-flash" # Default model

        print(f"AnswerEvaluator initialized with model: {self.model_name}")

    def _build_evaluation_prompt(self, question_text: str, context: str, learner_answer: str) -> str:
        """
        Builds the prompt for the LLM to evaluate the learner's answer.
        Instructs the LLM to return a JSON object with 'accuracy_score' and 'feedback'.
        """
        prompt = f"""You are an expert evaluator. Your task is to assess the learner's answer based on the provided question and context.

        Please provide your evaluation in a JSON format with two keys:
        1. "accuracy_score": A float between 0.0 (completely incorrect) and 1.0 (perfectly correct and comprehensive based on the context).
        2. "feedback": A brief textual explanation for the score, highlighting strengths or areas for improvement. If the answer is incorrect, point out the mistake.
        3. "correct answer": Give a precise and short correct answer highlighting the correct areas of the answer, as well as patiently explaining the incorrect areas.
        
        Question:
        "{question_text}"

        Context provided to the learner (the answer should be based on this):
        \"\"\"
        {context}
        \"\"\"

        Learner's Answer:
        "{learner_answer}"

        Please return ONLY the JSON object containing your evaluation. For example:
        {{
          "accuracy_score": 0.8,
          "feedback": "The answer correctly identifies the main point but misses some nuances from the context."
          "correct_answer": "Your answer is mostly correct, but let's refine it a bit to be more precise based on the text.

                            A differential equation (DE) is an equation involving an unknown function and one or more of its derivatives.

                            An Ordinary Differential Equation (ODE) contains only ordinary derivatives of one or more unknown functions with respect to a single independent variable.
                            The order of an ODE is the order of the highest derivative of the unknown function appearing in the equation.
                            So, for example,  $x + 2y = \sin(x)$ is an ODE of order 0"
        }}
        
        If the learner's answer is completely off-topic or nonsensical, assign a low score and indicate this in the feedback.
        If the answer is perfect based on the context, assign a score of 1.0.
        """
        return prompt

    async def _call_llm_api_for_evaluation(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Makes an API call to the LLM with the evaluation prompt.
        Attempts to parse the response as JSON.
        """
        if not prompt:
            print("AnswerEvaluator Error: Prompt is empty. Cannot call LLM API.")
            return None
        
        print(f"\n--- Sending Evaluation Prompt to LLM ({self.model_name}) ---")
        # print(prompt) # Uncomment for debugging the exact prompt
        print("--- End of Evaluation Prompt ---")

        chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
        
        # For Gemini API expecting JSON output, we might need to configure generationConfig
        payload = {
            "contents": chat_history,
            "generationConfig": {
                "responseMimeType": "application/json", # Request JSON output
                # Optionally, define a schema if the model supports strict schema enforcement
                # For now, we'll rely on the prompt to guide JSON structure.
            }
        }
        
        api_key_to_use = self.api_key
        if api_key_to_use is None: 
            api_key_to_use = getattr(config, 'GEMINI_API_KEY', "") 
        
        if api_key_to_use:
            masked_key = api_key_to_use[:4] + "..." + api_key_to_use[-4:] if len(api_key_to_use) > 8 else "key_is_short"
            print(f"DEBUG (AnswerEvaluator): Using API Key (masked): {masked_key}")
        else:
            print("DEBUG (AnswerEvaluator): No API Key found/provided for evaluation LLM.")
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key_to_use}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(api_url, json=payload, headers={'Content-Type': 'application/json'}) as response:
                    response_text_for_error = await response.text()
                    response.raise_for_status() 
                    
                    # The response from Gemini when requesting JSON is expected to be in result.candidates[0].content.parts[0].text
                    # and this text itself should be the JSON string.
                    raw_json_response = json.loads(response_text_for_error) # Parse the outer response

                    if raw_json_response.get("candidates") and \
                       len(raw_json_response["candidates"]) > 0 and \
                       raw_json_response["candidates"][0].get("content") and \
                       raw_json_response["candidates"][0]["content"].get("parts") and \
                       len(raw_json_response["candidates"][0]["content"]["parts"]) > 0:
                        
                        json_string_from_llm = raw_json_response["candidates"][0]["content"]["parts"][0].get("text")
                        if json_string_from_llm:
                            try:
                                evaluation_data = json.loads(json_string_from_llm)
                                print("LLM Evaluation Response (JSON) Received and Parsed.")
                                return evaluation_data
                            except json.JSONDecodeError as je:
                                print(f"Error: LLM returned text that is not valid JSON: {je}")
                                print(f"LLM Raw Text Output: {json_string_from_llm}")
                                return {"accuracy_score": 0.0, "feedback": "Error: LLM response was not valid JSON."}
                        else:
                            print("Error: LLM response part contains no text.")
                            return None
                    else:
                        print("Error: Unexpected LLM API response structure (outer).")
                        print("Full LLM Response:", json.dumps(raw_json_response, indent=2)) 
                        return None

            except aiohttp.ClientResponseError as e:
                print(f"Error calling LLM API (HTTP Status {e.status}): {e.message}\nResponse Body: {response_text_for_error}")
                return None
            except Exception as e:
                print(f"Error calling LLM API (General Exception): {e}")
                import traceback
                traceback.print_exc()
                return None

    async def evaluate_answer(self, 
                              question_text: str, 
                              context: str, 
                              learner_answer: str
                              ) -> Dict[str, Any]:
        """
        Evaluates the learner's answer using the LLM.

        Args:
            question_text: The text of the question.
            context: The context based on which the question was asked.
            learner_answer: The learner's submitted answer.

        Returns:
            A dictionary containing 'accuracy_score' (float 0-1) and 'feedback' (str).
            Defaults to score 0.0 and error feedback if evaluation fails.
        """
        if not all([question_text, context, learner_answer]):
            print("AnswerEvaluator: Missing question, context, or answer for evaluation.")
            return {"accuracy_score": 0.0, "feedback": "Evaluation error: Missing input."}

        prompt = self._build_evaluation_prompt(question_text, context, learner_answer)
        
        evaluation_data = await self._call_llm_api_for_evaluation(prompt)

        if evaluation_data and isinstance(evaluation_data.get("accuracy_score"), (float, int)) \
           and isinstance(evaluation_data.get("feedback"), str):
            # Ensure score is within 0-1 range
            score = float(evaluation_data["accuracy_score"])
            evaluation_data["accuracy_score"] = max(0.0, min(1.0, score))
            return evaluation_data
        else:
            print("AnswerEvaluator: Failed to get a valid structured evaluation from LLM.")
            return {
                "accuracy_score": 0.0, 
                "feedback": "Could not evaluate the answer due to an LLM communication or parsing error."
            }

async def demo_evaluator():
    print("--- AnswerEvaluator Demo ---")
    # Ensure GEMINI_API_KEY is in config or .env for this demo to make a real call
    
    evaluator = AnswerEvaluator()

    question = "What is the primary function of mitochondria?"
    context = "Mitochondria are often called the powerhouses of the cell. They generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy. They are found in nearly all eukaryotic cells."
    
    good_answer = "Mitochondria generate ATP, which is the main energy currency of the cell. They are the cell's powerhouses."
    okay_answer = "They make energy."
    poor_answer = "They are green."

    print("\nEvaluating a good answer:")
    result_good = await evaluator.evaluate_answer(question, context, good_answer)
    print(f"Evaluation: {result_good}")

    print("\nEvaluating an okay answer:")
    result_okay = await evaluator.evaluate_answer(question, context, okay_answer)
    print(f"Evaluation: {result_okay}")

    print("\nEvaluating a poor answer:")
    result_poor = await evaluator.evaluate_answer(question, context, poor_answer)
    print(f"Evaluation: {result_poor}")
    
    print("\n--- AnswerEvaluator Demo Finished ---")

if __name__ == "__main__":
    # Load .env for direct script execution if needed
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') # Path to .env in project root
    if os.path.exists(dotenv_path):
        print(f"AnswerEvaluator Demo: Found .env file at {dotenv_path}, loading.")
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
        # This ensures config module can pick up GEMINI_API_KEY if set in .env
        # You might need to re-import or ensure config is loaded after dotenv if it's a module-level var.
        # For simplicity, we assume config.GEMINI_API_KEY will be available.

    asyncio.run(demo_evaluator())
