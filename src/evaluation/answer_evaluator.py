# src/evaluation/answer_evaluator.py
import json
import asyncio
import os
import re # For handling escapes
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
        elif hasattr(config, 'EVALUATION_LLM_MODEL'): 
            self.model_name = config.EVALUATION_LLM_MODEL
        elif hasattr(config, 'QUESTION_GENERATION_LLM_MODEL'): 
            self.model_name = config.QUESTION_GENERATION_LLM_MODEL
        else:
            self.model_name = "gemini-2.0-flash" 

        print(f"AnswerEvaluator initialized with model: {self.model_name}")

    def _build_evaluation_prompt(self, question_text: str, context: str, learner_answer: str) -> str:
        """
        Builds the prompt for the LLM to evaluate the learner's answer.
        Instructs the LLM to return a JSON object with 'accuracy_score', 'feedback', and optionally 'correct_answer'.
        """
        prompt = f"""You are an expert evaluator. Your task is to assess the learner's answer based on the provided question and context.

        Please provide your evaluation in a JSON format with the following keys:
        1. "accuracy_score": A float between 0.0 (completely incorrect) and 1.0 (perfectly correct and comprehensive based on the context).
        2. "feedback": A brief textual explanation for the score, highlighting strengths or areas for improvement. If the answer is incorrect, point out the mistake.
        3. "correct_answer": (Optional) If the learner's answer is significantly incorrect or incomplete, provide a concise model correct answer based *only* on the provided context. If the learner's answer is mostly correct, this field can be null or omitted.

        Question:
        "{question_text}"

        Context provided to the learner (the answer should be based on this):
        \"\"\"
        {context}
        \"\"\"

        Learner's Answer:
        "{learner_answer}"

        Please return ONLY the JSON object containing your evaluation.
        Example of a valid JSON response:
        {{
          "accuracy_score": 0.8,
          "feedback": "The answer correctly identifies the main point but misses some nuances from the context. For instance, the context also mentions X, which was omitted.",
          "correct_answer": "The main point is Y, and it's also important to consider X from the context."
        }}
        
        Ensure all string values within the JSON are properly escaped. For example, any literal backslash characters (e.g., in LaTeX like \\mu or \\int) must be represented as \\\\ (a double backslash) within the JSON string value. Quotes within strings must be escaped as \\\". Newlines within strings must be escaped as \\n.
        """
        return prompt

    async def _call_llm_api_for_evaluation(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Makes an API call to the LLM with the evaluation prompt.
        Attempts to parse the response as JSON, handling potential escape issues.
        """
        if not prompt:
            print("AnswerEvaluator Error: Prompt is empty. Cannot call LLM API.")
            return None
        
        print(f"\n--- Sending Evaluation Prompt to LLM ({self.model_name}) ---")
        # print(prompt) 
        print("--- End of Evaluation Prompt ---")

        chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
        
        payload = {
            "contents": chat_history,
            "generationConfig": {
                "responseMimeType": "application/json", 
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
                    response_text_content = await response.text() 
                    response.raise_for_status() 
                    
                    raw_outer_json_response = json.loads(response_text_content)

                    if raw_outer_json_response.get("candidates") and \
                       len(raw_outer_json_response["candidates"]) > 0 and \
                       raw_outer_json_response["candidates"][0].get("content") and \
                       raw_outer_json_response["candidates"][0]["content"].get("parts") and \
                       len(raw_outer_json_response["candidates"][0]["content"]["parts"]) > 0:
                        
                        json_string_from_llm = raw_outer_json_response["candidates"][0]["content"]["parts"][0].get("text")
                        if json_string_from_llm:
                            print(f"DEBUG: Raw JSON string from LLM part: {repr(json_string_from_llm)}")
                            try:
                                evaluation_data = json.loads(json_string_from_llm)
                                print("LLM Evaluation Response (JSON) Parsed successfully (1st attempt).")
                                return evaluation_data
                            except json.JSONDecodeError as je:
                                print(f"Warning: Initial JSON parsing failed: {je}. Attempting escape fixes.")
                                
                                # Attempt 1: Fix common unescaped backslashes (not part of valid JSON escapes)
                                fixed_json_string_attempt1 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_string_from_llm)
                                # Attempt 2: Escape literal newlines and carriage returns
                                fixed_json_string_attempt2 = fixed_json_string_attempt1.replace("\n", "\\n").replace("\r", "\\r")
                                
                                print(f"DEBUG: JSON string after escape fixes: {repr(fixed_json_string_attempt2)}")
                                try:
                                    evaluation_data = json.loads(fixed_json_string_attempt2)
                                    print("LLM Evaluation Response (JSON) Parsed successfully (2nd attempt after fixes).")
                                    return evaluation_data
                                except json.JSONDecodeError as je2:
                                    print(f"Error: LLM returned text that is not valid JSON even after escape fixes: {je2}")
                                    print(f"  Error details: msg='{je2.msg}', doc (first 100)='{je2.doc[:100]}...', lineno={je2.lineno}, colno={je2.colno}, pos={je2.pos}")
                                    if je2.doc and isinstance(je2.doc, str): # Check if je2.doc is a string
                                        print(f"  Problematic char context: '{je2.doc[max(0,je2.pos-10):je2.pos+10]}'")
                                    print(f"LLM Original Text Output (from part): {repr(json_string_from_llm)}")
                                    print(f"Attempted Fixed JSON string (final): {repr(fixed_json_string_attempt2)}")
                                    return {"accuracy_score": 0.0, "feedback": "Error: LLM response was not valid JSON.", "correct_answer": None}
                        else:
                            print("Error: LLM response part contains no text.")
                            return None 
                    else:
                        print("Error: Unexpected LLM API response structure (outer candidates/parts).")
                        print("Full Outer LLM Response:", json.dumps(raw_outer_json_response, indent=2)) 
                        return None

            except aiohttp.ClientResponseError as e:
                print(f"Error calling LLM API (HTTP Status {e.status}): {e.message}\nResponse Body: {response_text_content}")
                return None
            except json.JSONDecodeError as je_outer: 
                print(f"Error: Could not parse the main Gemini API response as JSON: {je_outer}")
                print(f"Main Gemini API Raw Response Text: {response_text_content if 'response_text_content' in locals() else 'Response text not available'}")
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
        if not all([question_text, context, learner_answer]):
            print("AnswerEvaluator: Missing question, context, or answer for evaluation.")
            return {"accuracy_score": 0.0, "feedback": "Evaluation error: Missing input.", "correct_answer": None}

        prompt = self._build_evaluation_prompt(question_text, context, learner_answer)
        evaluation_data = await self._call_llm_api_for_evaluation(prompt)

        if evaluation_data and isinstance(evaluation_data.get("accuracy_score"), (float, int)) \
           and isinstance(evaluation_data.get("feedback"), str):
            score = float(evaluation_data["accuracy_score"])
            evaluation_data["accuracy_score"] = max(0.0, min(1.0, score))
            evaluation_data.setdefault("correct_answer", None) 
            return evaluation_data
        else:
            print("AnswerEvaluator: Failed to get a valid structured evaluation from LLM.")
            return {
                "accuracy_score": 0.0, 
                "feedback": evaluation_data.get("feedback") if isinstance(evaluation_data, dict) else "Could not evaluate the answer due to an LLM communication or parsing error.",
                "correct_answer": evaluation_data.get("correct_answer") if isinstance(evaluation_data, dict) else None
            }

async def demo_evaluator():
    print("--- AnswerEvaluator Demo ---")
    evaluator = AnswerEvaluator()
    question = "What is the primary function of mitochondria?"
    context = "Mitochondria are often called the powerhouses of the cell. They generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy. They are found in nearly all eukaryotic cells."
    good_answer = "Mitochondria generate ATP, which is the main energy currency of the cell."
    
    print("\nEvaluating a good answer:")
    result_good = await evaluator.evaluate_answer(question, context, good_answer)
    print(f"Evaluation: {result_good}")
    
    print("\n--- AnswerEvaluator Demo Finished ---")

if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') 
    if os.path.exists(dotenv_path):
        print(f"AnswerEvaluator Demo: Found .env file at {dotenv_path}, loading.")
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
    asyncio.run(demo_evaluator())
