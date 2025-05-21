# src/generation/question_generator_rag.py

import json
from typing import List, Dict, Optional, Any
import aiohttp 
import os

from src import config 

class RAGQuestionGenerator:
    """
    Generates questions based on provided text chunks using a RAG approach with an LLM.
    """

    def __init__(self, llm_api_key: Optional[str] = None, llm_model_name: Optional[str] = None):
        self.api_key = llm_api_key 
        
        if llm_model_name:
            self.model_name = llm_model_name
        elif hasattr(config, 'QUESTION_GENERATION_LLM_MODEL'):
            self.model_name = config.QUESTION_GENERATION_LLM_MODEL
        else:
            self.model_name = "gemini-2.0-flash" 

        print(f"RAGQuestionGenerator initialized with model: {self.model_name}")


    def _build_prompt(self, 
                      context_chunks: List[Dict], 
                      num_questions: int = 1, 
                      question_type: str = "conceptual",
                      difficulty_level: str = "intermediate"
                      ) -> str:
        """
        Builds the prompt for the LLM based on the provided context chunks.
        """
        if not context_chunks:
            return ""

        context_str = "\n\n---\n\n".join([chunk.get("chunk_text", "") for chunk in context_chunks if chunk.get("chunk_text","").strip()])
        
        if not context_str.strip():
            print("Warning: No valid text content found in context_chunks to build prompt.")
            return ""

        # --- Refined Prompt Engineering ---
        difficulty_instruction = ""
        if difficulty_level == "beginner":
            difficulty_instruction = "The question should be straightforward, focusing on direct recall of key facts or definitions explicitly stated in the context. Aim for clarity and simplicity."
        elif difficulty_level == "advanced":
            difficulty_instruction = "The question should be challenging and potentially longer, requiring synthesis of information, critical thinking, or application of concepts from the context to implied or new scenarios. It may involve multiple steps or deeper analysis. Avoid overly simple or direct recall questions."
        else: # intermediate (default)
            difficulty_instruction = "The question should test comprehension and the ability to connect ideas within the context, going beyond simple recall but not overly complex. It can be moderately long."

        phrasing_constraint = ""
        if question_type.lower() not in ["conceptual", "definition"]: # For most question types
            phrasing_constraint = "Avoid starting the question with phrases like 'According to the text...' or 'Based on the provided context...'. Instead, formulate the question directly."
        else: # For conceptual/definitional, such phrasing might be acceptable if it helps clarity
            phrasing_constraint = "If asking for a definition or a core concept explanation, you may use phrases like 'According to the definition provided...' if it enhances clarity, but prefer direct questions where possible."

        prompt = f"""You are an expert at creating educational questions.
        Based on the following context, please generate {num_questions} distinct and insightful {question_type} question(s).
        The question(s) should be at a {difficulty_level} difficulty level.
        {difficulty_instruction}
        {phrasing_constraint}
        Ensure each question is specific, clear, and answerable primarily from the provided text. Aim for questions that encourage thoughtful engagement with the material.
        
        Context:
        \"\"\"
        {context_str}
        \"\"\"

        Generate exactly {num_questions} question(s), each on a new line, starting with a number and a period (e.g., "1. Question text"):
        """
        for i in range(1, num_questions + 1):
            prompt += f"{i}. \n" 
        # --- End of Refined Prompt Engineering ---
        
        return prompt

    async def _call_llm_api(self, prompt: str) -> Optional[str]:
        if not prompt:
            print("Error: Prompt is empty. Cannot call LLM API.")
            return None
        
        print(f"\n--- Sending Prompt to LLM ({self.model_name}) ---")
        # print(prompt) 
        print("--- End of Prompt ---")

        chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
        payload = {"contents": chat_history}
        
        api_key_to_use = self.api_key
        if api_key_to_use is None: 
            api_key_to_use = getattr(config, 'GEMINI_API_KEY', "") 
        
        if api_key_to_use:
            masked_key = api_key_to_use[:4] + "..." + api_key_to_use[-4:] if len(api_key_to_use) > 8 else "key_is_short"
            print(f"DEBUG (RAGQuestionGenerator): Using API Key (masked): {masked_key}")
        else:
            print("DEBUG (RAGQuestionGenerator): No API Key found/provided. Relying on environment injection for gemini-2.0-flash or public access.")
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key_to_use}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(api_url, json=payload, headers={'Content-Type': 'application/json'}) as response:
                    response_text_for_error = await response.text()
                    response.raise_for_status() 
                    result = json.loads(response_text_for_error)
                    
                    if result.get("candidates") and \
                       len(result["candidates"]) > 0 and \
                       result["candidates"][0].get("content") and \
                       result["candidates"][0]["content"].get("parts") and \
                       len(result["candidates"][0]["content"]["parts"]) > 0:
                        generated_text = result["candidates"][0]["content"]["parts"][0].get("text")
                        print("LLM Response Received for question generation.")
                        return generated_text
                    else:
                        print("Error: Unexpected LLM API response structure for question generation.")
                        print("Full LLM Response:", json.dumps(result, indent=2)) 
                        return None
            except aiohttp.ClientResponseError as e:
                error_body_text = "Could not read error body (already attempted)."
                print(f"Error calling LLM API (HTTP Status {e.status}): {e.message}\nResponse Body: {response_text_for_error}")
                return None
            except Exception as e:
                print(f"Error calling LLM API (General Exception): {e}")
                import traceback
                traceback.print_exc()
                return None

    def _parse_llm_response(self, llm_response_text: str, num_questions: int) -> List[str]:
        if not llm_response_text:
            return []

        questions = []
        lines = llm_response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit(): 
                try:
                    dot_index = line.index('.')
                    question_text = line[dot_index+1:].strip()
                    if question_text: 
                        questions.append(question_text)
                except ValueError:
                    pass 
        
        if len(questions) < num_questions and len(questions) < len(lines):
            print(f"Warning: Parsed {len(questions)} questions, but expected {num_questions} and had {len(lines)} lines. LLM output format might differ from expected.")

        return questions[:num_questions]


    async def generate_questions(self,
                                 context_chunks: List[Dict],
                                 num_questions: int = 1, 
                                 question_type: str = "conceptual",
                                 difficulty_level: str = "intermediate" 
                                 ) -> List[str]:
        if not context_chunks:
            print("No context chunks provided, cannot generate questions.")
            return []
        if num_questions <= 0:
            print("Number of questions must be positive.")
            return []

        prompt = self._build_prompt(context_chunks, num_questions, question_type, difficulty_level)
        if not prompt:
            print("Failed to build prompt from context.")
            return []

        llm_response = await self._call_llm_api(prompt)

        if llm_response:
            return self._parse_llm_response(llm_response, num_questions)
        else:
            print("Failed to get a valid response from LLM for question generation.")
            return []

async def demo():
    print("--- RAG Question Generator Demo (Refined Prompts) ---")
    
    generator = RAGQuestionGenerator() 

    sample_chunks = [
        {"chunk_id": "c1", "chunk_text": "The theory of relativity was proposed by Albert Einstein. It has two main parts: special relativity and general relativity. Special relativity deals with the relationship between space and time for objects moving at constant speeds."},
        {"chunk_id": "c2", "chunk_text": "General relativity, published in 1915, is a theory of gravitation. It describes gravity not as a force, but as a curvature of spacetime caused by mass and energy."}
    ]
    
    print("\nGenerating 1 'beginner' factual question:")
    questions_beginner = await generator.generate_questions(sample_chunks, num_questions=1, question_type="factual", difficulty_level="beginner")
    if questions_beginner: print(f"  Q_Beginner_Factual: {questions_beginner[0]}")

    print("\nGenerating 1 'intermediate' conceptual question:")
    questions_intermediate = await generator.generate_questions(sample_chunks, num_questions=1, question_type="conceptual", difficulty_level="intermediate")
    if questions_intermediate: print(f"  Q_Intermediate_Conceptual: {questions_intermediate[0]}")

    print("\nGenerating 1 'advanced' analytical question:")
    questions_advanced = await generator.generate_questions(sample_chunks, num_questions=1, question_type="analytical", difficulty_level="advanced")
    if questions_advanced: print(f"  Q_Advanced_Analytical: {questions_advanced[0]}")


    print("\n--- RAG Question Generator Demo Finished ---")

if __name__ == '__main__':
    import asyncio
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') 
    if os.path.exists(dotenv_path):
        print(f"RAGQuestionGenerator Demo: Found .env file at {dotenv_path}, loading.")
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)

    try:
        asyncio.run(demo())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("Demo cannot be run directly from a running event loop (e.g. Jupyter).")
        else:
            raise
    except Exception as e:
        print(f"An error occurred in the demo: {e}")
        import traceback
        traceback.print_exc()
