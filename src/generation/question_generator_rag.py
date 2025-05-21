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
                      question_type: str = "conceptual", # e.g., conceptual, factual, application
                      difficulty_level: str = "intermediate", 
                      question_style: str = "standard" # New: standard, fill_in_blank, complete_proof_step
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
            difficulty_instruction = "The question should be straightforward, focusing on direct recall of key facts or definitions explicitly stated in the context. Aim for clarity and simplicity. Questions can be shorter."
        elif difficulty_level == "advanced":
            difficulty_instruction = "The question should be challenging and potentially longer, requiring synthesis of information, critical thinking, or application of concepts from the context to implied or new scenarios. It may involve multiple steps or deeper analysis. Avoid overly simple or direct recall questions."
        else: # intermediate (default)
            difficulty_instruction = "The question should test comprehension and the ability to connect ideas within the context, going beyond simple recall but not overly complex. It can be moderately long."

        phrasing_constraint = ""
        # Allow "According to the text" for conceptual/definitional questions if it helps clarity,
        # but generally encourage direct questions.
        if question_type.lower() not in ["conceptual", "definition_recall"]: 
            phrasing_constraint = "Formulate the question directly, avoiding phrases like 'According to the text...' or 'Based on the provided context...' unless absolutely necessary for clarity."
        
        style_instruction = ""
        if question_style == "fill_in_blank":
            style_instruction = "The question should be phrased as a statement with a key term or phrase missing, indicated by '[BLANK]'. The learner needs to fill in the blank."
            difficulty_instruction += " The blank should target a specific, important piece of information."
        elif question_style == "complete_proof_step" or question_style == "complete_definition_step":
            # This is more complex and might require the LLM to identify a suitable step.
            # For now, a general instruction.
            style_instruction = f"The question should present a part of a {question_style.split('_')[1]} or explanation from the context and ask the learner to provide the next logical step, a missing justification, or a concluding part. Indicate the missing part clearly, perhaps with '[YOUR_TASK_HERE]'."
            difficulty_instruction += " This requires the learner to understand the flow of the argument or definition."
        else: # standard
            style_instruction = "The question should be a standard interrogative question."


        prompt = f"""You are an expert at creating high-quality educational questions based on provided text.
        Your goal is to generate {num_questions} distinct and insightful {question_type} question(s).
        
        Question Characteristics:
        - Difficulty Level: {difficulty_level}. {difficulty_instruction}
        - Style: {style_instruction}
        - Phrasing: {phrasing_constraint}
        - Specificity: Ensure each question is specific, clear, and unambiguous. Avoid overly vague questions.
        - Context-Bound: Each question must be answerable primarily from the provided text context.
        - Engagement: Aim for questions that encourage thoughtful engagement with the material, not just trivial recall unless 'beginner' difficulty is specified for factual recall.

        Context:
        \"\"\"
        {context_str}
        \"\"\"

        Output Format:
        Generate exactly {num_questions} question(s). Each question must be on a new line and start with a number followed by a period (e.g., "1. What is...?"). Mathematical expressions should be in valid latex.
        """
        # Add numbered placeholders to guide the LLM for the exact number of questions
        for i in range(1, num_questions + 1):
            prompt += f"{i}. \n" 
        
        return prompt

    async def _call_llm_api(self, prompt: str) -> Optional[str]:
        # ... (LLM API call logic remains the same as question_generator_rag_v3_difficulty) ...
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
        # ... (Parsing logic remains the same as question_generator_rag_v3_difficulty) ...
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
                                 difficulty_level: str = "intermediate",
                                 question_style: str = "standard" # New parameter
                                 ) -> List[str]:
        """
        Generates questions based on the provided context chunks, difficulty, and style.
        """
        if not context_chunks:
            print("No context chunks provided, cannot generate questions.")
            return []
        if num_questions <= 0:
            print("Number of questions must be positive.")
            return []

        prompt = self._build_prompt(
            context_chunks, num_questions, question_type, difficulty_level, question_style
        )
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
    print("--- RAG Question Generator Demo (Refined Prompts & Styles) ---")
    generator = RAGQuestionGenerator() 
    sample_chunks = [{"chunk_text": "A vector space is a set V of objects, called vectors, on which two operations called vector addition and scalar multiplication are defined. Scalars are often real numbers, but can also be complex numbers or, more generally, elements of any field."}]

    print("\nGenerating 1 'beginner' 'fill_in_blank' question:")
    q1 = await generator.generate_questions(sample_chunks, num_questions=1, question_type="definition_recall", difficulty_level="beginner", question_style="fill_in_blank")
    if q1: print(f"  Q: {q1[0]}")

    print("\nGenerating 1 'intermediate' 'standard' conceptual question:")
    q2 = await generator.generate_questions(sample_chunks, num_questions=1, question_type="conceptual", difficulty_level="intermediate", question_style="standard")
    if q2: print(f"  Q: {q2[0]}")
    
    proof_context = [{"chunk_text": "Theorem: The sum of angles in a triangle is 180 degrees. Proof: Step 1: Draw a line parallel to one side of the triangle through the opposite vertex. Step 2: Identify alternate interior angles formed by transversals. Step 3: Sum these angles, which form a straight line, to show they equal 180 degrees."}]
    print("\nGenerating 1 'intermediate' 'complete_proof_step' question:")
    q3 = await generator.generate_questions(proof_context, num_questions=1, question_type="reasoning", difficulty_level="intermediate", question_style="complete_proof_step")
    if q3: print(f"  Q: {q3[0]}")

    print("\n--- RAG Question Generator Demo Finished ---")

if __name__ == '__main__':
    # ... (main block remains the same) ...
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
        else: raise
    except Exception as e:
        print(f"An error occurred in the demo: {e}")
        import traceback
        traceback.print_exc()
