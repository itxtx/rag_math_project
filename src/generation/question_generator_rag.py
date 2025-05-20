
import json
from typing import List, Dict, Optional, Any
import aiohttp # For making asynchronous HTTP requests
import os
from src import config # For API keys or model names, if any

class RAGQuestionGenerator:
    """
    Generates questions based on provided text chunks using a RAG approach with an LLM.
    """

    def __init__(self, llm_api_key: Optional[str] = None, llm_model_name: Optional[str] = None):
        """
        Initializes the RAGQuestionGenerator.

        Args:
            llm_api_key: API key for the LLM service. If not provided, it might be
                         sourced from config or environment.
            llm_model_name: The specific LLM model to use for question generation.
        """
        self.api_key = llm_api_key # User-provided key takes precedence
        
        if llm_model_name:
            self.model_name = llm_model_name
        elif hasattr(config, 'QUESTION_GENERATION_LLM_MODEL'):
            self.model_name = config.QUESTION_GENERATION_LLM_MODEL
        else:
            self.model_name = "gemini-2.0-flash" # Default model

        print(f"RAGQuestionGenerator initialized with model: {self.model_name}")


    def _build_prompt(self, context_chunks: List[Dict], num_questions: int = 3, question_type: str = "conceptual") -> str:
        """
        Builds the prompt for the LLM based on the provided context chunks.

        Args:
            context_chunks: A list of dictionaries, where each dictionary is a retrieved text chunk
                            (e.g., from the Retriever). Expected to have at least a "chunk_text" key.
            num_questions: The desired number of questions to generate.
            question_type: The type of questions to generate (e.g., "conceptual", "factual", "problem-solving").

        Returns:
            A string representing the prompt to be sent to the LLM.
        """
        if not context_chunks:
            return ""

        context_str = "\n\n---\n\n".join([chunk.get("chunk_text", "") for chunk in context_chunks if chunk.get("chunk_text","").strip()])
        
        if not context_str.strip():
            print("Warning: No valid text content found in context_chunks to build prompt.")
            return ""

        prompt = f"""Based on the following context, please generate {num_questions} distinct and insightful {question_type} questions.
Each question should be answerable primarily from the provided text.
Avoid questions that are too simple (yes/no) unless specifically asking for a definition or confirmation of a fact stated.
Focus on understanding, application, or elaboration of the concepts presented.

Context:
\"\"\"
{context_str}
\"\"\"

Generate exactly {num_questions} questions, each on a new line, starting with a number and a period (e.g., "1. Question text"):
"""
        # Add numbered placeholders to guide the LLM for the exact number of questions
        for i in range(1, num_questions + 1):
            prompt += f"{i}. \n" # LLM should fill in after the period and space
        
        return prompt

    async def _call_llm_api(self, prompt: str) -> Optional[str]:
        """
        Makes an API call to the LLM with the given prompt using aiohttp.

        Args:
            prompt: The prompt string to send to the LLM.

        Returns:
            The raw text response from the LLM, or None if an error occurs.
        """
        if not prompt:
            print("Error: Prompt is empty. Cannot call LLM API.")
            return None
        
        print(f"\n--- Sending Prompt to LLM ({self.model_name}) ---")
        # print(prompt) # Uncomment for debugging the exact prompt
        print("--- End of Prompt ---")

        chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
        payload = {"contents": chat_history}
        
        # Determine API key: User-provided > config.GEMINI_API_KEY > "" (for Canvas injection)
        api_key_to_use = self.api_key
        if api_key_to_use is None: # If not provided during init
            api_key_to_use = getattr(config, 'GEMINI_API_KEY', "") # Check config
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key_to_use}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(api_url, json=payload, headers={'Content-Type': 'application/json'}) as response:
                    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                    result = await response.json()
                    
                    if result.get("candidates") and \
                       len(result["candidates"]) > 0 and \
                       result["candidates"][0].get("content") and \
                       result["candidates"][0]["content"].get("parts") and \
                       len(result["candidates"][0]["content"]["parts"]) > 0:
                        generated_text = result["candidates"][0]["content"]["parts"][0].get("text")
                        print("LLM Response Received.")
                        return generated_text
                    else:
                        print("Error: Unexpected LLM API response structure.")
                        print("Full LLM Response:", json.dumps(result, indent=2)) # For debugging
                        return None
            except aiohttp.ClientResponseError as e:
                error_body_text = "Could not read error body."
                try:
                    error_body_text = await e.response.text()
                except Exception as read_err:
                    print(f"Additionally, failed to read error response body: {read_err}")
                print(f"Error calling LLM API (HTTP Status {e.status}): {e.message}\nResponse Body: {error_body_text}")
                return None
            except Exception as e:
                print(f"Error calling LLM API (General Exception): {e}")
                import traceback
                traceback.print_exc()
                return None

    def _parse_llm_response(self, llm_response_text: str, num_questions: int) -> List[str]:
        """
        Parses the raw text response from the LLM into a list of questions.
        Assumes questions are numbered (e.g., "1. Question one?", "2. Question two?").

        Args:
            llm_response_text: The raw text output from the LLM.
            num_questions: The expected number of questions.

        Returns:
            A list of strings, where each string is a generated question.
        """
        if not llm_response_text:
            return []

        questions = []
        lines = llm_response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit(): # Starts with a digit
                # Find the first period, then take text after it.
                # This is more robust to "1.Question" vs "1. Question"
                try:
                    dot_index = line.index('.')
                    question_text = line[dot_index+1:].strip()
                    if question_text: 
                        questions.append(question_text)
                except ValueError:
                    # Line started with digit but no period found, might be a malformed line
                    # Or it could be part of a multi-line question if not careful.
                    # For simplicity, we'll skip such lines for now.
                    # If the LLM strictly follows "X. Question text", this is fine.
                    pass 
        
        # If parsing failed to get enough questions, but we have lines,
        # it might indicate a formatting issue from the LLM.
        if len(questions) < num_questions and len(questions) < len(lines):
            print(f"Warning: Parsed {len(questions)} questions, but expected {num_questions} and had {len(lines)} lines. LLM output format might differ from expected.")
            # Could add a fallback here to take raw lines if needed, but it's less reliable.

        return questions[:num_questions]


    async def generate_questions(self,
                                 context_chunks: List[Dict],
                                 num_questions: int = 3,
                                 question_type: str = "conceptual") -> List[str]:
        """
        Generates questions based on the provided context chunks.
        """
        if not context_chunks:
            print("No context chunks provided, cannot generate questions.")
            return []
        if num_questions <= 0:
            print("Number of questions must be positive.")
            return []

        prompt = self._build_prompt(context_chunks, num_questions, question_type)
        if not prompt:
            print("Failed to build prompt from context.")
            return []

        llm_response = await self._call_llm_api(prompt)

        if llm_response:
            return self._parse_llm_response(llm_response, num_questions)
        else:
            print("Failed to get a valid response from LLM.")
            return []

async def demo():
    print("--- RAG Question Generator Demo (Real LLM Call) ---")
    
    # For this demo to work with a real LLM, you might need to set GEMINI_API_KEY
    # in your src/config.py or as an environment variable if not using the default free tier.
    # Example: In src/config.py, add: GEMINI_API_KEY = "YOUR_ACTUAL_API_KEY"
    # If GEMINI_API_KEY is not set and self.api_key is not passed, it will use ""
    # which relies on Canvas environment injection for gemini-2.0-flash.

    generator = RAGQuestionGenerator() # Uses default model, potentially "" API key

    sample_chunks = [
        {"chunk_id": "c1", "chunk_text": "The theory of relativity was proposed by Albert Einstein. It has two main parts: special relativity and general relativity. Special relativity deals with the relationship between space and time for objects moving at constant speeds."},
        {"chunk_id": "c2", "chunk_text": "General relativity, published in 1915, is a theory of gravitation. It describes gravity not as a force, but as a curvature of spacetime caused by mass and energy."},
        {"chunk_id": "c3", "chunk_text": "Key concepts in relativity include time dilation, length contraction, and the equivalence principle. These have been experimentally verified numerous times."}
    ]
    
    print("\nGenerating 3 'conceptual' questions from sample context:")
    # This will make a real API call if GEMINI_API_KEY is valid or Canvas injects it.
    questions = await generator.generate_questions(sample_chunks, num_questions=3, question_type="conceptual")
    if questions:
        for i, q in enumerate(questions):
            print(f"  Q{i+1}: {q}")
    else:
        print("  No questions generated or an error occurred.")

    print("\n--- RAG Question Generator Demo Finished ---")

if __name__ == '__main__':
    import asyncio
    # Load environment variables if .env file exists (for API keys)
    if os.path.exists(".env"):
        print("Found .env file, attempting to load environment variables.")
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
                    if not hasattr(config, key): # Set in config if not already there
                         setattr(config, key, value)
        print("Environment variables from .env potentially loaded.")

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

