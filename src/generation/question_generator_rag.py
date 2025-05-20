import json
from typing import List, Dict, Optional, Any
# We'll need to make an API call to an LLM.
# For now, this is a placeholder for how you might integrate it.
# from google.generativeai import GenerativeModel # Example, replace with actual library/API call

from src import config # For API keys or model names, if any

class RAGQuestionGenerator:
    """
    Generates questions based on provided text chunks using a RAG approach with an LLM.
    """

    # Potentially load model name from config
    LLM_MODEL_NAME = config.QUESTION_GENERATION_LLM_MODEL if hasattr(config, 'QUESTION_GENERATION_LLM_MODEL') else "gemini-2.0-flash"


    def __init__(self, llm_api_key: Optional[str] = None, llm_model_name: Optional[str] = None):
        """
        Initializes the RAGQuestionGenerator.

        Args:
            llm_api_key: API key for the LLM service. If not provided, it might be
                         sourced from environment variables or a config file.
            llm_model_name: The specific LLM model to use for question generation.
        """
        # Store API key if provided, or rely on environment/global config
        self.api_key = llm_api_key if llm_api_key else getattr(config, 'LLM_API_KEY', None)
        
        # Determine the model name
        if llm_model_name:
            self.model_name = llm_model_name
        elif hasattr(config, 'QUESTION_GENERATION_LLM_MODEL'):
            self.model_name = config.QUESTION_GENERATION_LLM_MODEL
        else:
            self.model_name = "gemini-2.0-flash" # Default model

        # Placeholder for LLM client initialization if needed
        # self.llm_client = self._initialize_llm_client()
        print(f"RAGQuestionGenerator initialized with model: {self.model_name}")


    # def _initialize_llm_client(self):
    #     """
    #     Initializes and returns the LLM client.
    #     This is a placeholder and depends on the specific LLM service.
    #     """
    #     # Example for a hypothetical LLM client
    #     # if self.api_key:
    #     #     return GenerativeModel(model_name=self.model_name, api_key=self.api_key)
    #     # else:
    #     #     # Attempt to initialize without explicit key if supported (e.g., via env vars)
    #     #     return GenerativeModel(model_name=self.model_name)
    #     print("LLM client initialization placeholder.")
    #     return None

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

        # Prompt Engineering: This is a critical part.
        # The prompt should clearly instruct the LLM on its task.
        prompt = f"""Based on the following context, please generate {num_questions} distinct and insightful {question_type} questions.
Each question should be answerable primarily from the provided text.
Avoid questions that are too simple (yes/no) unless specifically asking for a definition or confirmation of a fact stated.
Focus on understanding, application, or elaboration of the concepts presented.

Context:
\"\"\"
{context_str}
\"\"\"

Generate exactly {num_questions} questions:
1.
2.
3.
...
"""
        # If num_questions is greater than 3, ensure the example list continues
        if num_questions > 3:
            for i in range(4, num_questions + 1):
                prompt += f"{i}.\n"
        
        return prompt

    async def _call_llm_api(self, prompt: str) -> Optional[str]:
        """
        Makes an API call to the LLM with the given prompt.
        This is a placeholder and needs to be implemented based on the chosen LLM API.

        Args:
            prompt: The prompt string to send to the LLM.

        Returns:
            The raw text response from the LLM, or None if an error occurs.
        """
        if not prompt:
            return None
        
        print(f"\n--- Sending Prompt to LLM ({self.model_name}) ---")
        # print(prompt) # For debugging, can be verbose
        print("--- End of Prompt ---")

        chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
        payload = {"contents": chat_history}
        
        # Use the API key from self.api_key if it's set, otherwise, it will be an empty string
        # which is fine for gemini-2.0-flash as Canvas will provide it.
        api_key_to_use = self.api_key if self.api_key else ""
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key_to_use}"

        try:
            # Using a simple fetch for demonstration.
            # In a real application, you might use a library like 'requests' or 'aiohttp'.
            # For simplicity, this is a synchronous-like fetch call placeholder.
            # Actual implementation would need an async HTTP client if this method is async.
            
            # This is a simplified representation of an async fetch call.
            # You'd typically use `aiohttp` or similar in an async context.
            # For now, we'll simulate the structure.
            
            # Placeholder for actual fetch call:
            # response = await fetch_wrapper(api_url, method='POST', body=json.dumps(payload), headers={'Content-Type': 'application/json'})
            # result = await response.json()
            
            # Simulating the fetch call structure for now
            # In a real async environment, replace this with actual async HTTP request
            response_obj = await self._execute_fetch(api_url, payload)
            
            if response_obj.get("candidates") and \
               len(response_obj["candidates"]) > 0 and \
               response_obj["candidates"][0].get("content") and \
               response_obj["candidates"][0]["content"].get("parts") and \
               len(response_obj["candidates"][0]["content"]["parts"]) > 0:
                generated_text = response_obj["candidates"][0]["content"]["parts"][0].get("text")
                print("LLM Response Received.")
                return generated_text
            else:
                print("Error: Unexpected LLM API response structure.")
                # print("Full LLM Response:", response_obj) # For debugging
                return None
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            # print("Full LLM Response (on error):", response_obj if 'response_obj' in locals() else "N/A")
            return None

    async def _execute_fetch(self, url: str, payload: Dict):
        """
        A helper to simulate or execute the actual fetch.
        In a real scenario, this would use an HTTP library like aiohttp.
        This is a simplified placeholder.
        """
        # This is where you'd use `aiohttp.ClientSession().post(...)` or similar
        # For the purpose of this structure, we'll assume a global `fetch` is available
        # or this method gets replaced by actual HTTP client usage.
        # This is a conceptual placeholder for the actual async fetch.
        
        # The actual fetch needs to be implemented in the environment where this code runs.
        # For example, in a browser context, `fetch` is global. In Node.js, you'd import it.
        # In Python backend, you'd use `requests` (sync) or `aiohttp` (async).
        
        # This is a simplified way to make it runnable IF a global fetch is polyfilled or available.
        # It's NOT robust for production Python server-side code without a proper HTTP client.
        class MockResponse:
            def __init__(self, status_code, json_data):
                self._status = status_code
                self._json_data = json_data
            async def json(self): return self._json_data
            @property
            def ok(self): return 200 <= self._status < 300

        # This is where the actual `fetch` call would be.
        # Since `fetch` is not standard in Python server-side, this will need adjustment
        # based on the execution environment (e.g., using `aiohttp`).
        # For now, let's assume a placeholder that would need to be filled.
        # We can't directly call browser `fetch` here.
        
        # To make this runnable in a Python environment without a browser's fetch:
        # We will use a simplified structure that mimics the Gemini API call.
        # This part needs to be replaced with actual `aiohttp` or `requests` call in a real app.
        
        # This is a conceptual representation. The actual `fetch` call is environment-dependent.
        # For Gemini API, the structure is:
        # response = await fetch(apiUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        # result = await response.json();
        
        # This is a highly simplified placeholder.
        # print(f"Mock Fetch: POST to {url} with payload: {json.dumps(payload)}")
        if "gemini-2.0-flash:generateContent" in url:
             # Simulate a successful response structure
            return {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "1. What is the main topic?\n2. Explain concept X.\n3. How does Y relate to Z?"}],
                        "role": "model"
                    },
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "NEGLIGIBLE"},] # ... more ratings]
                }],
                # "promptFeedback": { ... } # Optional
            }
        return {"error": "Mock fetch URL not recognized"}


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
            # Try to match lines starting with a number and a period (e.g., "1. ", "01. ")
            # and then capture the rest of the line as the question.
            if line and line[0].isdigit():
                parts = line.split('.', 1)
                if len(parts) > 1 and parts[0].isdigit():
                    question_text = parts[1].strip()
                    if question_text: # Ensure it's not an empty question
                        questions.append(question_text)
        
        # Fallback or alternative parsing if the above is too strict or LLM format varies
        if not questions and len(lines) >= num_questions:
            # If no numbered list was found, but we have enough lines,
            # assume each line is a question (less robust).
            # This is a simple fallback, more sophisticated parsing might be needed.
            # questions = [line.strip() for line in lines if line.strip()]
            pass # For now, rely on the numbered list format.

        # Ensure we don't return more questions than requested if parsing is loose
        return questions[:num_questions]


    async def generate_questions(self,
                                 context_chunks: List[Dict],
                                 num_questions: int = 3,
                                 question_type: str = "conceptual") -> List[str]:
        """
        Generates questions based on the provided context chunks.

        Args:
            context_chunks: A list of dictionaries, each representing a retrieved text chunk.
                            Each chunk dictionary is expected to have a "chunk_text" key.
            num_questions: The desired number of questions to generate.
            question_type: The type of questions (e.g., "conceptual", "factual").

        Returns:
            A list of generated question strings.
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

# Example Usage (Async context needed to run this directly)
async def demo():
    print("--- RAG Question Generator Demo ---")

    # Initialize with default model and no explicit API key (relies on environment or Canvas providing it)
    generator = RAGQuestionGenerator()

    # Sample context chunks (normally retrieved by the Retriever)
    sample_chunks = [
        {"chunk_id": "c1", "chunk_text": "The theory of relativity was proposed by Albert Einstein. It has two main parts: special relativity and general relativity. Special relativity deals with the relationship between space and time for objects moving at constant speeds."},
        {"chunk_id": "c2", "chunk_text": "General relativity, published in 1915, is a theory of gravitation. It describes gravity not as a force, but as a curvature of spacetime caused by mass and energy."},
        {"chunk_id": "c3", "chunk_text": "Key concepts in relativity include time dilation, length contraction, and the equivalence principle. These have been experimentally verified numerous times."}
    ]
    
    sample_chunks_empty_text = [
        {"chunk_id": "c1", "chunk_text": "  "},
        {"chunk_id": "c2", "chunk_text": ""}
    ]

    print("\nGenerating 3 'conceptual' questions from sample context:")
    questions = await generator.generate_questions(sample_chunks, num_questions=3, question_type="conceptual")
    if questions:
        for i, q in enumerate(questions):
            print(f"  Q{i+1}: {q}")
    else:
        print("  No questions generated or an error occurred.")

    print("\nGenerating 2 'factual' questions from sample context:")
    questions_factual = await generator.generate_questions(sample_chunks, num_questions=2, question_type="factual")
    if questions_factual:
        for i, q in enumerate(questions_factual):
            print(f"  Q{i+1}: {q}")
    else:
        print("  No factual questions generated or an error occurred.")
        
    print("\nGenerating questions from context with empty text chunks:")
    questions_empty = await generator.generate_questions(sample_chunks_empty_text, num_questions=2)
    if questions_empty:
        for i, q in enumerate(questions_empty):
            print(f"  Q{i+1}: {q}")
    else:
        print("  No questions generated (as expected due to empty context).")

    print("\n--- RAG Question Generator Demo Finished ---")

if __name__ == '__main__':
    # To run this demo, you need an async event loop.
    # For example, using asyncio:
    import asyncio
    try:
        asyncio.run(demo())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("Demo cannot be run directly from a running event loop (e.g. Jupyter).")
            print("Try running from a standard Python script.")
        else:
            raise
