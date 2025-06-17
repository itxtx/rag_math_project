const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

const handleResponse = async (response) => {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error response.' }));
    throw new Error(errorData.detail || 'API request failed');
  }
  return response.json();
};

const api = {
  // Topics
  fetchTopics: async () => {
    const response = await fetch(`${API_BASE_URL}/topics`, {
      headers: {
        'X-API-Key': API_KEY
      }
    });
    return handleResponse(response);
  },

  // Learning Session
  startInteraction: async ({ learnerId, topicId }) => {
    const response = await fetch(`${API_BASE_URL}/interaction/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY
      },
      body: JSON.stringify({
        learner_id: learnerId,
        topic_id: topicId,
      }),
    });
    return handleResponse(response);
  },

  submitAnswer: async ({ learnerId, questionId, topicId, questionText, contextForEvaluation, learnerAnswer }) => {
    const response = await fetch(`${API_BASE_URL}/interaction/submit_answer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY
      },
      body: JSON.stringify({
        learner_id: learnerId,
        question_id: questionId,
        doc_id: topicId,
        question_text: questionText,
        context_for_evaluation: contextForEvaluation,
        learner_answer: learnerAnswer,
      }),
    });
    return handleResponse(response);
  },

  getNextQuestion: async ({ learnerId, topicId }) => {
    const response = await fetch(`${API_BASE_URL}/interaction/next_question`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY
      },
      body: JSON.stringify({
        learner_id: learnerId,
        topic_id: topicId,
      }),
    });
    return handleResponse(response);
  },
};

export default api; 