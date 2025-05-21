"use client"; // This directive marks the component as a Client Component in Next.js App Router

import React, { useState } from 'react';

// Shadcn UI components (assuming they are set up in your project)
// The import paths have been adjusted to use the '@/components' alias,
// which is the standard setup for Shadcn UI in Next.js.
// This requires your jsconfig.json or tsconfig.json to have the path alias configured.
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent, CardFooter, CardTitle, CardDescription } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { AlertDialog, AlertDialogAction, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog';
import { Progress } from '@/components/ui/progress'; // For loading indicator

// Main App component
function App() {
  // State variables for managing the application's data and UI
  const [learnerId, setLearnerId] = useState(''); // Stores the learner's ID
  const [question, setQuestion] = useState(null); // Stores the current question object from the API
  const [answer, setAnswer] = useState(''); // Stores the learner's typed answer
  const [feedback, setFeedback] = useState(null); // Stores the evaluation feedback from the API
  const [error, setError] = useState(null); // Stores any error messages
  const [loading, setLoading] = useState(false); // Indicates if an API call is in progress
  const [isErrorDialogOpen, setIsErrorDialogOpen] = useState(false); // Controls the visibility of the error dialog

  // Base URL for your FastAPI backend
  // IMPORTANT: Adjust this if your FastAPI server is running on a different host or port.
  const API_BASE_URL = 'http://localhost:8000/api/v1';

  /**
   * Handles the "Start Interaction" button click.
   * Sends a request to the /interaction/start endpoint to get the next question.
   */
  const handleStartInteraction = async () => {
    setLoading(true); // Set loading state to true
    setError(null); // Clear any previous errors
    setQuestion(null); // Clear previous question
    setAnswer(''); // Clear previous answer
    setFeedback(null); // Clear previous feedback

    try {
      // Make a POST request to the start interaction endpoint
      const response = await fetch(`${API_BASE_URL}/interaction/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ learner_id: learnerId }), // Send the learner ID in the request body
      });

      // Check if the response was successful (status code 2xx)
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start interaction.');
      }

      const data = await response.json();
      setQuestion(data); // Set the received question data
    } catch (err) {
      console.error('Error starting interaction:', err);
      setError(err.message); // Set the error message
      setIsErrorDialogOpen(true); // Open the error dialog
    } finally {
      setLoading(false); // Reset loading state
    }
  };

  /**
   * Handles the "Submit Answer" button click.
   * Sends the learner's answer to the /interaction/submit_answer endpoint for evaluation.
   */
  const handleSubmitAnswer = async () => {
    // Prevent submission if no question is loaded
    if (!question) {
      setError('No question to answer.');
      setIsErrorDialogOpen(true);
      return;
    }

    setLoading(true); // Set loading state to true
    setError(null); // Clear any previous errors
    setFeedback(null); // Clear previous feedback

    try {
      // Make a POST request to the submit answer endpoint
      const response = await fetch(`${API_BASE_URL}/interaction/submit_answer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          learner_id: learnerId,
          question_id: question.question_id,
          question_text: question.question_text,
          context_for_evaluation: question.context_for_evaluation,
          learner_answer: answer,
        }),
      });

      // Check if the response was successful
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to submit answer.');
      }

      const data = await response.json();
      setFeedback(data.evaluation.feedback); // Set the received feedback
    } catch (err) {
      console.error('Error submitting answer:', err);
      setError(err.message); // Set the error message
      setIsErrorDialogOpen(true); // Open the error dialog
    } finally {
      setLoading(false); // Reset loading state
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4 font-inter">
      <div className="w-full max-w-2xl space-y-6">
        {/* Application Title */}
        <h1 className="text-4xl font-extrabold text-center text-gray-800 drop-shadow-md">
          Adaptive Learning System
        </h1>

        {/* Learner ID Input Card */}
        <Card className="w-full rounded-xl shadow-lg border border-gray-200">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-gray-700">Get Started</CardTitle>
            <CardDescription className="text-gray-500">Enter your unique learner ID to begin your learning journey.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid w-full items-center gap-1.5">
              <Label htmlFor="learnerId" className="text-gray-700">Learner ID</Label>
              <Input
                type="text"
                id="learnerId"
                placeholder="e.g., learner-123"
                value={learnerId}
                onChange={(e) => setLearnerId(e.target.value)}
                className="rounded-md focus:ring-2 focus:ring-blue-300"
                disabled={loading}
              />
            </div>
          </CardContent>
          <CardFooter className="flex justify-end">
            <Button
              onClick={handleStartInteraction}
              disabled={!learnerId || loading}
              className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg shadow-md transition-all duration-200 ease-in-out transform hover:scale-105"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Starting...
                </>
              ) : (
                'Start Interaction'
              )}
            </Button>
          </CardFooter>
        </Card>

        {/* Loading Progress Bar */}
        {loading && (
          <Progress value={100} className="w-full animate-pulse h-2 bg-blue-200" />
        )}

        {/* Question Display Card */}
        {question && (
          <Card className="w-full rounded-xl shadow-lg border border-gray-200">
            <CardHeader>
              <CardTitle className="text-2xl font-bold text-gray-700">Current Question</CardTitle>
              <CardDescription className="text-gray-500">Reflect on the concept and provide your best answer.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <p className="text-gray-800 text-lg leading-relaxed">{question.question_text}</p>
              </div>
              <div className="grid w-full items-center gap-1.5">
                <Label htmlFor="answer" className="text-gray-700">Your Answer</Label>
                <Textarea
                  id="answer"
                  placeholder="Type your answer here..."
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  className="min-h-[100px] rounded-md focus:ring-2 focus:ring-blue-300"
                  disabled={loading}
                />
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button
                onClick={handleSubmitAnswer}
                disabled={loading || !answer.trim()}
                className="w-full sm:w-auto bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-6 rounded-lg shadow-md transition-all duration-200 ease-in-out transform hover:scale-105"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Submitting...
                  </>
                ) : (
                  'Submit Answer'
                )}
              </Button>
            </CardFooter>
          </Card>
        )}

        {/* Feedback Display Card */}
        {feedback && (
          <Card className="w-full rounded-xl shadow-lg border border-gray-200">
            <CardHeader>
              <CardTitle className="text-2xl font-bold text-gray-700">Evaluation Feedback</CardTitle>
              <CardDescription className="text-gray-500">Here's how you did on the last question.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <p className="text-blue-800 text-lg leading-relaxed">{feedback}</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Error Dialog */}
        <AlertDialog open={isErrorDialogOpen} onOpenChange={setIsErrorDialogOpen}>
          <AlertDialogContent className="rounded-xl shadow-lg">
            <AlertDialogHeader>
              <AlertDialogTitle className="text-red-600">Error</AlertDialogTitle>
              <AlertDialogDescription>
                {error}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogAction onClick={() => setIsErrorDialogOpen(false)} className="bg-red-500 hover:bg-red-600 text-white rounded-lg">
                Close
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
}

export default App;
