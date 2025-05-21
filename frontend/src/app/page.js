"use client"; // This directive marks the component as a Client Component in Next.js App Router

import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown'; // Import ReactMarkdown
import remarkMath from 'remark-math'; // Import remark-math for parsing math
import rehypeKatex from 'rehype-katex'; // Import rehype-katex for rendering math

// Shadcn UI components
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent, CardFooter, CardTitle, CardDescription } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { AlertDialog, AlertDialogAction, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'; // Import Select components

// Lucide React icons
import { Sun, Moon } from 'lucide-react';

// Main App component
function App() {
  // State variables for managing the application's data and UI
  const [learnerId, setLearnerId] = useState('');
  const [question, setQuestion] = useState(null);
  const [answer, setAnswer] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isErrorDialogOpen, setIsErrorDialogOpen] = useState(false);
  const [isLearnerIdDialogOpen, setIsLearnerIdDialogOpen] = useState(true);
  const [theme, setTheme] = useState('light');

  // New state for topic selection
  const [availableTopics, setAvailableTopics] = useState([]);
  const [selectedTopicId, setSelectedTopicId] = useState('');
  const [topicsLoading, setTopicsLoading] = useState(true);
  const [topicsError, setTopicsError] = useState(null);

  // Base URL for your FastAPI backend
  const API_BASE_URL = 'http://localhost:8000/api/v1';

  // Effect to apply the 'dark' class to the html element based on the theme state
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  // Effect to load KaTeX CSS dynamically
  useEffect(() => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
    link.integrity = 'sha384-n8MV3dEkVvJoQWT9sJILfSTgA9rwYQq8qxbl8tB7vO0FrvPz9HTZkYet5PkeFNHS';
    link.crossOrigin = 'anonymous';
    document.head.appendChild(link);

    return () => {
      document.head.removeChild(link);
    };
  }, []);

  // Effect to fetch available topics on component mount
  useEffect(() => {
    const fetchTopics = async () => {
      setTopicsLoading(true);
      setTopicsError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/topics`);
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to fetch topics.');
        }
        const data = await response.json();
        setAvailableTopics(data);
        // Optionally pre-select the first topic if available
        if (data.length > 0) {
          setSelectedTopicId(data[0].topic_id);
        }
      } catch (err) {
        console.error('Error fetching topics:', err);
        setTopicsError(err.message);
        setError(`Failed to load topics: ${err.message}`);
        setIsErrorDialogOpen(true);
      } finally {
        setTopicsLoading(false);
      }
    };

    fetchTopics();
  }, []); // Run once on mount

  /**
   * Toggles the current theme between 'light' and 'dark'.
   */
  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  /**
   * Handles the "Start Interaction" button click.
   * Sends a request to the /interaction/start endpoint to get the next question.
   */
  const handleStartInteraction = async () => {
    setLoading(true);
    setError(null);
    setQuestion(null);
    setAnswer('');
    setFeedback(null);

    try {
      // API change: topic_id is now in the request body, not a query parameter.
      const response = await fetch(`${API_BASE_URL}/interaction/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          learner_id: learnerId,
          topic_id: selectedTopicId || null, // Send selectedTopicId in the body
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start interaction.');
      }

      const data = await response.json();
      setQuestion(data);
      setIsLearnerIdDialogOpen(false); // Close the learner ID dialog on successful start
    } catch (err) {
      console.error('Error starting interaction:', err);
      setError(err.message);
      setIsErrorDialogOpen(true);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handles the "Submit Answer" button click.
   * Sends the learner's answer to the /interaction/submit_answer endpoint for evaluation.
   */
  const handleSubmitAnswer = async () => {
    if (!question) {
      setError('No question to answer.');
      setIsErrorDialogOpen(true);
      return;
    }

    setLoading(true);
    setError(null);
    setFeedback(null);

    try {
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

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to submit answer.');
      }

      const data = await response.json();
      setFeedback({
        feedbackText: data.evaluation.feedback,
        correctAnswer: data.evaluation['correct_answer'] || data.evaluation['correct answer'] || null,
      });
    } catch (err) {
      console.error('Error submitting answer:', err);
      setError(err.message);
      setIsErrorDialogOpen(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-4 font-inter relative">
      {/* Theme Toggle Button */}
      <Button
        onClick={toggleTheme}
        variant="outline"
        size="icon"
        className="absolute top-4 left-4 rounded-full shadow-md bg-card hover:bg-accent text-foreground border-border"
      >
        {theme === 'light' ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
      </Button>

      <div className="w-full max-w-2xl space-y-6">
        {/* Learner ID & Topic Selection Dialog */}
        <AlertDialog open={isLearnerIdDialogOpen} onOpenChange={setIsLearnerIdDialogOpen}>
          <AlertDialogContent className="rounded-xl shadow-lg bg-card border-border">
            <AlertDialogHeader>
              <AlertDialogTitle className="text-2xl font-bold text-card-foreground">Get Started</AlertDialogTitle>
              <AlertDialogDescription className="text-muted-foreground">
                Enter your unique learner ID and select a topic to begin your learning journey.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <div className="grid w-full items-center gap-4">
              <div className="flex flex-col space-y-1.5">
                <Label htmlFor="learnerIdDialog" className="text-card-foreground">Learner ID</Label>
                <Input
                  type="text"
                  id="learnerIdDialog"
                  placeholder="e.g., learner-123"
                  value={learnerId}
                  onChange={(e) => setLearnerId(e.target.value)}
                  className="rounded-md focus:ring-2 focus:ring-ring bg-input text-foreground border-border"
                  disabled={loading || topicsLoading}
                />
              </div>
              <div className="flex flex-col space-y-1.5">
                <Label htmlFor="topicSelect" className="text-card-foreground">Select Topic</Label>
                <Select
                  onValueChange={setSelectedTopicId}
                  value={selectedTopicId}
                  disabled={loading || topicsLoading || availableTopics.length === 0}
                >
                  <SelectTrigger id="topicSelect" className="rounded-md focus:ring-2 focus:ring-ring bg-input text-foreground border-border">
                    <SelectValue placeholder={topicsLoading ? "Loading topics..." : topicsError ? "Error loading topics" : "Select a topic"} />
                  </SelectTrigger>
                  <SelectContent className="rounded-md bg-popover text-popover-foreground border-border">
                    {availableTopics.length === 0 && !topicsLoading && !topicsError && (
                      <SelectItem value="no-topics" disabled>No topics available</SelectItem>
                    )}
                    {topicsError && (
                      <SelectItem value="error-loading" disabled>{`Error: ${topicsError}`}</SelectItem>
                    )}
                    {availableTopics.map((topic) => (
                      <SelectItem key={topic.topic_id} value={topic.topic_id}>
                        {topic.source_file.replace('.tex', '').replace(/_/g, ' ')}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <AlertDialogFooter>
              <Button
                onClick={handleStartInteraction}
                disabled={!learnerId || loading || topicsLoading || !selectedTopicId}
                className="w-full sm:w-auto bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-2 px-6 rounded-lg shadow-md transition-all duration-200 ease-in-out transform hover:scale-105"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-primary-foreground" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Starting...
                  </>
                ) : (
                  'Start Interaction'
                )}
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        {/* Loading Progress Bar */}
        {loading && (
          <Progress value={100} className="w-full animate-pulse h-2 bg-primary" />
        )}

        {/* Question Display Card */}
        {question && (
          <Card className="w-full rounded-xl shadow-lg border border-border bg-card">
            <CardHeader>
              <CardTitle className="text-2xl font-bold text-card-foreground">Current Question</CardTitle>
              <CardDescription className="text-muted-foreground">Reflect on the concept and provide your best answer.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className="p-4 rounded-lg border border-border text-foreground text-lg leading-relaxed markdown-body"
                key={question.question_text}
              >
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {question.question_text}
                </ReactMarkdown>
              </div>
              <div className="grid w-full items-center gap-1.5">
                <Label htmlFor="answer" className="text-card-foreground">Your Answer</Label>
                <Textarea
                  id="answer"
                  placeholder="Type your answer here..."
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  className="min-h-[200px] rounded-md focus:ring-2 focus:ring-ring bg-transparent text-foreground border-border"
                  disabled={loading}
                />
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button
                onClick={handleSubmitAnswer}
                disabled={loading || !answer.trim()}
                className="w-full sm:w-auto bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-2 px-6 rounded-lg shadow-md transition-all duration-200 ease-in-out transform hover:scale-105"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-primary-foreground" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
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
          <Card className="w-full rounded-xl shadow-lg border border-border bg-card">
            <CardHeader>
              <CardTitle className="text-2xl font-bold text-card-foreground">Evaluation Feedback</CardTitle>
              <CardDescription className="text-muted-foreground">Here's how you did on the last question.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className="p-4 rounded-lg border border-border text-foreground text-lg leading-relaxed markdown-body"
                key={feedback.feedbackText + (feedback.correctAnswer || '')}
              >
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {`**Feedback:** ${feedback.feedbackText}`}
                </ReactMarkdown>

                {feedback.correctAnswer && (
                  <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    className="mt-2"
                  >
                    {`**Correct Answer:** ${feedback.correctAnswer}`}
                  </ReactMarkdown>
                )}
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button
                onClick={handleStartInteraction}
                disabled={loading}
                className="w-full sm:w-auto bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-2 px-6 rounded-lg shadow-md transition-all duration-200 ease-in-out transform hover:scale-105"
              >
                {loading ? 'Loading Next Question...' : 'Next Question'}
              </Button>
            </CardFooter>
          </Card>
        )}

        {/* Error Dialog */}
        <AlertDialog open={isErrorDialogOpen} onOpenChange={setIsErrorDialogOpen}>
          <AlertDialogContent className="rounded-xl shadow-lg bg-card border-border">
            <AlertDialogHeader>
              <AlertDialogTitle className="text-destructive">Error</AlertDialogTitle>
              <AlertDialogDescription className="text-muted-foreground">
                {error}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogAction onClick={() => setIsErrorDialogOpen(false)} className="bg-destructive hover:bg-destructive/90 text-destructive-foreground rounded-lg">
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