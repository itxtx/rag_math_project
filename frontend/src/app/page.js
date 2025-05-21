"use client"; // This directive marks the component as a Client Component in Next.js App Router

import React, { useState, useEffect, useCallback, useMemo } from 'react';
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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

// Lucide React icons
import { Sun, Moon } from 'lucide-react';

// Helper function to clean LaTeX strings
const cleanLatexString = (text) => {
  if (typeof text !== 'string') {
    return text;
  }
  return text.replace(/`(\${1,2}[^`]*?\${1,2})`/g, '$1');
};


// Main App component
function App() {
  // State variables
  const [learnerId, setLearnerId] = useState('');
  const [question, setQuestion] = useState(null); // Active question being answered
  const [rawQuestionResponse, setRawQuestionResponse] = useState(null); // Full API response for current/next question
  const [conceptContext, setConceptContext] = useState(null); // Context to display before a question
  const [answer, setAnswer] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isErrorDialogOpen, setIsErrorDialogOpen] = useState(false);
  const [isLearnerIdDialogOpen, setIsLearnerIdDialogOpen] = useState(true);
  const [theme, setTheme] = useState('light');

  const [availableTopics, setAvailableTopics] = useState([]);
  const [selectedTopicId, setSelectedTopicId] = useState('');
  const [topicsLoading, setTopicsLoading] = useState(true);
  const [topicsError, setTopicsError] = useState(null);

  const API_BASE_URL = 'http://localhost:8000/api/v1';

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  useEffect(() => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
    link.integrity = 'sha384-n8MV3dEkVvJoQWT9sJILfSTgA9rwYQq8qxbl8tB7vO0FrvPz9HTZkYet5PkeFNHS';
    link.crossOrigin = 'anonymous';
    document.head.appendChild(link);
    return () => {
      if (document.head.contains(link)) {
        document.head.removeChild(link);
      }
    };
  }, []);

  useEffect(() => {
    const fetchTopics = async () => {
      setTopicsLoading(true);
      setTopicsError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/topics`);
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error from topics endpoint.' }));
          throw new Error(errorData.detail || 'Failed to fetch topics.');
        }
        const data = await response.json();
        setAvailableTopics(data);
        if (data.length > 0 && !selectedTopicId) {
          setSelectedTopicId(data[0].topic_id);
        }
      } catch (err) {
        console.error('Error fetching topics:', err);
        setTopicsError(err.message);
      } finally {
        setTopicsLoading(false);
      }
    };
    fetchTopics();
  }, [API_BASE_URL]);

  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  const handleStartInteraction = useCallback(async () => {
    if (!learnerId.trim() || !selectedTopicId) {
      setError("Learner ID and Topic must be provided.");
      setIsErrorDialogOpen(true);
      return;
    }
    setLoading(true);
    setError(null);
    setQuestion(null);
    setConceptContext(null);
    setRawQuestionResponse(null);
    setAnswer('');
    setFeedback(null);

    try {
      const response = await fetch(`${API_BASE_URL}/interaction/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          learner_id: learnerId,
          topic_id: selectedTopicId,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error from start interaction.' }));
        throw new Error(errorData.detail || 'Failed to start interaction.');
      }
      const data = await response.json();
      setRawQuestionResponse(data); // Store the full response

      if (data.context_for_new_concept && data.context_for_new_concept.content_markdown) {
        setConceptContext(data.context_for_new_concept); // Show context first
        setQuestion(null); // Ensure question is not shown yet
      } else {
        setQuestion(data); // No context, show question directly
        setConceptContext(null);
      }
      setIsLearnerIdDialogOpen(false);
    } catch (err) {
      console.error('Error starting interaction:', err);
      setError(err.message);
      setIsErrorDialogOpen(true);
    } finally {
      setLoading(false);
    }
  }, [learnerId, selectedTopicId, API_BASE_URL]);

  const handleProceedToQuestion = useCallback(() => {
    if (rawQuestionResponse) {
      setQuestion(rawQuestionResponse); // Set the active question from the stored full response
    }
    setConceptContext(null); // Hide context view
  }, [rawQuestionResponse]);

  const handleSubmitAnswer = useCallback(async () => {
    if (!question || !question.question_id) {
      setError('No question to answer or question ID is missing.');
      setIsErrorDialogOpen(true);
      return;
    }
    if (!answer.trim()) {
      setError('Answer cannot be empty.');
      setIsErrorDialogOpen(true);
      return;
    }
    setLoading(true);
    setError(null);
    setFeedback(null);
    try {
      const response = await fetch(`${API_BASE_URL}/interaction/submit_answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          learner_id: learnerId,
          question_id: question.question_id,
          question_text: question.question_text,
          context_for_evaluation: question.context_for_evaluation,
          learner_answer: answer,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error from submit answer.' }));
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
  }, [learnerId, question, answer, API_BASE_URL]);

  const cleanedQuestionText = useMemo(() => {
    if (question && question.question_text) {
      return cleanLatexString(question.question_text);
    }
    return "";
  }, [question]);

  const cleanedConceptContextMarkdown = useMemo(() => {
    if (conceptContext && conceptContext.content_markdown) {
      return cleanLatexString(conceptContext.content_markdown);
    }
    return "";
  }, [conceptContext]);

  const cleanedFeedback = useMemo(() => {
    if (!feedback) return null;
    return {
      feedbackText: cleanLatexString(feedback.feedbackText),
      correctAnswer: feedback.correctAnswer ? cleanLatexString(feedback.correctAnswer) : null,
    };
  }, [feedback]);

  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-4 font-inter relative">
      <Button
        onClick={toggleTheme}
        variant="outline"
        size="icon"
        className="absolute top-4 left-4 rounded-full bg-card hover:bg-accent text-foreground border-border"
        aria-label="Toggle theme"
      >
        {theme === 'light' ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
      </Button>

      <div className="w-full max-w-2xl space-y-6">
        <AlertDialog open={isLearnerIdDialogOpen} onOpenChange={setIsLearnerIdDialogOpen}>
          <AlertDialogContent className="rounded-xl bg-card border-border">
            <AlertDialogHeader>
              <AlertDialogTitle className="text-2xl font-bold text-card-foreground">Get Started</AlertDialogTitle>
              <AlertDialogDescription className="text-muted-foreground">
                Enter your unique learner ID and select a topic to begin.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <div className="grid w-full items-center gap-4 py-4">
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
                    {topicsLoading && <SelectItem value="loading" disabled>Loading topics...</SelectItem>}
                    {topicsError && <SelectItem value="error" disabled>Error: {topicsError}</SelectItem>}
                    {!topicsLoading && !topicsError && availableTopics.length === 0 && (
                      <SelectItem value="no-topics" disabled>No topics available</SelectItem>
                    )}
                    {availableTopics.map((topic) => (
                      <SelectItem key={topic.topic_id} value={topic.topic_id}>
                        {topic.source_file.replace(/\.tex$/i, '').replace(/_/g, ' ')}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <AlertDialogFooter>
              <Button
                onClick={handleStartInteraction}
                disabled={!learnerId.trim() || loading || topicsLoading || !selectedTopicId || availableTopics.length === 0}
                className="w-full sm:w-auto bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-3 px-6 rounded-lg shadow-md transition-all duration-200 ease-in-out transform hover:scale-105 disabled:opacity-50 disabled:transform-none"
              >
                {loading && !conceptContext && !question ? (
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

        {loading && <Progress value={undefined} className="w-full h-2 bg-primary animate-pulse" />}

        {/* Concept Context Display */}
        {!loading && conceptContext && (
          <Card className="w-full rounded-xl border border-border bg-card">
            <CardHeader>
              <CardTitle className="text-2xl font-bold text-card-foreground">
                Context: {conceptContext.topic_name || "Review Material"}
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                Please review the following information before answering the question.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className="p-4 rounded-lg border border-border text-foreground text-base leading-relaxed markdown-body bg-background/30 max-h-[50vh] overflow-y-auto"
                key={(conceptContext.topic_name || "") + cleanedConceptContextMarkdown.substring(0,10)} 
              >
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {cleanedConceptContextMarkdown}
                </ReactMarkdown>
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button
                onClick={handleProceedToQuestion}
                className="w-full sm:w-auto bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-3 px-6 rounded-lg shadow-md transition-all duration-200 ease-in-out transform hover:scale-105"
              >
                Proceed to Question
              </Button>
            </CardFooter>
          </Card>
        )}

        {/* Question Display */}
        {!loading && question && !conceptContext && (
          <Card className="w-full rounded-xl border border-border bg-card">
            <CardHeader>
              <CardTitle className="text-2xl font-bold text-card-foreground">Question {question.question_number || ''}</CardTitle>
              <CardDescription className="text-muted-foreground">Provide your answer below.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className="p-4 rounded-lg border border-border text-foreground text-lg leading-relaxed markdown-body bg-background/30"
                key={question.question_id}
              >
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {cleanedQuestionText}
                </ReactMarkdown>
              </div>
              <div className="grid w-full items-center gap-1.5">
                <Label htmlFor="answer" className="text-card-foreground">Your Answer</Label>
                <Textarea
                  id="answer"
                  placeholder="Type your answer here..."
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  className="min-h-[150px] rounded-md focus:ring-2 focus:ring-ring bg-input text-foreground border-border"
                  disabled={loading}
                />
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button
                onClick={handleSubmitAnswer}
                disabled={loading || !answer.trim()}
                className="w-full sm:w-auto bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-3 px-6 rounded-lg shadow-md transition-all duration-200 ease-in-out transform hover:scale-105 disabled:opacity-50 disabled:transform-none"
              >
                {loading && feedback === null ? (
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

        {/* Feedback Display */}
        {!loading && cleanedFeedback && !conceptContext && (
          <Card className="w-full rounded-xl border border-border bg-card mt-6">
            <CardHeader>
              <CardTitle className="text-2xl font-bold text-card-foreground">Evaluation Feedback</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className="p-4 rounded-lg border border-border text-foreground text-lg leading-relaxed markdown-body bg-background/30"
                key={(cleanedFeedback.feedbackText || "") + (cleanedFeedback.correctAnswer || "")}
              >
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {`**Feedback:** ${cleanedFeedback.feedbackText}`}
                </ReactMarkdown>

                {cleanedFeedback.correctAnswer && (
                  <div className="mt-4 pt-4 border-t border-border">
                    <ReactMarkdown
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {`**Correct Answer:** ${cleanedFeedback.correctAnswer}`}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button
                onClick={handleStartInteraction}
                disabled={loading}
                className="w-full sm:w-auto bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-3 px-6 rounded-lg shadow-md transition-all duration-200 ease-in-out transform hover:scale-105 disabled:opacity-50 disabled:transform-none"
              >
                {loading ? 'Loading Next...' : 'Next Question'}
              </Button>
            </CardFooter>
          </Card>
        )}

        <AlertDialog open={isErrorDialogOpen} onOpenChange={setIsErrorDialogOpen}>
          <AlertDialogContent className="rounded-xl bg-card border-border">
            <AlertDialogHeader>
              <AlertDialogTitle className="text-destructive font-semibold">Error</AlertDialogTitle>
              <AlertDialogDescription className="text-muted-foreground py-2">
                {error || "An unexpected error occurred."}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <Button onClick={() => setIsErrorDialogOpen(false)} className="bg-destructive hover:bg-destructive/90 text-destructive-foreground rounded-lg w-full sm:w-auto">
                Close
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
}

export default App;