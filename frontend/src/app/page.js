"use client"; // This directive marks the component as a Client Component in Next.js App Router

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import ReactMarkdown from 'react-markdown'; // Import ReactMarkdown
import remarkMath from 'remark-math'; // Import remark-math for parsing math
import rehypeKatex from 'rehype-katex'; // Import rehype-katex for rendering math
import 'katex/dist/katex.css'; // Import KaTeX CSS directly

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

// Import our new components
import LearnerIdModal from '@/components/domain/LearnerIdModal';
import QuestionCard from '@/components/domain/QuestionCard';
import FeedbackCard from '@/components/domain/FeedbackCard';
import ContextView from '@/components/domain/ContextView';

// Helper function to clean LaTeX strings
const cleanLatexString = (text) => {
  if (typeof text !== 'string') {
    return text;
  }
  // First, remove any backticks around LaTeX expressions, which is good practice
  let cleanedText = text.replace(/`(\${1,2}[^`]*?\${1,2})`/g, '$1');

  // Normalize multiple dollar signs.
  // This regex finds three or more consecutive dollar signs and replaces them with two.
  cleanedText = cleanedText.replace(/\${3,}/g, '$$');
  
  // You could also add a rule for single dollar signs if you want all math to be display style:
  // cleanedText = cleanedText.replace(/(?<!\$)\$([^\$]+)\$(?!\$)/g, '$$$1$$');

  return cleanedText;
};


// Main App component
function App() {
  // State variables
  const [learnerId, setLearnerId] = useState('');
  const [question, setQuestion] = useState(null);
  const [rawQuestionResponse, setRawQuestionResponse] = useState(null);
  const [conceptContext, setConceptContext] = useState(null);
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

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;
  const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  // Effect to load custom serif font (KaTeX CSS is now imported)
  useEffect(() => {
    // Load Lora font for LaTeX-compatible text
    const loraFontLink = document.createElement('link');
    loraFontLink.href = "https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&display=swap";
    loraFontLink.rel = "stylesheet";
    document.head.appendChild(loraFontLink);

    return () => {
      if (document.head.contains(loraFontLink)) {
        document.head.removeChild(loraFontLink);
      }
    };
  }, []);

  useEffect(() => {
    const fetchTopics = async () => {
      setTopicsLoading(true);
      setTopicsError(null);
      try {
        console.log('Fetching topics from:', `${API_BASE_URL}/topics`);
        const response = await fetch(`${API_BASE_URL}/topics`, {
          headers: {
            'X-API-Key': API_KEY
          }
        });
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error from topics endpoint.' }));
          throw new Error(errorData.detail || 'Failed to fetch topics.');
        }
        const data = await response.json();
        console.log('Received topics data:', data);
        
        if (!Array.isArray(data)) {
          console.error('Topics data is not an array:', data);
          throw new Error('Invalid topics data format');
        }
        
        // Filter out any invalid topics
        const validTopics = data.filter(topic => 
          topic && typeof topic === 'object' && 
          topic.doc_id && 
          (topic.title || topic.doc_id)
        );
        
        console.log('Valid topics:', validTopics);
        setAvailableTopics(validTopics);
        
        if (validTopics.length > 0 && !selectedTopicId) {
          setSelectedTopicId(validTopics[0].doc_id);
        }
      } catch (err) {
        console.error('Error fetching topics:', err);
        setTopicsError(err.message);
        setAvailableTopics([]);
      } finally {
        setTopicsLoading(false);
      }
    };
    fetchTopics();
  }, [API_BASE_URL, API_KEY]);

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
        headers: { 
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY
        },
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
      setRawQuestionResponse(data);

      if (data.is_new_concept_context_presented === true && data.context_for_evaluation) {
        setConceptContext({
          topic_name: data.concept_name,
          content_markdown: data.context_for_evaluation
        });
        setQuestion(null);
      } else {
        setQuestion(data);
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
  }, [learnerId, selectedTopicId, API_BASE_URL, API_KEY]);

  const handleProceedToQuestion = useCallback(() => {
    if (rawQuestionResponse) {
      setQuestion(rawQuestionResponse);
    }
    setConceptContext(null);
  }, [rawQuestionResponse]);

  const handleSubmitAnswer = useCallback(async () => {
    if (!question || !question.question_id) {
      setError('No question to answer or question ID is missing.');
      setIsErrorDialogOpen(true);
      return;
    }
    if (!selectedTopicId) {
      setError('Topic ID is missing. Cannot submit answer.');
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
        headers: { 
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY
        },
        body: JSON.stringify({
          learner_id: learnerId,
          question_id: question.question_id,
          doc_id: selectedTopicId,
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
  }, [learnerId, question, answer, API_BASE_URL, API_KEY, selectedTopicId]);

  const handleNextQuestion = useCallback(async () => {
    setLoading(true);
    setError(null);
    setQuestion(null);
    setAnswer('');
    setFeedback(null);

    try {
      const response = await fetch(`${API_BASE_URL}/interaction/next_question`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY
        },
        body: JSON.stringify({
          learner_id: learnerId,
          topic_id: selectedTopicId,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error from next question.' }));
        throw new Error(errorData.detail || 'Failed to get next question.');
      }
      const data = await response.json();
      setRawQuestionResponse(data);

      if (data.is_new_concept_context_presented === true && data.context_for_evaluation) {
        setConceptContext({
          topic_name: data.concept_name,
          content_markdown: data.context_for_evaluation
        });
        setQuestion(null);
      } else {
        setQuestion(data);
        setConceptContext(null);
      }
    } catch (err) {
      console.error('Error getting next question:', err);
      setError(err.message);
      setIsErrorDialogOpen(true);
    } finally {
      setLoading(false);
    }
  }, [learnerId, selectedTopicId, API_BASE_URL, API_KEY]);

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
    <>
      {/* Style tag to define the Lora font class and apply it */}
      <style jsx global>{`
        .font-latex-serif {
          font-family: 'Lora', serif;
          line-height: 1.7; 
        }
        .font-latex-serif p, 
        .font-latex-serif li {
            font-size: 1.05rem; 
        }
        textarea.font-latex-serif {
          font-family: 'Lora', serif;
          line-height: 1.6;
          font-size: 1.05rem;
        }
        .font-latex-serif .katex {
          font-family: KaTeX_Main, Times New Roman, serif !important; 
        }
      `}</style>
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

        <div className="w-full max-w-3xl space-y-6"> {/* Adjusted max-width for the main content area */}
          <LearnerIdModal
            learnerId={learnerId}
            setLearnerId={setLearnerId}
            availableTopics={availableTopics}
            selectedTopicId={selectedTopicId}
            setSelectedTopicId={setSelectedTopicId}
            topicsLoading={topicsLoading}
            onStartInteraction={handleStartInteraction}
            isOpen={isLearnerIdDialogOpen}
          />

          {loading && <Progress value={undefined} className="w-full h-2 bg-primary animate-pulse" />}

          {!loading && conceptContext && (
            <ContextView
              conceptContext={conceptContext}
              onProceedToQuestion={handleProceedToQuestion}
            />
          )}

          {!loading && question && !conceptContext && ( 
            <QuestionCard
              question={question}
              answer={answer}
              setAnswer={setAnswer}
              onSubmitAnswer={handleSubmitAnswer}
              loading={loading}
            />
          )}

          {!loading && cleanedFeedback && !conceptContext && (
            <FeedbackCard
              feedback={cleanedFeedback}
              onNextQuestion={handleNextQuestion}
              onReviewMaterial={() => setConceptContext({
                topic_name: question.concept_name,
                content_markdown: question.context_for_evaluation
              })}
              onShowCorrectAnswer={() => setFeedback(prev => ({
                ...prev,
                showCorrectAnswer: true
              }))}
            />
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
    </>
  );
}

export default App;
