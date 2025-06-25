// frontend/src/components/FeedbackCard.jsx
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ThumbsUp, ThumbsDown, MessageCircle, TrendingUp, Brain, Target } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

const FeedbackCard = ({ 
  interactionId, 
  onFeedbackSubmitted, 
  showRewardBreakdown = false,
  className = "" 
}) => {
  const [feedbackStatus, setFeedbackStatus] = useState('pending'); // pending, submitting, submitted, error
  const [submittedRating, setSubmittedRating] = useState(null);
  const [rewardData, setRewardData] = useState(null);
  const [error, setError] = useState(null);

  const submitFeedback = async (rating) => {
    if (feedbackStatus !== 'pending') return;
    
    setFeedbackStatus('submitting');
    setError(null);

    try {
      const response = await fetch(`/api/v1/interaction/${interactionId}/rate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ rating })
      });

      if (!response.ok) {
        throw new Error(`Failed to submit feedback: ${response.statusText}`);
      }

      const result = await response.json();
      
      setSubmittedRating(rating);
      setRewardData(result);
      setFeedbackStatus('submitted');
      
      // Notify parent component
      if (onFeedbackSubmitted) {
        onFeedbackSubmitted(rating, result);
      }

    } catch (err) {
      console.error('Error submitting feedback:', err);
      setError(err.message);
      setFeedbackStatus('error');
    }
  };

  const renderRewardBreakdown = () => {
    if (!showRewardBreakdown || !rewardData?.reward_components) return null;

    const { reward_components } = rewardData;
    
    return (
      <div className="mt-4 p-3 bg-slate-50 rounded-lg border">
        <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
          <Brain className="w-4 h-4" />
          AI Learning Feedback
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1">
              <MessageCircle className="w-3 h-3" />
              Engagement
            </span>
            <span className={`font-mono ${reward_components.vote_reward >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {reward_components.vote_reward.toFixed(2)}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              Learning
            </span>
            <span className={`font-mono ${reward_components.learning_reward >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {reward_components.learning_reward.toFixed(2)}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1">
              <Target className="w-3 h-3" />
              Effort
            </span>
            <span className="font-mono text-blue-600">
              {reward_components.effort_reward.toFixed(2)}
            </span>
          </div>
        </div>
        <div className="mt-2 pt-2 border-t flex items-center justify-between">
          <span className="text-sm font-medium">Total Reward</span>
          <span className={`font-mono font-bold ${reward_components.total_reward >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {reward_components.total_reward.toFixed(3)}
          </span>
        </div>
      </div>
    );
  };

  const getFeedbackMessage = () => {
    if (feedbackStatus === 'submitted') {
      const isPositive = submittedRating === 'up';
      return {
        type: isPositive ? 'success' : 'info',
        message: isPositive 
          ? "Thanks! Your feedback helps the AI learn better teaching strategies." 
          : "Thanks for the feedback! The AI will adjust its approach."
      };
    }
    return null;
  };

  const feedbackMessage = getFeedbackMessage();

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <MessageCircle className="w-5 h-5" />
          How was this question?
        </CardTitle>
        <CardDescription>
          Your feedback helps our AI learn to ask better questions for you.
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {feedbackStatus === 'pending' && (
          <div className="flex gap-3 justify-center">
            <Button
              onClick={() => submitFeedback('up')}
              variant="outline"
              size="lg"
              className="flex-1 flex items-center gap-2 hover:bg-green-50 hover:border-green-300 hover:text-green-700"
            >
              <ThumbsUp className="w-5 h-5" />
              Helpful
            </Button>
            <Button
              onClick={() => submitFeedback('down')}
              variant="outline"
              size="lg"
              className="flex-1 flex items-center gap-2 hover:bg-red-50 hover:border-red-300 hover:text-red-700"
            >
              <ThumbsDown className="w-5 h-5" />
              Not Helpful
            </Button>
          </div>
        )}

        {feedbackStatus === 'submitting' && (
          <div className="flex items-center justify-center py-4">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <span className="ml-2 text-sm text-slate-600">Submitting feedback...</span>
          </div>
        )}

        {feedbackStatus === 'submitted' && feedbackMessage && (
          <Alert className={feedbackMessage.type === 'success' ? 'border-green-200 bg-green-50' : 'border-blue-200 bg-blue-50'}>
            <AlertDescription className="flex items-center gap-2">
              {submittedRating === 'up' ? (
                <ThumbsUp className="w-4 h-4 text-green-600" />
              ) : (
                <ThumbsDown className="w-4 h-4 text-blue-600" />
              )}
              {feedbackMessage.message}
            </AlertDescription>
          </Alert>
        )}

        {feedbackStatus === 'error' && (
          <Alert className="border-red-200 bg-red-50">
            <AlertDescription className="text-red-700">
              Failed to submit feedback: {error}
              <Button 
                variant="link" 
                size="sm" 
                className="ml-2 h-auto p-0 text-red-700"
                onClick={() => setFeedbackStatus('pending')}
              >
                Try again
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {renderRewardBreakdown()}
      </CardContent>
    </Card>
  );
};

export default FeedbackCard;