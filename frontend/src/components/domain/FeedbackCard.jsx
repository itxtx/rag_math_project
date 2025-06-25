import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Card, CardHeader, CardContent, CardFooter, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { cleanLatexString } from '@/lib/utils';

const FeedbackCard = ({
  feedback,
  onNextQuestion,
  onReviewMaterial,
  onShowCorrectAnswer
}) => {
  if (!feedback) return null;

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Feedback</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="prose dark:prose-invert max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {cleanLatexString(feedback.feedbackText)}
          </ReactMarkdown>
        </div>
        {feedback.correctAnswer && (
          <div className="prose dark:prose-invert max-w-none">
            <h4>Correct Answer:</h4>
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {cleanLatexString(feedback.correctAnswer)}
            </ReactMarkdown>
          </div>
        )}
      </CardContent>
      <CardFooter className="flex flex-wrap gap-2 justify-center">
        <Button
          variant="outline"
          onClick={onReviewMaterial}
        >
          Review Material
        </Button>
        <Button
          variant="outline"
          onClick={onShowCorrectAnswer}
        >
          Show Correct Answer
        </Button>
        <Button
          onClick={onNextQuestion}
        >
          Next Question
        </Button>
      </CardFooter>
    </Card>
  );
};

export default FeedbackCard; 