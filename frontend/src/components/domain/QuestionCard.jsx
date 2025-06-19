import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Card, CardHeader, CardContent, CardFooter, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { cleanLatexString } from '@/lib/utils';

const QuestionCard = ({
  question,
  answer,
  setAnswer,
  onSubmitAnswer,
  loading,
  progress
}) => {
  if (!question) return null;

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Question</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="prose dark:prose-invert max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {cleanLatexString(question.question_text)}
          </ReactMarkdown>
        </div>
        <div className="space-y-2">
          <Textarea
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            placeholder="Type your answer here..."
            className="min-h-[100px]"
            disabled={loading}
          />
        </div>
        {progress !== undefined && (
          <Progress value={progress} className="w-full" />
        )}
      </CardContent>
      <CardFooter>
        <Button
          onClick={onSubmitAnswer}
          disabled={!answer.trim() || loading}
          className="w-full"
        >
          {loading ? 'Submitting...' : 'Submit Answer'}
        </Button>
      </CardFooter>
    </Card>
  );
};

export default QuestionCard; 