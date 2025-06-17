import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Card, CardHeader, CardContent, CardFooter, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

const cleanLatexString = (text) => {
  if (typeof text !== 'string') {
    return text;
  }
  let cleanedText = text.replace(/`(\${1,2}[^`]*?\${1,2})`/g, '$1');
  cleanedText = cleanedText.replace(/\${3,}/g, '$$');
  return cleanedText;
};

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