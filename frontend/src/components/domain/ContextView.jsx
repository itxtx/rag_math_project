import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Card, CardHeader, CardContent, CardFooter, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { cleanLatexString } from '@/lib/utils';

const ContextView = ({
  conceptContext,
  onProceedToQuestion
}) => {
  if (!conceptContext) return null;

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Review Material: {conceptContext.topic_name}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="prose dark:prose-invert max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {cleanLatexString(conceptContext.content_markdown)}
          </ReactMarkdown>
        </div>
      </CardContent>
      <CardFooter>
        <Button
          onClick={onProceedToQuestion}
          className="w-full"
        >
          Proceed to Question
        </Button>
      </CardFooter>
    </Card>
  );
};

export default ContextView; 