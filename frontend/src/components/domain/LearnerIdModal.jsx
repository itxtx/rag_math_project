import React from 'react';
import { Card, CardHeader, CardContent, CardFooter, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

const LearnerIdModal = ({
  learnerId,
  setLearnerId,
  availableTopics,
  selectedTopicId,
  setSelectedTopicId,
  topicsLoading,
  onStartInteraction,
  isOpen
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <Card className="w-[90%] max-w-md">
        <CardHeader>
          <CardTitle>Welcome to Math Learning Assistant</CardTitle>
          <CardDescription>
            Please enter your learner ID and select a topic to begin
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="learnerId">Learner ID</Label>
            <Input
              id="learnerId"
              value={learnerId}
              onChange={(e) => setLearnerId(e.target.value)}
              placeholder="Enter your learner ID"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="topic">Select Topic</Label>
            <Select
              value={selectedTopicId}
              onValueChange={setSelectedTopicId}
              disabled={topicsLoading}
            >
              <SelectTrigger id="topic">
                <SelectValue placeholder="Select a topic" />
              </SelectTrigger>
              <SelectContent>
                {availableTopics.map((topic) => (
                  <SelectItem key={topic.doc_id} value={topic.doc_id}>
                    {topic.title || topic.doc_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
        <CardFooter>
          <Button
            className="w-full"
            onClick={onStartInteraction}
            disabled={!learnerId.trim() || !selectedTopicId || topicsLoading}
          >
            Start Learning
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

export default LearnerIdModal; 