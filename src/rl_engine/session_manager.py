# src/rl_engine/session_manager.py
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class SessionData:
    """Data structure for tracking learner session state"""
    learner_id: str
    session_start: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    questions_answered: int = 0
    engagement_scores: List[float] = field(default_factory=list)
    interaction_times: List[float] = field(default_factory=list)  # Time taken per question
    current_streak: int = 0  # Consecutive correct answers
    total_upvotes: int = 0
    total_downvotes: int = 0
    session_active: bool = True
    
    def add_engagement_score(self, score: float):
        """Add an engagement score (based on vote feedback)"""
        self.engagement_scores.append(score)
        # Keep only last 10 scores for recent engagement
        if len(self.engagement_scores) > 10:
            self.engagement_scores.pop(0)
    
    def add_interaction_time(self, time_seconds: float):
        """Add time taken for an interaction"""
        self.interaction_times.append(time_seconds)
        # Keep only last 5 interaction times
        if len(self.interaction_times) > 5:
            self.interaction_times.pop(0)
    
    def get_average_engagement(self) -> float:
        """Get average engagement score"""
        if not self.engagement_scores:
            return 0.5  # Neutral engagement
        return sum(self.engagement_scores) / len(self.engagement_scores)
    
    def get_average_interaction_time(self) -> float:
        """Get average interaction time in seconds"""
        if not self.interaction_times:
            return 30.0  # Default 30 seconds
        return sum(self.interaction_times) / len(self.interaction_times)
    
    def get_session_length_minutes(self) -> float:
        """Get session length in minutes"""
        return (self.last_activity - self.session_start).total_seconds() / 60.0
    
    def get_time_since_last_question_minutes(self) -> float:
        """Get time since last question in minutes"""
        return (datetime.now() - self.last_activity).total_seconds() / 60.0
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_session_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired"""
        return self.get_time_since_last_question_minutes() > timeout_minutes

class SessionManager:
    """
    Manages learner sessions for the RL environment
    Tracks engagement, timing, and session state
    """
    
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, SessionData] = {}
        self.session_timeout_minutes = session_timeout_minutes
        self.interaction_start_times: Dict[str, datetime] = {}  # interaction_id -> start_time
        
        logger.info(f"SessionManager initialized with {session_timeout_minutes}min timeout")
    
    def get_or_create_session(self, learner_id: str) -> SessionData:
        """Get existing session or create new one for learner"""
        if learner_id not in self.sessions:
            self.sessions[learner_id] = SessionData(learner_id=learner_id)
            logger.info(f"Created new session for learner {learner_id}")
        
        session = self.sessions[learner_id]
        
        # Check if session has expired
        if session.is_session_expired(self.session_timeout_minutes):
            # Create new session
            logger.info(f"Session expired for learner {learner_id}, creating new session")
            self.sessions[learner_id] = SessionData(learner_id=learner_id)
            session = self.sessions[learner_id]
        
        return session
    
    def start_interaction(self, learner_id: str, interaction_id: str):
        """Record start of an interaction"""
        session = self.get_or_create_session(learner_id)
        self.interaction_start_times[interaction_id] = datetime.now()
        session.update_activity()
        
        logger.debug(f"Started interaction {interaction_id} for learner {learner_id}")
    
    def complete_interaction(self, 
                           learner_id: str, 
                           interaction_id: str,
                           was_correct: bool,
                           vote_type: str = None):
        """Record completion of an interaction"""
        session = self.get_or_create_session(learner_id)
        
        # Calculate interaction time
        if interaction_id in self.interaction_start_times:
            start_time = self.interaction_start_times[interaction_id]
            interaction_time = (datetime.now() - start_time).total_seconds()
            session.add_interaction_time(interaction_time)
            del self.interaction_start_times[interaction_id]
        
        # Update question count
        session.questions_answered += 1
        
        # Update streak
        if was_correct:
            session.current_streak += 1
        else:
            session.current_streak = 0
        
        # Process vote feedback
        if vote_type == "up":
            session.total_upvotes += 1
            session.add_engagement_score(1.0)
        elif vote_type == "down":
            session.total_downvotes += 1
            session.add_engagement_score(0.0)
        else:
            # Timeout or no feedback - neutral engagement
            session.add_engagement_score(0.3)
        
        session.update_activity()
        
        logger.debug(f"Completed interaction {interaction_id} for learner {learner_id}: "
                    f"correct={was_correct}, vote={vote_type}")
    
    def get_session_state_for_rl(self, learner_id: str) -> Dict[str, float]:
        """Get session state features for RL environment"""
        session = self.get_or_create_session(learner_id)
        
        # Normalize values for RL
        normalized_session_length = min(session.questions_answered / 20.0, 1.0)  # Cap at 20 questions
        average_engagement = session.get_average_engagement()
        time_since_last = min(session.get_time_since_last_question_minutes() / 30.0, 1.0)  # Cap at 30 minutes
        
        # Calculate fatigue factor (performance decreases with session length)
        fatigue_factor = max(0.3, 1.0 - (session.questions_answered * 0.02))  # Gradual decrease
        
        # Calculate momentum (recent performance trend)
        recent_engagement = session.engagement_scores[-3:] if len(session.engagement_scores) >= 3 else session.engagement_scores
        momentum = 0.5  # Default neutral
        if len(recent_engagement) >= 2:
            recent_trend = sum(recent_engagement[-2:]) / 2.0 - sum(recent_engagement[-3:-1]) / 1.0 if len(recent_engagement) >= 3 else 0
            momentum = 0.5 + (recent_trend * 0.5)  # Scale to 0-1
            momentum = max(0.0, min(1.0, momentum))
        
        return {
            'session_length_normalized': normalized_session_length,
            'average_engagement': average_engagement,
            'time_since_last_normalized': time_since_last,
            'fatigue_factor': fatigue_factor,
            'momentum': momentum,
            'current_streak_normalized': min(session.current_streak / 5.0, 1.0),  # Cap at 5
            'questions_answered': float(session.questions_answered),
            'upvote_rate': session.total_upvotes / max(1, session.total_upvotes + session.total_downvotes)
        }
    
    def get_session_stats(self, learner_id: str) -> Dict[str, Any]:
        """Get detailed session statistics"""
        if learner_id not in self.sessions:
            return {}
        
        session = self.sessions[learner_id]
        
        return {
            'learner_id': learner_id,
            'session_start': session.session_start.isoformat(),
            'session_duration_minutes': session.get_session_length_minutes(),
            'questions_answered': session.questions_answered,
            'current_streak': session.current_streak,
            'total_upvotes': session.total_upvotes,
            'total_downvotes': session.total_downvotes,
            'upvote_rate': session.total_upvotes / max(1, session.total_upvotes + session.total_downvotes),
            'average_engagement': session.get_average_engagement(),
            'average_interaction_time_seconds': session.get_average_interaction_time(),
            'time_since_last_question_minutes': session.get_time_since_last_question_minutes(),
            'session_active': session.session_active and not session.is_session_expired(self.session_timeout_minutes)
        }
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions to prevent memory leaks"""
        expired_learners = []
        
        for learner_id, session in self.sessions.items():
            if session.is_session_expired(self.session_timeout_minutes * 2):  # Double timeout for cleanup
                expired_learners.append(learner_id)
        
        for learner_id in expired_learners:
            del self.sessions[learner_id]
            logger.info(f"Cleaned up expired session for learner {learner_id}")
        
        # Also cleanup old interaction start times
        current_time = datetime.now()
        expired_interactions = []
        
        for interaction_id, start_time in self.interaction_start_times.items():
            if (current_time - start_time).total_seconds() > 3600:  # 1 hour timeout
                expired_interactions.append(interaction_id)
        
        for interaction_id in expired_interactions:
            del self.interaction_start_times[interaction_id]
        
        if expired_learners or expired_interactions:
            logger.info(f"Cleaned up {len(expired_learners)} expired sessions and "
                       f"{len(expired_interactions)} expired interactions")
    
    def get_all_active_sessions(self) -> List[Dict[str, Any]]:
        """Get statistics for all active sessions"""
        active_sessions = []
        
        for learner_id, session in self.sessions.items():
            if not session.is_session_expired(self.session_timeout_minutes):
                active_sessions.append(self.get_session_stats(learner_id))
        
        return active_sessions
    
    def reset_session(self, learner_id: str):
        """Reset/clear session for a learner"""
        if learner_id in self.sessions:
            del self.sessions[learner_id]
            logger.info(f"Reset session for learner {learner_id}")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global session management statistics"""
        active_sessions = self.get_all_active_sessions()
        
        if not active_sessions:
            return {
                'total_sessions': 0,
                'active_sessions': 0,
                'average_session_length': 0.0,
                'total_questions_answered': 0,
                'global_upvote_rate': 0.0
            }
        
        total_questions = sum(s['questions_answered'] for s in active_sessions)
        total_upvotes = sum(s['total_upvotes'] for s in active_sessions)
        total_votes = sum(s['total_upvotes'] + s['total_downvotes'] for s in active_sessions)
        avg_session_length = sum(s['session_duration_minutes'] for s in active_sessions) / len(active_sessions)
        
        return {
            'total_sessions': len(self.sessions),
            'active_sessions': len(active_sessions),
            'average_session_length_minutes': avg_session_length,
            'total_questions_answered': total_questions,
            'global_upvote_rate': total_upvotes / max(1, total_votes),
            'cleanup_needed': len(self.sessions) > 100  # Suggest cleanup if too many sessions
        }