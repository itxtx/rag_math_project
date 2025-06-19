# src/learner_model/profile_manager.py - Improved version with batch operations
import sqlite3
import json
import datetime
import os
from typing import List, Dict, Optional, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

from src import config 

DEFAULT_DB_PATH = os.path.join(config.DATA_DIR, "learner_profiles.sqlite3")

DIFFICULTY_LEVELS = ["beginner", "intermediate", "high", "super_high"]
CORRECT_ANSWERS_TO_ADVANCE_DIFFICULTY = 3

logger = logging.getLogger(__name__)

class LearnerProfileManager:
    """
    Improved learner profile manager with better async support and batch operations
    - Thread-safe database operations
    - Batch processing for better performance
    - Connection pooling simulation with thread-local storage
    """

    def __init__(self, db_path: Optional[str] = None, max_workers: int = 4):
        self.db_path = db_path if db_path else DEFAULT_DB_PATH
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.local = threading.local()  # Thread-local storage for connections
        
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir): 
            os.makedirs(db_dir, exist_ok=True)
        
        # Initialize the database schema
        self._init_database()
        
        logger.info(f"ProfileManager initialized. Database at: {self.db_path}")

    def _get_connection(self):
        """Get a thread-local database connection"""
        if not hasattr(self.local, 'connection'):
            self.local.connection = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=30.0  # 30 second timeout
            )
            self.local.connection.row_factory = sqlite3.Row
            self.local.connection.execute("PRAGMA foreign_keys = ON;")
            self.local.connection.execute("PRAGMA journal_mode = WAL;")  # Better concurrency
            self.local.connection.execute("PRAGMA synchronous = NORMAL;")  # Better performance
        
        return self.local.connection

    def _init_database(self):
        """Initialize database schema"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Learners table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learners (
                    learner_id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    overall_progress REAL DEFAULT 0.0,
                    last_active_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Concept knowledge table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_knowledge (
                    knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    concept_id TEXT NOT NULL, 
                    doc_id TEXT,
                    current_score REAL DEFAULT 0.0,          
                    last_answered_correctly INTEGER DEFAULT 0, 
                    total_attempts INTEGER DEFAULT 0,
                    correct_attempts INTEGER DEFAULT 0,
                    last_attempted_at DATETIME,
                    srs_repetitions INTEGER DEFAULT 0,       
                    srs_interval_days INTEGER DEFAULT 0,     
                    next_review_at DATETIME,                 
                    current_difficulty_level TEXT DEFAULT 'beginner', 
                    consecutive_correct_at_difficulty INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (learner_id) REFERENCES learners (learner_id) ON DELETE CASCADE,
                    UNIQUE (learner_id, concept_id) 
                )
            """)
            
            # Score history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS score_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_id INTEGER NOT NULL, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    score REAL NOT NULL, 
                    raw_eval_data TEXT,
                    interaction_time_seconds REAL,
                    FOREIGN KEY (knowledge_id) REFERENCES concept_knowledge (knowledge_id) ON DELETE CASCADE
                )
            """)
            
            # Create indices for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_concept_knowledge_learner_concept ON concept_knowledge(learner_id, concept_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_concept_knowledge_next_review ON concept_knowledge(next_review_at);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_score_history_knowledge_id ON score_history(knowledge_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_learners_last_active ON learners(last_active_at);")
            
            conn.commit()
            logger.info("Database schema initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
            raise

    def _execute_query_sync(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a query synchronously with proper error handling"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = [dict(row) for row in cursor.fetchall()]
            conn.commit()
            return results
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Database query error: {e}")
            raise

    def _execute_update_sync(self, query: str, params: tuple = None) -> bool:
        """Execute an update query synchronously"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Database update error: {e}")
            raise

    # Async wrappers for backwards compatibility
    async def create_profile(self, learner_id: str) -> bool:
        """Create a learner profile asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._create_profile_sync, 
            learner_id
        )

    def _create_profile_sync(self, learner_id: str) -> bool:
        """Create a learner profile synchronously"""
        try:
            return self._execute_update_sync(
                "INSERT OR IGNORE INTO learners (learner_id, last_active_at) VALUES (?, ?)",
                (learner_id, datetime.datetime.now())
            )
        except sqlite3.Error as e:
            logger.error(f"Error creating profile for learner '{learner_id}': {e}")
            return False

    async def get_profile(self, learner_id: str) -> Optional[Dict[str, Any]]:
        """Get learner profile asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._get_profile_sync,
            learner_id
        )

    def _get_profile_sync(self, learner_id: str) -> Optional[Dict[str, Any]]:
        """Get learner profile synchronously"""
        try:
            results = self._execute_query_sync(
                "SELECT * FROM learners WHERE learner_id = ?",
                (learner_id,)
            )
            return results[0] if results else None
        except sqlite3.Error as e:
            logger.error(f"Error getting profile for learner '{learner_id}': {e}")
            return None

    # Synchronous methods (for backwards compatibility with non-async code)
    def get_concept_knowledge(self, learner_id: str, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get concept knowledge synchronously"""
        try:
            results = self._execute_query_sync(
                """
                SELECT * FROM concept_knowledge 
                WHERE learner_id = ? AND concept_id = ?
                """,
                (learner_id, concept_id)
            )
            return results[0] if results else None
        except sqlite3.Error as e:
            logger.error(f"Error getting concept knowledge: {e}")
            return None

    async def get_concept_knowledge_async(self, learner_id: str, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get concept knowledge asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_concept_knowledge,
            learner_id,
            concept_id
        )

    def get_concept_knowledge_batch(self, learner_id: str, concept_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get knowledge for multiple concepts in one query"""
        if not concept_ids:
            return {}
        
        try:
            # Create placeholders for the IN clause
            placeholders = ','.join(['?' for _ in concept_ids])
            query = f"""
                SELECT * FROM concept_knowledge 
                WHERE learner_id = ? AND concept_id IN ({placeholders})
            """
            params = [learner_id] + concept_ids
            
            results = self._execute_query_sync(query, tuple(params))
            
            # Convert to dictionary keyed by concept_id
            return {row['concept_id']: dict(row) for row in results}
            
        except sqlite3.Error as e:
            logger.error(f"Error getting batch concept knowledge: {e}")
            return {}

    async def get_concept_knowledge_batch_async(self, learner_id: str, concept_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get knowledge for multiple concepts asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_concept_knowledge_batch,
            learner_id,
            concept_ids
        )

    def update_concept_srs_and_difficulty(self, 
                                 learner_id: str, 
                                 concept_id: str, 
                                 doc_id: Optional[str],
                                 score: float, 
                                 answered_correctly: bool,
                                 srs_details: Dict[str, Any], 
                                 raw_eval_data: Optional[Dict] = None,
                                 interaction_time_seconds: Optional[float] = None
                                 ) -> bool:
        """Update concept knowledge with improved error handling"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            now_timestamp = datetime.datetime.now()
            raw_eval_json = json.dumps(raw_eval_data) if raw_eval_data else None

            current_knowledge = self.get_concept_knowledge(learner_id, concept_id)
            
            knowledge_id: Optional[int] = None
            new_total_attempts: int = 1
            new_correct_attempts: int = 1 if answered_correctly else 0
            current_difficulty = 'beginner'
            consecutive_correct_at_curr_difficulty = 0

            if current_knowledge: 
                knowledge_id = current_knowledge["knowledge_id"]
                new_total_attempts = current_knowledge.get("total_attempts", 0) + 1
                new_correct_attempts = current_knowledge.get("correct_attempts", 0) + (1 if answered_correctly else 0)
                current_difficulty = current_knowledge.get("current_difficulty_level", 'beginner')
                consecutive_correct_at_curr_difficulty = current_knowledge.get("consecutive_correct_at_difficulty", 0)
            
            # Update difficulty progression
            if answered_correctly:
                consecutive_correct_at_curr_difficulty += 1
                if consecutive_correct_at_curr_difficulty >= CORRECT_ANSWERS_TO_ADVANCE_DIFFICULTY:
                    try:
                        current_difficulty_idx = DIFFICULTY_LEVELS.index(current_difficulty)
                        if current_difficulty_idx < len(DIFFICULTY_LEVELS) - 1:
                            current_difficulty = DIFFICULTY_LEVELS[current_difficulty_idx + 1]
                            consecutive_correct_at_curr_difficulty = 0 
                            logger.debug(f"Difficulty advanced to: {current_difficulty} for concept {concept_id}")
                        else:
                            logger.debug(f"Concept {concept_id} already at max difficulty: {current_difficulty}")
                    except ValueError: 
                        logger.warning(f"Unknown difficulty '{current_difficulty}', resetting to beginner")
                        current_difficulty = 'beginner'
                        consecutive_correct_at_curr_difficulty = 1
            else: 
                consecutive_correct_at_curr_difficulty = 0 

            next_review_dt = srs_details.get("next_review_at")
            next_interval = srs_details.get("next_interval_days")
            new_srs_reps = srs_details.get("new_srs_repetitions")

            if current_knowledge:
                cursor.execute("""
                    UPDATE concept_knowledge 
                    SET current_score = ?, last_answered_correctly = ?, 
                        total_attempts = ?, correct_attempts = ?, last_attempted_at = ?,
                        srs_repetitions = ?, srs_interval_days = ?, next_review_at = ?,
                        current_difficulty_level = ?, consecutive_correct_at_difficulty = ?,
                        doc_id = ?, updated_at = ?
                    WHERE knowledge_id = ?
                """, (score, 1 if answered_correctly else 0, new_total_attempts, new_correct_attempts, now_timestamp,
                      new_srs_reps, next_interval, next_review_dt,
                      current_difficulty, consecutive_correct_at_curr_difficulty,
                      doc_id, now_timestamp, knowledge_id))
            else: 
                cursor.execute("""
                    INSERT INTO concept_knowledge 
                        (learner_id, concept_id, doc_id, current_score, last_answered_correctly, 
                         total_attempts, correct_attempts, last_attempted_at,
                         srs_repetitions, srs_interval_days, next_review_at,
                         current_difficulty_level, consecutive_correct_at_difficulty)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (learner_id, concept_id, doc_id, score, 1 if answered_correctly else 0, 
                      new_total_attempts, new_correct_attempts, now_timestamp,
                      new_srs_reps, next_interval, next_review_dt,
                      current_difficulty, consecutive_correct_at_curr_difficulty))
                knowledge_id = cursor.lastrowid

            if knowledge_id is None: 
                logger.error(f"Failed to get knowledge_id for learner '{learner_id}', concept '{concept_id}'")
                conn.rollback()
                return False

            # Add score history entry
            cursor.execute("""
                INSERT INTO score_history (knowledge_id, score, raw_eval_data, timestamp, interaction_time_seconds)
                VALUES (?, ?, ?, ?, ?)
            """, (knowledge_id, score, raw_eval_json, now_timestamp, interaction_time_seconds))
            
            # Update learner's last active time
            cursor.execute("""
                UPDATE learners SET last_active_at = ? WHERE learner_id = ?
            """, (now_timestamp, learner_id))
            
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error updating concept knowledge: {e}")
            conn.rollback()
            return False

    async def update_concept_srs_and_difficulty_async(self, *args, **kwargs) -> bool:
        """Async wrapper for update_concept_srs_and_difficulty"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.update_concept_srs_and_difficulty,
            *args,
            **kwargs
        )

    def get_concepts_for_review(self, learner_id: str, review_date: Optional[datetime.datetime] = None, target_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get concepts due for review"""
        try:
            if review_date is None:
                review_date = datetime.datetime.now()
            
            query = """
                SELECT concept_id, current_score, next_review_at, srs_interval_days, current_difficulty_level, doc_id
                FROM concept_knowledge
                WHERE learner_id = ? AND next_review_at IS NOT NULL AND next_review_at <= ?
            """
            params = [learner_id, review_date.strftime("%Y-%m-%d %H:%M:%S")]

            if target_doc_id:
                query += " AND doc_id = ?"
                params.append(target_doc_id)
            
            query += " ORDER BY next_review_at ASC"
            
            return self._execute_query_sync(query, tuple(params))
            
        except sqlite3.Error as e:
            logger.error(f"Error fetching concepts for review: {e}")
            return []

    async def get_concepts_for_review_async(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Async wrapper for get_concepts_for_review"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_concepts_for_review,
            *args,
            **kwargs
        )

    def get_last_attempted_concept_and_doc(self, learner_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Get the last attempted concept and document"""
        try:
            results = self._execute_query_sync("""
                SELECT concept_id, doc_id FROM concept_knowledge 
                WHERE learner_id = ? AND last_attempted_at IS NOT NULL
                ORDER BY last_attempted_at DESC LIMIT 1
            """, (learner_id,))
            
            if results:
                return results[0]['concept_id'], results[0]['doc_id']
            return None, None
            
        except sqlite3.Error as e:
            logger.error(f"Error getting last attempted concept: {e}")
            return None, None

    async def get_last_attempted_concept_and_doc_async(self, learner_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Async wrapper for get_last_attempted_concept_and_doc"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_last_attempted_concept_and_doc,
            learner_id
        )

    def get_learner_stats(self, learner_id: str) -> Dict[str, Any]:
        """Get comprehensive learner statistics"""
        try:
            # Get basic profile info
            profile = self._get_profile_sync(learner_id)
            if not profile:
                return {}
            
            # Get concept statistics
            concept_stats = self._execute_query_sync("""
                SELECT 
                    COUNT(*) as total_concepts,
                    AVG(current_score) as avg_score,
                    SUM(total_attempts) as total_attempts,
                    SUM(correct_attempts) as correct_attempts,
                    MAX(last_attempted_at) as last_activity
                FROM concept_knowledge 
                WHERE learner_id = ?
            """, (learner_id,))
            
            # Get recent performance
            recent_scores = self._execute_query_sync("""
                SELECT sh.score, sh.timestamp 
                FROM score_history sh
                JOIN concept_knowledge ck ON sh.knowledge_id = ck.knowledge_id
                WHERE ck.learner_id = ?
                ORDER BY sh.timestamp DESC
                LIMIT 10
            """, (learner_id,))
            
            stats = dict(profile)
            if concept_stats:
                stats.update(concept_stats[0])
            
            stats['recent_scores'] = [row['score'] for row in recent_scores]
            stats['accuracy_rate'] = (
                stats.get('correct_attempts', 0) / max(1, stats.get('total_attempts', 1))
            )
            
            return stats
            
        except sqlite3.Error as e:
            logger.error(f"Error getting learner stats: {e}")
            return {}

    def cleanup_old_data(self, days_old: int = 90):
        """Clean up old data to prevent database bloat"""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Clean up old score history
            cursor.execute("""
                DELETE FROM score_history 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} old score history entries")
            
            # Vacuum to reclaim space
            cursor.execute("VACUUM")
            
        except sqlite3.Error as e:
            logger.error(f"Error during cleanup: {e}")

    def close_db(self):
        """Close database connections"""
        try:
            if hasattr(self.local, 'connection'):
                self.local.connection.close()
                delattr(self.local, 'connection')
            
            self.executor.shutdown(wait=True)
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close_db()
        except:
            pass