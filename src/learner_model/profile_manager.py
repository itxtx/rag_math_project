# src/learner_model/profile_manager.py
import sqlite3
import json
import datetime
import os
from typing import List, Dict, Optional, Any, Tuple

from src import config 

DEFAULT_DB_PATH = os.path.join(config.DATA_DIR, "learner_profiles.sqlite3")

DIFFICULTY_LEVELS = ["beginner", "intermediate", "high", "super_high"]
CORRECT_ANSWERS_TO_ADVANCE_DIFFICULTY = 3

class LearnerProfileManager:
    """
    Manages learner profiles, storing data in an SQLite database.
    - Learner ID, overall progress.
    - Concept-specific knowledge states (score, history, SRS data, difficulty).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path if db_path else DEFAULT_DB_PATH
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir): 
            os.makedirs(db_dir, exist_ok=True)
        self._connect_db()
        self._create_tables() 
        print(f"ProfileManager initialized. Database at: {self.db_path}")

    def _connect_db(self):
        try:
            # Enable check_same_thread=False to allow connections from different threads
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row 
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.cursor = self.conn.cursor()
            print("ProfileManager: Database connection established and foreign keys enabled.")
        except sqlite3.Error as e:
            print(f"ProfileManager: Error connecting to database '{self.db_path}': {e}")
            raise

    def _get_connection(self):
        """Get a new connection for the current thread"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            return conn
        except sqlite3.Error as e:
            print(f"ProfileManager: Error creating new connection: {e}")
            raise

    def _execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a query with a new connection"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = [dict(row) for row in cursor.fetchall()]
            conn.commit()
            return results
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            print(f"ProfileManager: Error executing query: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _execute_update(self, query: str, params: tuple = None) -> bool:
        """Execute an update query with a new connection"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            print(f"ProfileManager: Error executing update: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _create_tables(self):
        if not self.cursor: return

        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS learners (
                    learner_id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    overall_progress REAL DEFAULT 0.0 
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_knowledge (
                    knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    concept_id TEXT NOT NULL, 
                    doc_id TEXT, -- Added to link concept to a document/topic
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
                    FOREIGN KEY (learner_id) REFERENCES learners (learner_id) ON DELETE CASCADE,
                    UNIQUE (learner_id, concept_id) 
                )
            """)
            self._add_missing_concept_knowledge_columns() 
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS score_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_id INTEGER NOT NULL, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    score REAL NOT NULL, 
                    raw_eval_data TEXT, 
                    FOREIGN KEY (knowledge_id) REFERENCES concept_knowledge (knowledge_id) ON DELETE CASCADE
                )
            """)
            self.conn.commit()
            print("ProfileManager: Tables ensured/created/updated successfully.")
        except sqlite3.Error as e:
            print(f"ProfileManager: Error creating/updating tables: {e}")
            if self.conn: self.conn.rollback()
            raise

    def _add_missing_concept_knowledge_columns(self):
        if not self.cursor: return
        
        columns_to_add = {
            "srs_repetitions": "INTEGER DEFAULT 0",
            "srs_interval_days": "INTEGER DEFAULT 0",
            "next_review_at": "DATETIME",
            "current_difficulty_level": "TEXT DEFAULT 'beginner'",
            "consecutive_correct_at_difficulty": "INTEGER DEFAULT 0",
            "doc_id": "TEXT" # Added doc_id column
        }
        
        self.cursor.execute("PRAGMA table_info(concept_knowledge);")
        existing_columns = [row['name'] for row in self.cursor.fetchall()]
        
        for col_name, col_def in columns_to_add.items():
            if col_name not in existing_columns:
                try:
                    self.cursor.execute(f"ALTER TABLE concept_knowledge ADD COLUMN {col_name} {col_def};")
                    print(f"ProfileManager: Added column '{col_name}' to 'concept_knowledge' table.")
                except sqlite3.Error as e:
                    print(f"ProfileManager: Warning - Could not add column '{col_name}': {e}")
        self.conn.commit()

    def create_profile(self, learner_id: str) -> bool:
        try:
            return self._execute_update(
                "INSERT OR IGNORE INTO learners (learner_id) VALUES (?)",
                (learner_id,)
            )
        except sqlite3.Error as e:
            print(f"ProfileManager: Error creating profile for learner '{learner_id}': {e}")
            return False

    def get_profile(self, learner_id: str) -> Optional[Dict[str, Any]]:
        try:
            results = self._execute_query(
                "SELECT * FROM learners WHERE learner_id = ?",
                (learner_id,)
            )
            return results[0] if results else None
        except sqlite3.Error as e:
            print(f"ProfileManager: Error getting profile for learner '{learner_id}': {e}")
            return None

    def update_overall_progress(self, learner_id: str, progress: float) -> bool:
        try:
            return self._execute_update(
                "UPDATE learners SET overall_progress = ? WHERE learner_id = ?",
                (progress, learner_id)
            )
        except sqlite3.Error as e:
            print(f"ProfileManager: Error updating overall progress for learner '{learner_id}': {e}")
            return False

    def get_concept_knowledge(self, learner_id: str, concept_id: str) -> Optional[Dict[str, Any]]:
        try:
            results = self._execute_query(
                """
                SELECT * FROM concept_knowledge 
                WHERE learner_id = ? AND concept_id = ?
                """,
                (learner_id, concept_id)
            )
            return results[0] if results else None
        except sqlite3.Error as e:
            print(f"ProfileManager: Error getting concept knowledge for learner '{learner_id}', concept '{concept_id}': {e}")
            return None

    def update_concept_srs_and_difficulty(self, 
                                 learner_id: str, 
                                 concept_id: str, 
                                 doc_id: Optional[str],
                                 score: float, 
                                 answered_correctly: bool,
                                 srs_details: Dict[str, Any], 
                                 raw_eval_data: Optional[Dict] = None
                                 ) -> bool:
        conn = None
        try:
            conn = self._get_connection()
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
            
            if answered_correctly:
                consecutive_correct_at_curr_difficulty += 1
                if consecutive_correct_at_curr_difficulty >= CORRECT_ANSWERS_TO_ADVANCE_DIFFICULTY:
                    try:
                        current_difficulty_idx = DIFFICULTY_LEVELS.index(current_difficulty)
                        if current_difficulty_idx < len(DIFFICULTY_LEVELS) - 1:
                            current_difficulty = DIFFICULTY_LEVELS[current_difficulty_idx + 1]
                            consecutive_correct_at_curr_difficulty = 0 
                            print(f"  Difficulty for concept '{concept_id}' advanced to: {current_difficulty}")
                        else:
                            print(f"  Concept '{concept_id}' already at max difficulty: {current_difficulty}")
                    except ValueError: 
                        print(f"  Warning: Unknown current difficulty '{current_difficulty}', resetting to beginner.")
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
                        doc_id = ? 
                    WHERE knowledge_id = ?
                """, (score, 1 if answered_correctly else 0, new_total_attempts, new_correct_attempts, now_timestamp,
                      new_srs_reps, next_interval, next_review_dt,
                      current_difficulty, consecutive_correct_at_curr_difficulty,
                      doc_id,
                      knowledge_id))
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
                print(f"ProfileManager: Failed to get knowledge_id for learner '{learner_id}', concept '{concept_id}'.")
                conn.rollback()
                return False

            cursor.execute("""
                INSERT INTO score_history (knowledge_id, score, raw_eval_data, timestamp)
                VALUES (?, ?, ?, ?)
            """, (knowledge_id, score, raw_eval_json, now_timestamp))
            
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"ProfileManager: Error updating concept knowledge/SRS for learner '{learner_id}', concept '{concept_id}': {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def get_score_history(self, learner_id: str, concept_id: str) -> List[Dict[str, Any]]:
        try:
            # First get the knowledge_id
            knowledge_results = self._execute_query(
                "SELECT knowledge_id FROM concept_knowledge WHERE learner_id = ? AND concept_id = ?",
                (learner_id, concept_id)
            )
            
            if not knowledge_results:
                return []
                
            knowledge_id = knowledge_results[0]["knowledge_id"]
            
            # Then get the history
            history_results = self._execute_query(
                "SELECT timestamp, score, raw_eval_data FROM score_history WHERE knowledge_id = ? ORDER BY timestamp ASC",
                (knowledge_id,)
            )
            
            # Process the results
            history = []
            for row in history_results:
                entry = dict(row)
                if entry.get('raw_eval_data'):
                    try:
                        entry['raw_eval_data'] = json.loads(entry['raw_eval_data'])
                    except json.JSONDecodeError:
                        pass
                history.append(entry)
                
            return history
        except sqlite3.Error as e:
            print(f"ProfileManager: Error retrieving score history: {e}")
            return []

    def get_concepts_for_review(self, learner_id: str, review_date: Optional[datetime.datetime] = None, target_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
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
            
            return self._execute_query(query, tuple(params))
        except sqlite3.Error as e:
            print(f"ProfileManager: Error fetching concepts for review for learner '{learner_id}': {e}")
            return []

    def close_db(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            print("ProfileManager: Database connection closed.")