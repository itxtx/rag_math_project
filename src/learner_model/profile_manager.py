# src/learner_model/profile_manager.py
import sqlite3
import json
import datetime
import os
from typing import List, Dict, Optional, Any, Tuple

from src import config 

DEFAULT_DB_PATH = os.path.join(config.DATA_DIR, "learner_profiles.sqlite3")

# Define difficulty levels (ordered)
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
        self.conn = None
        self.cursor = None
        self._connect_db()
        self._create_tables() # Will update table if needed
        print(f"ProfileManager initialized. Database at: {self.db_path}")

    def _connect_db(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row 
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.cursor = self.conn.cursor()
            print("ProfileManager: Database connection established and foreign keys enabled.")
        except sqlite3.Error as e:
            print(f"ProfileManager: Error connecting to database '{self.db_path}': {e}")
            raise 

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

            # --- UPDATED concept_knowledge table ---
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_knowledge (
                    knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    concept_id TEXT NOT NULL, 
                    current_score REAL DEFAULT 0.0,          -- Graded score 0-10
                    last_answered_correctly INTEGER DEFAULT 0, 
                    total_attempts INTEGER DEFAULT 0,
                    correct_attempts INTEGER DEFAULT 0,
                    last_attempted_at DATETIME,
                    
                    -- SRS Fields
                    srs_repetitions INTEGER DEFAULT 0,       -- Consecutive correct reviews
                    srs_interval_days INTEGER DEFAULT 0,     -- Current interval in days
                    next_review_at DATETIME,                 -- When this concept is next due for review
                    
                    -- Difficulty Tracking
                    current_difficulty_level TEXT DEFAULT 'beginner', -- Current difficulty for this learner on this concept
                    consecutive_correct_at_difficulty INTEGER DEFAULT 0, -- Count for advancing difficulty
                    
                    FOREIGN KEY (learner_id) REFERENCES learners (learner_id) ON DELETE CASCADE,
                    UNIQUE (learner_id, concept_id) 
                )
            """)
            self._add_missing_concept_knowledge_columns() # Ensure new columns are added if table exists
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS score_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_id INTEGER NOT NULL, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    score REAL NOT NULL, -- Graded score 0-10
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
        """Adds new columns to concept_knowledge if they don't exist."""
        if not self.cursor: return
        
        columns_to_add = {
            "srs_repetitions": "INTEGER DEFAULT 0",
            "srs_interval_days": "INTEGER DEFAULT 0",
            "next_review_at": "DATETIME",
            "current_difficulty_level": "TEXT DEFAULT 'beginner'",
            "consecutive_correct_at_difficulty": "INTEGER DEFAULT 0"
        }
        
        self.cursor.execute("PRAGMA table_info(concept_knowledge);")
        existing_columns = [row['name'] for row in self.cursor.fetchall()]
        
        for col_name, col_def in columns_to_add.items():
            if col_name not in existing_columns:
                try:
                    self.cursor.execute(f"ALTER TABLE concept_knowledge ADD COLUMN {col_name} {col_def};")
                    print(f"ProfileManager: Added column '{col_name}' to 'concept_knowledge' table.")
                except sqlite3.Error as e:
                    print(f"ProfileManager: Warning - Could not add column '{col_name}': {e} (might exist from partial creation)")
        self.conn.commit()


    def close_db(self):
        if self.conn:
            self.conn.close(); self.conn = None; self.cursor = None
            print("ProfileManager: Database connection closed.")

    def create_profile(self, learner_id: str) -> bool:
        if not self.cursor or not self.conn: return False
        try:
            self.cursor.execute("INSERT OR IGNORE INTO learners (learner_id) VALUES (?)", (learner_id,))
            self.conn.commit()
            return self.cursor.rowcount >= 0 
        except sqlite3.Error as e:
            print(f"ProfileManager: Error creating profile for learner '{learner_id}': {e}")
            if self.conn: self.conn.rollback()
            return False

    def get_profile(self, learner_id: str) -> Optional[Dict[str, Any]]:
        if not self.cursor: return None
        try:
            self.cursor.execute("SELECT * FROM learners WHERE learner_id = ?", (learner_id,))
            row = self.cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            print(f"ProfileManager: Error getting profile for learner '{learner_id}': {e}")
            return None

    def update_overall_progress(self, learner_id: str, progress: float) -> bool:
        """
        Updates the overall progress for a learner.
        """
        if not self.cursor or not self.conn: return False
        try:
            self.cursor.execute("UPDATE learners SET overall_progress = ? WHERE learner_id = ?", (progress, learner_id))
            self.conn.commit()
            return self.cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"ProfileManager: Error updating overall progress for learner '{learner_id}': {e}")
            if self.conn: self.conn.rollback()
            return False


    def get_concept_knowledge(self, learner_id: str, concept_id: str) -> Optional[Dict[str, Any]]:
        if not self.cursor: return None
        try:
            self.cursor.execute("""
                SELECT * FROM concept_knowledge 
                WHERE learner_id = ? AND concept_id = ?
            """, (learner_id, concept_id))
            row = self.cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            print(f"ProfileManager: Error getting concept knowledge for learner '{learner_id}', concept '{concept_id}': {e}")
            return None

    def update_concept_srs_and_difficulty(self, 
                                 learner_id: str, 
                                 concept_id: str, 
                                 score: float, # Graded score 0-10
                                 answered_correctly: bool,
                                 srs_details: Dict[str, Any], # From SRSScheduler
                                 raw_eval_data: Optional[Dict] = None
                                 ) -> bool:
        """
        Updates concept knowledge including score, SRS data, and difficulty progression.
        """
        if not self.cursor or not self.conn: return False
        
        now_timestamp = datetime.datetime.now()
        raw_eval_json = json.dumps(raw_eval_data) if raw_eval_data else None

        try:
            current_knowledge = self.get_concept_knowledge(learner_id, concept_id)
            
            knowledge_id: Optional[int] = None
            new_total_attempts: int = 1
            new_correct_attempts: int = 1 if answered_correctly else 0
            current_difficulty = 'beginner'
            consecutive_correct_at_curr_difficulty = 0

            if current_knowledge: # Update existing entry
                knowledge_id = current_knowledge["knowledge_id"]
                new_total_attempts = current_knowledge.get("total_attempts", 0) + 1
                new_correct_attempts = current_knowledge.get("correct_attempts", 0) + (1 if answered_correctly else 0)
                current_difficulty = current_knowledge.get("current_difficulty_level", 'beginner')
                consecutive_correct_at_curr_difficulty = current_knowledge.get("consecutive_correct_at_difficulty", 0)
            
            # Difficulty Progression Logic
            if answered_correctly:
                consecutive_correct_at_curr_difficulty += 1
                if consecutive_correct_at_curr_difficulty >= CORRECT_ANSWERS_TO_ADVANCE_DIFFICULTY:
                    try:
                        current_difficulty_idx = DIFFICULTY_LEVELS.index(current_difficulty)
                        if current_difficulty_idx < len(DIFFICULTY_LEVELS) - 1:
                            current_difficulty = DIFFICULTY_LEVELS[current_difficulty_idx + 1]
                            consecutive_correct_at_curr_difficulty = 0 # Reset for new level
                            print(f"  Difficulty for concept '{concept_id}' advanced to: {current_difficulty}")
                        else:
                            print(f"  Concept '{concept_id}' already at max difficulty: {current_difficulty}")
                    except ValueError: # Should not happen if current_difficulty is always valid
                        print(f"  Warning: Unknown current difficulty '{current_difficulty}', resetting to beginner.")
                        current_difficulty = 'beginner'
                        consecutive_correct_at_curr_difficulty = 1
            else: # Answered incorrectly
                consecutive_correct_at_curr_difficulty = 0 # Reset counter
                # Optional: Demote difficulty on incorrect answer? For now, no.

            # Prepare SRS data from scheduler
            next_review_dt = srs_details.get("next_review_at")
            next_interval = srs_details.get("next_interval_days")
            new_srs_reps = srs_details.get("new_srs_repetitions")

            if current_knowledge:
                self.cursor.execute("""
                    UPDATE concept_knowledge 
                    SET current_score = ?, last_answered_correctly = ?, 
                        total_attempts = ?, correct_attempts = ?, last_attempted_at = ?,
                        srs_repetitions = ?, srs_interval_days = ?, next_review_at = ?,
                        current_difficulty_level = ?, consecutive_correct_at_difficulty = ?
                    WHERE knowledge_id = ?
                """, (score, 1 if answered_correctly else 0, new_total_attempts, new_correct_attempts, now_timestamp,
                      new_srs_reps, next_interval, next_review_dt,
                      current_difficulty, consecutive_correct_at_curr_difficulty,
                      knowledge_id))
            else: # Insert new entry
                self.cursor.execute("""
                    INSERT INTO concept_knowledge 
                        (learner_id, concept_id, current_score, last_answered_correctly, 
                         total_attempts, correct_attempts, last_attempted_at,
                         srs_repetitions, srs_interval_days, next_review_at,
                         current_difficulty_level, consecutive_correct_at_difficulty)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (learner_id, concept_id, score, 1 if answered_correctly else 0, 
                      new_total_attempts, new_correct_attempts, now_timestamp,
                      new_srs_reps, next_interval, next_review_dt,
                      current_difficulty, consecutive_correct_at_curr_difficulty))
                knowledge_id = self.cursor.lastrowid 

            if knowledge_id is None: 
                print(f"ProfileManager: Failed to get knowledge_id for learner '{learner_id}', concept '{concept_id}'.")
                if self.conn: self.conn.rollback()
                return False

            self.cursor.execute("""
                INSERT INTO score_history (knowledge_id, score, raw_eval_data, timestamp)
                VALUES (?, ?, ?, ?)
            """, (knowledge_id, score, raw_eval_json, now_timestamp))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"ProfileManager: Error updating concept knowledge/SRS for learner '{learner_id}', concept '{concept_id}': {e}")
            if self.conn: self.conn.rollback()
            return False

    def get_score_history(self, learner_id: str, concept_id: str) -> List[Dict[str, Any]]:
        # ... (remains the same as profile_manager_v1) ...
        if not self.cursor: return []
        history = []
        try:
            self.cursor.execute("SELECT knowledge_id FROM concept_knowledge WHERE learner_id = ? AND concept_id = ?", (learner_id, concept_id))
            knowledge_row = self.cursor.fetchone()
            if knowledge_row:
                knowledge_id = knowledge_row["knowledge_id"]
                self.cursor.execute("SELECT timestamp, score, raw_eval_data FROM score_history WHERE knowledge_id = ? ORDER BY timestamp ASC", (knowledge_id,))
                for row in self.cursor.fetchall():
                    entry = dict(row)
                    if entry.get('raw_eval_data'):
                        try: entry['raw_eval_data'] = json.loads(entry['raw_eval_data'])
                        except json.JSONDecodeError: pass 
                    history.append(entry)
        except sqlite3.Error as e:
            print(f"ProfileManager: Error retrieving score history: {e}")
        return history
    
    def get_concepts_for_review(self, learner_id: str, review_date: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """
        Retrieves concepts due for review for a given learner.
        """
        if not self.cursor: return []
        if review_date is None:
            review_date = datetime.datetime.now()
        
        try:
            self.cursor.execute("""
                SELECT concept_id, concept_name, current_score, next_review_at, srs_interval_days, current_difficulty_level
                FROM concept_knowledge ck
                JOIN learners l ON ck.learner_id = l.learner_id 
                WHERE ck.learner_id = ? AND ck.next_review_at <= ?
                ORDER BY ck.next_review_at ASC 
            """, (learner_id, review_date)) # Fetches concept_name from chunk metadata if available
            # Note: concept_name in concept_knowledge might not be directly stored.
            # This query assumes concept_id is sufficient or concept_name is populated.
            # For now, we'll rely on concept_id and QuestionSelector can fetch name from curriculum_map.
            # Simplified:
            self.cursor.execute("""
                SELECT concept_id, current_score, next_review_at, srs_interval_days, current_difficulty_level
                FROM concept_knowledge
                WHERE learner_id = ? AND next_review_at IS NOT NULL AND next_review_at <= ?
                ORDER BY next_review_at ASC
            """, (learner_id, review_date.strftime("%Y-%m-%d %H:%M:%S"))) # Format datetime for SQLite comparison
            
            return [dict(row) for row in self.cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"ProfileManager: Error fetching concepts for review for learner '{learner_id}': {e}")
            return []

