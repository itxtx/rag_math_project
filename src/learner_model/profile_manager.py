# src/learner_model/profile_manager.py
import sqlite3
import json
import datetime
import os
from typing import List, Dict, Optional, Any, Tuple

from src import config # For DB path

# Default DB path can be set in config.py or here
DEFAULT_DB_PATH = os.path.join(config.DATA_DIR, "learner_profiles.sqlite3")

class LearnerProfileManager:
    """
    Manages learner profiles, storing data in an SQLite database.
    - Learner ID, overall progress.
    - Concept-specific knowledge states (score, history).
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initializes the ProfileManager and connects to the SQLite database.
        Creates necessary tables if they don't exist.

        Args:
            db_path (Optional[str]): Path to the SQLite database file.
                                     Defaults to DEFAULT_DB_PATH.
        """
        self.db_path = db_path if db_path else DEFAULT_DB_PATH
        
        # Ensure the directory for the database exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir): # Check if db_dir is not empty string
            os.makedirs(db_dir, exist_ok=True)
            print(f"ProfileManager: Created directory for database: {db_dir}")

        self.conn = None
        self.cursor = None
        self._connect_db()
        self._create_tables()
        print(f"ProfileManager initialized. Database at: {self.db_path}")

    def _connect_db(self):
        """Establishes a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row # Access columns by name
            self.cursor = self.conn.cursor()
            print("ProfileManager: Database connection established.")
        except sqlite3.Error as e:
            print(f"ProfileManager: Error connecting to database '{self.db_path}': {e}")
            raise # Re-raise the exception if connection fails

    def _create_tables(self):
        """Creates the necessary tables in the database if they don't already exist."""
        if not self.cursor:
            print("ProfileManager: Database cursor not available. Cannot create tables.")
            return

        try:
            # learners table: Stores basic learner information
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS learners (
                    learner_id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    overall_progress REAL DEFAULT 0.0 
                )
            """)

            # concept_knowledge table: Stores knowledge about specific concepts for each learner
            # concept_id could be a hash of the concept name/text or a UUID from Weaviate chunk_id
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_knowledge (
                    knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    concept_id TEXT NOT NULL, 
                    current_score REAL DEFAULT 0.0,
                    last_answered_correctly INTEGER DEFAULT 0, -- 0 for false, 1 for true
                    total_attempts INTEGER DEFAULT 0,
                    correct_attempts INTEGER DEFAULT 0,
                    last_attempted_at DATETIME,
                    FOREIGN KEY (learner_id) REFERENCES learners (learner_id) ON DELETE CASCADE,
                    UNIQUE (learner_id, concept_id) 
                )
            """)
            
            # score_history table: Logs each score update for a concept
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS score_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_id INTEGER NOT NULL, -- Links to concept_knowledge primary key
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    score REAL NOT NULL,
                    raw_eval_data TEXT, -- Store raw evaluation details (e.g., LLM feedback) as JSON string
                    FOREIGN KEY (knowledge_id) REFERENCES concept_knowledge (knowledge_id) ON DELETE CASCADE
                )
            """)
            self.conn.commit()
            print("ProfileManager: Tables ensured/created successfully.")
        except sqlite3.Error as e:
            print(f"ProfileManager: Error creating tables: {e}")
            if self.conn: self.conn.rollback() # Rollback on error
            raise

    def close_db(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("ProfileManager: Database connection closed.")

    def create_profile(self, learner_id: str) -> bool:
        """
        Creates a new learner profile if one doesn't already exist.

        Args:
            learner_id: The unique identifier for the learner.

        Returns:
            True if the profile was created or already existed, False on error.
        """
        if not self.cursor or not self.conn: return False
        try:
            self.cursor.execute("INSERT OR IGNORE INTO learners (learner_id) VALUES (?)", (learner_id,))
            self.conn.commit()
            # Check if a new row was inserted or if it already existed
            return self.cursor.rowcount >= 0 # Returns 1 if inserted, 0 if ignored (already exists), -1 on error for some drivers
        except sqlite3.Error as e:
            print(f"ProfileManager: Error creating profile for learner '{learner_id}': {e}")
            self.conn.rollback()
            return False

    def get_profile(self, learner_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a learner's profile information.

        Args:
            learner_id: The unique identifier for the learner.

        Returns:
            A dictionary containing the learner's profile data, or None if not found.
        """
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

        Args:
            learner_id: The learner's ID.
            progress: The new progress value (e.g., a float between 0.0 and 1.0).

        Returns:
            True if successful, False otherwise.
        """
        if not self.cursor or not self.conn: return False
        try:
            self.cursor.execute("UPDATE learners SET overall_progress = ? WHERE learner_id = ?", (progress, learner_id))
            self.conn.commit()
            return self.cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"ProfileManager: Error updating overall progress for learner '{learner_id}': {e}")
            self.conn.rollback()
            return False

    def get_concept_knowledge(self, learner_id: str, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the knowledge state for a specific concept for a learner.

        Args:
            learner_id: The learner's ID.
            concept_id: The concept's unique identifier.

        Returns:
            A dictionary with the concept knowledge data, or None if not found.
        """
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

    def update_concept_knowledge(self, 
                                 learner_id: str, 
                                 concept_id: str, 
                                 score: float, 
                                 answered_correctly: bool,
                                 raw_eval_data: Optional[Dict] = None) -> bool:
        """
        Updates the learner's knowledge state for a specific concept and logs the score.
        If the concept knowledge entry doesn't exist, it creates one.

        Args:
            learner_id: The learner's ID.
            concept_id: The concept's ID.
            score: The new score for this attempt (e.g., 0.0 to 10.0).
            answered_correctly: Boolean indicating if the answer was correct.
            raw_eval_data: Optional dictionary with raw evaluation details from the evaluator.

        Returns:
            True if successful, False otherwise.
        """
        if not self.cursor or not self.conn: return False
        
        now_timestamp = datetime.datetime.now()
        raw_eval_json = json.dumps(raw_eval_data) if raw_eval_data else None

        try:
            # Check if entry exists
            self.cursor.execute("""
                SELECT knowledge_id, total_attempts, correct_attempts FROM concept_knowledge 
                WHERE learner_id = ? AND concept_id = ?
            """, (learner_id, concept_id))
            row = self.cursor.fetchone()

            knowledge_id: Optional[int] = None
            new_total_attempts: int = 1
            new_correct_attempts: int = 1 if answered_correctly else 0

            if row: # Update existing entry
                knowledge_id = row["knowledge_id"]
                new_total_attempts = row["total_attempts"] + 1
                new_correct_attempts = row["correct_attempts"] + (1 if answered_correctly else 0)
                
                self.cursor.execute("""
                    UPDATE concept_knowledge 
                    SET current_score = ?, last_answered_correctly = ?, 
                        total_attempts = ?, correct_attempts = ?, last_attempted_at = ?
                    WHERE knowledge_id = ?
                """, (score, 1 if answered_correctly else 0, new_total_attempts, new_correct_attempts, now_timestamp, knowledge_id))
            else: # Insert new entry
                self.cursor.execute("""
                    INSERT INTO concept_knowledge 
                        (learner_id, concept_id, current_score, last_answered_correctly, 
                         total_attempts, correct_attempts, last_attempted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (learner_id, concept_id, score, 1 if answered_correctly else 0, 
                      new_total_attempts, new_correct_attempts, now_timestamp))
                knowledge_id = self.cursor.lastrowid # Get the ID of the newly inserted row

            if knowledge_id is None: # Should not happen if insert/update was successful
                print(f"ProfileManager: Failed to get knowledge_id for learner '{learner_id}', concept '{concept_id}'.")
                self.conn.rollback()
                return False

            # Log this score in the history
            self.cursor.execute("""
                INSERT INTO score_history (knowledge_id, score, raw_eval_data, timestamp)
                VALUES (?, ?, ?, ?)
            """, (knowledge_id, score, raw_eval_json, now_timestamp))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"ProfileManager: Error updating concept knowledge for learner '{learner_id}', concept '{concept_id}': {e}")
            self.conn.rollback()
            return False

    def get_score_history(self, learner_id: str, concept_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the score history for a specific concept for a learner.
        
        Args:
            learner_id: The learner's ID.
            concept_id: The concept's ID.

        Returns:
            A list of dictionaries, each representing a historical score entry, ordered by timestamp.
        """
        if not self.cursor: return []
        history = []
        try:
            # First, get the knowledge_id for the learner and concept
            self.cursor.execute("""
                SELECT knowledge_id FROM concept_knowledge
                WHERE learner_id = ? AND concept_id = ?
            """, (learner_id, concept_id))
            knowledge_row = self.cursor.fetchone()

            if knowledge_row:
                knowledge_id = knowledge_row["knowledge_id"]
                self.cursor.execute("""
                    SELECT timestamp, score, raw_eval_data FROM score_history
                    WHERE knowledge_id = ? ORDER BY timestamp ASC
                """, (knowledge_id,))
                for row in self.cursor.fetchall():
                    entry = dict(row)
                    if entry.get('raw_eval_data'):
                        try:
                            entry['raw_eval_data'] = json.loads(entry['raw_eval_data'])
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse raw_eval_data for history entry.")
                            # Keep as string if not parsable
                    history.append(entry)
            else:
                print(f"ProfileManager: No concept knowledge found for learner '{learner_id}', concept '{concept_id}' to get history.")
        except sqlite3.Error as e:
            print(f"ProfileManager: Error retrieving score history: {e}")
        return history


if __name__ == '__main__':
    print("--- LearnerProfileManager Demo ---")
    # Use an in-memory DB or a temporary file for demo
    # Ensure DATA_DIR exists if using DEFAULT_DB_PATH
    if not os.path.exists(config.DATA_DIR): os.makedirs(config.DATA_DIR)
    
    # For a clean demo, remove old db if it exists
    demo_db_path = os.path.join(config.DATA_DIR, "demo_learner_profiles.sqlite3")
    if os.path.exists(demo_db_path):
        os.remove(demo_db_path)
        print(f"Removed old demo database: {demo_db_path}")

    pm = LearnerProfileManager(db_path=demo_db_path)

    learner1_id = "learner_001"
    learner2_id = "learner_002"
    concept_math_101 = "math_vectors_intro"
    concept_bio_101 = "bio_photosynthesis_basics"

    # Create profiles
    print(f"\nCreating profile for {learner1_id}: {pm.create_profile(learner1_id)}")
    print(f"Creating profile for {learner2_id}: {pm.create_profile(learner2_id)}")
    print(f"Attempting to re-create profile for {learner1_id}: {pm.create_profile(learner1_id)}") # Should be ignored

    # Get profiles
    print(f"\nProfile for {learner1_id}: {pm.get_profile(learner1_id)}")
    print(f"Profile for non_existent_learner: {pm.get_profile('non_existent_learner')}")

    # Update overall progress
    print(f"\nUpdating progress for {learner1_id}: {pm.update_overall_progress(learner1_id, 0.25)}")
    print(f"Profile for {learner1_id} after progress update: {pm.get_profile(learner1_id)}")

    # Update concept knowledge
    print(f"\nUpdating concept '{concept_math_101}' for {learner1_id} (score: 7.5, correct: True)")
    pm.update_concept_knowledge(learner1_id, concept_math_101, 7.5, True, {"feedback": "Good understanding!"})
    
    print(f"\nUpdating concept '{concept_math_101}' for {learner1_id} (score: 5.0, correct: False)")
    pm.update_concept_knowledge(learner1_id, concept_math_101, 5.0, False, {"feedback": "Needs review."})

    print(f"\nUpdating concept '{concept_bio_101}' for {learner1_id} (score: 9.0, correct: True)")
    pm.update_concept_knowledge(learner1_id, concept_bio_101, 9.0, True)


    # Get concept knowledge
    print(f"\nKnowledge of '{concept_math_101}' for {learner1_id}: {pm.get_concept_knowledge(learner1_id, concept_math_101)}")
    print(f"Knowledge of '{concept_bio_101}' for {learner1_id}: {pm.get_concept_knowledge(learner1_id, concept_bio_101)}")
    
    # Get score history
    print(f"\nScore history for '{concept_math_101}' for {learner1_id}:")
    history1 = pm.get_score_history(learner1_id, concept_math_101)
    for entry in history1:
        print(f"  - Timestamp: {entry['timestamp']}, Score: {entry['score']}, Eval: {entry.get('raw_eval_data')}")

    print(f"\nScore history for '{concept_bio_101}' for {learner1_id}:")
    history2 = pm.get_score_history(learner1_id, concept_bio_101)
    for entry in history2:
        print(f"  - Timestamp: {entry['timestamp']}, Score: {entry['score']}, Eval: {entry.get('raw_eval_data')}")

    pm.close_db()
    print("\n--- LearnerProfileManager Demo Finished ---")

