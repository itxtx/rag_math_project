import pytest
import os
import sqlite3
import time
import datetime
from src.learner_model.profile_manager import LearnerProfileManager
from src import config

@pytest.fixture
def temp_db_path(tmp_path):
    db_path = tmp_path / "test_learner_profiles.sqlite3"
    return str(db_path)

@pytest.fixture
def profile_manager(temp_db_path):
    pm = LearnerProfileManager(db_path=temp_db_path)
    yield pm
    pm.close_db()
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

@pytest.fixture
def test_ids():
    return {
        "learner_id_1": "test_learner_001",
        "concept_id_1": "concept_alpha",
        "concept_id_2": "concept_beta",
        "doc_id_1": "doc_one"
    }

def test_database_connection_and_table_creation(profile_manager):
    assert profile_manager.conn is not None
    assert profile_manager.cursor is not None
    profile_manager.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learners';")
    assert profile_manager.cursor.fetchone() is not None
    profile_manager.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concept_knowledge';")
    assert profile_manager.cursor.fetchone() is not None
    profile_manager.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='score_history';")
    assert profile_manager.cursor.fetchone() is not None

def test_create_and_get_profile(profile_manager, test_ids):
    created = profile_manager.create_profile(test_ids["learner_id_1"])
    assert created is True
    profile = profile_manager.get_profile(test_ids["learner_id_1"])
    assert profile is not None
    assert profile["learner_id"] == test_ids["learner_id_1"]
    assert profile["overall_progress"] == 0.0
    assert "created_at" in profile
    created_again = profile_manager.create_profile(test_ids["learner_id_1"])
    assert created_again is False
    non_existent_profile = profile_manager.get_profile("non_existent_learner")
    assert non_existent_profile is None

def test_update_overall_progress(profile_manager, test_ids):
    profile_manager.create_profile(test_ids["learner_id_1"])
    updated = profile_manager.update_overall_progress(test_ids["learner_id_1"], 0.55)
    assert updated is True
    profile = profile_manager.get_profile(test_ids["learner_id_1"])
    assert profile["overall_progress"] == 0.55
    not_updated = profile_manager.update_overall_progress("non_existent_learner", 0.5)
    assert not_updated is False

def test_update_and_get_concept_knowledge(profile_manager, test_ids):
    profile_manager.create_profile(test_ids["learner_id_1"])
    raw_eval_1 = {"feedback": "Good start!"}
    srs_details_1 = {"next_interval_days": 1, "next_review_at": datetime.datetime.now(), "new_srs_repetitions": 1}
    
    updated1 = profile_manager.update_concept_srs_and_difficulty(
        test_ids["learner_id_1"], test_ids["concept_id_1"], test_ids["doc_id_1"], 7.0, True, srs_details_1, raw_eval_1)
    assert updated1 is True
    
    knowledge1 = profile_manager.get_concept_knowledge(test_ids["learner_id_1"], test_ids["concept_id_1"])
    assert knowledge1 is not None
    assert knowledge1["learner_id"] == test_ids["learner_id_1"]
    assert knowledge1["concept_id"] == test_ids["concept_id_1"]
    assert knowledge1["current_score"] == 7.0
    assert knowledge1["last_answered_correctly"] == 1
    assert knowledge1["total_attempts"] == 1
    assert knowledge1["correct_attempts"] == 1
    assert knowledge1["last_attempted_at"] is not None
    time.sleep(0.01)
    raw_eval_2 = {"feedback": "Improved slightly."}
    updated2 = profile_manager.update_concept_srs_and_difficulty(test_ids["learner_id_1"], test_ids["concept_id_1"], test_ids["doc_id_1"], 6.0, False, srs_details_1, raw_eval_2)
    assert updated2 is True
    knowledge2 = profile_manager.get_concept_knowledge(test_ids["learner_id_1"], test_ids["concept_id_1"])
    assert knowledge2["current_score"] == 6.0
    assert knowledge2["last_answered_correctly"] == 0
    assert knowledge2["total_attempts"] == 2
    assert knowledge2["correct_attempts"] == 1
    assert knowledge1["last_attempted_at"] != knowledge2["last_attempted_at"]
    non_existent_knowledge = profile_manager.get_concept_knowledge(test_ids["learner_id_1"], "non_existent_concept")
    assert non_existent_knowledge is None

def test_get_score_history(profile_manager, test_ids):
    profile_manager.create_profile(test_ids["learner_id_1"])
    raw_eval_1 = {"llm_feedback": "Attempt 1 feedback"}
    srs_details = {"next_interval_days": 1, "next_review_at": datetime.datetime.now(), "new_srs_repetitions": 1}

    profile_manager.update_concept_srs_and_difficulty(
        test_ids["learner_id_1"], test_ids["concept_id_1"], test_ids["doc_id_1"], 8.0, True, srs_details, raw_eval_1)
    
    history = profile_manager.get_score_history(test_ids["learner_id_1"], test_ids["concept_id_1"])
    assert len(history) == 1
    assert history[0]["score"] == 8.0
    assert history[0]["raw_eval_data"] == raw_eval_1
    assert "timestamp" in history[0]
    assert history[0]["timestamp"] is not None
    no_history = profile_manager.get_score_history(test_ids["learner_id_1"], test_ids["concept_id_2"])
    assert len(no_history) == 0
    no_learner_history = profile_manager.get_score_history("fake_learner", test_ids["concept_id_1"])
    assert len(no_learner_history) == 0

def test_delete_cascade(profile_manager, test_ids):
    profile_manager.create_profile(test_ids["learner_id_1"])
    srs_details = {"next_interval_days": 1, "next_review_at": datetime.datetime.now(), "new_srs_repetitions": 1}
    profile_manager.update_concept_srs_and_difficulty(
        test_ids["learner_id_1"], test_ids["concept_id_1"], test_ids["doc_id_1"], 9.0, True, srs_details)
    
    # Direct deletion for test purposes
    conn = sqlite3.connect(profile_manager.db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM learners WHERE learner_id = ?", (test_ids["learner_id_1"],))
    conn.commit()
    conn.close()

    profile_after_delete = profile_manager.get_profile(test_ids["learner_id_1"])
    assert profile_after_delete is None
    knowledge_after_delete = profile_manager.get_concept_knowledge(test_ids["learner_id_1"], test_ids["concept_id_1"])
    assert knowledge_after_delete is None
    history_after_delete = profile_manager.get_score_history(test_ids["learner_id_1"], test_ids["concept_id_1"])
    assert len(history_after_delete) == 0 