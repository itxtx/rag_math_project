# src/adaptive_engine/srs_scheduler.py
import datetime
from typing import Dict, Any, Optional

# --- Configuration for Simplified SRS ---
# These can be tuned or made more complex later (e.g., per-item ease factors)
INITIAL_INTERVAL_DAYS_CORRECT = 1  # Interval after first correct answer
SECOND_INTERVAL_DAYS_CORRECT = 3 # Interval after second consecutive correct answer
INTERVAL_MULTIPLIER_CORRECT = 2.0  # Multiplier for subsequent correct answers
INTERVAL_RESET_DAYS_INCORRECT = 1  # Interval if answered incorrectly
MAX_INTERVAL_DAYS = 180            # Maximum interval cap (e.g., 6 months)
MIN_INTERVAL_DAYS = 1              # Minimum interval for next review

class SRSScheduler:
    """
    Calculates the next review date for a concept based on spaced repetition principles.
    This is a simplified implementation inspired by SM-2 concepts but without an explicit ease factor.
    The "ease" is implicitly handled by how the interval grows based on consecutive correct repetitions.
    """

    def __init__(self):
        print("SRSScheduler initialized.")

    def calculate_next_review_details(self,
                                      answered_correctly: bool,
                                      current_srs_repetitions: int, # Number of times this item has been reviewed *correctly in a row*
                                      current_srs_interval_days: int # Current interval in days since the last review
                                     ) -> Dict[str, Any]:
        """
        Calculates the next review interval (in days) and the new number of consecutive correct repetitions.

        Args:
            answered_correctly: Boolean indicating if the learner answered correctly.
            current_srs_repetitions: How many times this item has been recalled correctly in a row
                                     (0 if last answer was wrong or it's the first time being learned).
            current_srs_interval_days: The current interval in days.
                                       (e.g., 1 for first review, 0 if never reviewed or just reset).

        Returns:
            A dictionary with 'next_interval_days' (int) and 'new_srs_repetitions' (int).
            'next_review_at' will be calculated based on 'next_interval_days'.
        """
        next_interval_days: int
        new_srs_repetitions: int

        if answered_correctly:
            new_srs_repetitions = current_srs_repetitions + 1
            if new_srs_repetitions == 1: # First time correct (or first time after incorrect)
                next_interval_days = INITIAL_INTERVAL_DAYS_CORRECT
            elif new_srs_repetitions == 2: # Second time correct in a row
                next_interval_days = SECOND_INTERVAL_DAYS_CORRECT
            else: # Subsequent correct answers (3rd correct onwards)
                # Apply multiplier to the *previous* interval
                # If current_srs_interval_days was from the (n-1)th correct answer,
                # the new interval is roughly current_srs_interval_days * multiplier
                next_interval_days = int(round(max(current_srs_interval_days, MIN_INTERVAL_DAYS) * INTERVAL_MULTIPLIER_CORRECT))
            
            next_interval_days = min(next_interval_days, MAX_INTERVAL_DAYS)
            next_interval_days = max(next_interval_days, MIN_INTERVAL_DAYS) # Ensure it's at least min

        else: # Answered incorrectly
            new_srs_repetitions = 0 # Reset repetition count
            next_interval_days = INTERVAL_RESET_DAYS_INCORRECT 

        next_review_at_datetime = datetime.datetime.now() + datetime.timedelta(days=next_interval_days)
        
        print(f"SRSScheduler: Correct: {answered_correctly}, Reps: {current_srs_repetitions}->{new_srs_repetitions}, "
              f"Interval: {current_srs_interval_days}->{next_interval_days}, Next Review Date: {next_review_at_datetime.strftime('%Y-%m-%d')}")

        return {
            "next_interval_days": next_interval_days,
            "next_review_at": next_review_at_datetime, # Store as datetime object
            "new_srs_repetitions": new_srs_repetitions
        }

if __name__ == '__main__':
    print("--- SRSScheduler Demo ---")
    scheduler = SRSScheduler()
    
    # Simulate learning a new item
    print("\nNew item, answered correctly (1st time):")
    # current_repetitions = 0 (new), current_interval = 0 (new)
    srs_data = scheduler.calculate_next_review_details(True, 0, 0) 
    print(srs_data) # Expected: interval=1, reps=1

    print("\nReviewed (1st rep), answered correctly (2nd time correct):")
    # current_repetitions = 1, current_interval = 1 (from previous step)
    srs_data = scheduler.calculate_next_review_details(True, 1, srs_data['next_interval_days']) 
    print(srs_data) # Expected: interval=3 (or 2.5 rounded), reps=2

    print("\nReviewed (2nd rep), answered correctly (3rd time correct):")
    # current_repetitions = 2, current_interval = 3
    srs_data = scheduler.calculate_next_review_details(True, 2, srs_data['next_interval_days']) 
    print(srs_data) # Expected: interval=6 (3*2), reps=3

    print("\nReviewed (3rd rep), answered incorrectly:")
    # current_repetitions = 3, current_interval = 6
    srs_data = scheduler.calculate_next_review_details(False, 3, srs_data['next_interval_days'])
    print(srs_data) # Expected: interval=1, reps=0

    print("\nReviewed after incorrect, answered correctly (1st time correct again):")
    # current_repetitions = 0, current_interval = 1
    srs_data = scheduler.calculate_next_review_details(True, 0, srs_data['next_interval_days'])
    print(srs_data) # Expected: interval=1, reps=1
    
    print("\n--- SRSScheduler Demo Finished ---")
