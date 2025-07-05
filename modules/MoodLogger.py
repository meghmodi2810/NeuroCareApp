import csv
from datetime import datetime
import os
from collections import namedtuple

LOG_PATH = "static/mood_log.csv"
MoodEntry = namedtuple('MoodEntry', ['timestamp', 'mood', 'confidence'])


def log_mood(mood, confidence):
    os.makedirs("static", exist_ok=True)
    with open(LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().isoformat(), mood, round(confidence, 2)])


def read_mood_log():
    if not os.path.exists(LOG_PATH):
        return []

    entries = []
    with open(LOG_PATH, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 3:  # Ensure we have all three fields
                try:
                    entries.append(MoodEntry(
                        timestamp=row[0],
                        mood=row[1],
                        confidence=float(row[2])
                    ))
                except (ValueError, IndexError):
                    continue
    return entries