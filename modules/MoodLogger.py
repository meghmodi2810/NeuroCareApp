import csv
from datetime import datetime
import os

LOG_PATH = "static/mood_log.csv"

def log_mood(mood, confidence):
    os.makedirs("static", exist_ok=True)
    with open(LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().isoformat(), mood, round(confidence, 2)])

def read_mood_log():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, newline='') as file:
        return list(csv.reader(file))
