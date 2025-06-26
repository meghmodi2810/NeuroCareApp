def get_suggestion(mood):
    tips = {
        "Happy": "Keep doing what makes you smile!",
        "Sad": "Take a deep breath and talk to a friend.",
        "Angry": "Step away and try journaling.",
        "Surprise": "Embrace the unexpected moment.",
        "Fear": "Ground yourself with calm breathing.",
        "Disgust": "Distract yourself with uplifting content.",
        "Neutral": "Check in with yourself and reflect.",
    }
    return tips.get(mood, "Care for yourself today.")
