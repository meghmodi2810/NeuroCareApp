import os
import time
import json
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import requests  # Import the requests library for making HTTP calls
import cv2  # Import cv2 for encoding blank frames
import numpy as np  # Import numpy for creating blank frames
import datetime  # Import datetime module for potential use, especially with Flask-Moment
from flask_moment import Moment  # Import Flask-Moment

# To suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import modules
from modules.MoodDetectionModule import MoodDetector
from modules.MoodLogger import log_mood, read_mood_log
from modules.MoodSuggestion import get_suggestion

# Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = '12345678'  # It's good practice to have a secret key

# Initialize Flask-Moment with the app
moment = Moment(app)

# Initialize the MoodDetector globally, but don't start the camera yet
try:
    detector = MoodDetector()
except Exception as e:
    print(f"Error initializing MoodDetector: {e}")
    # If the model fails to load, mark detector as None to prevent further issues
    detector = None


# --- Main Routes ---

@app.route('/')
def index():
    """Renders the home page with options for webcam or survey."""
    return render_template('index.html')


@app.route('/webcam_detection')
def webcam_detection_page():
    """Renders the webcam mood detection page. Webcam will start via JS."""
    if detector and detector.model is not None:
        return render_template('webcam_detection.html')
    else:
        # If detector or model failed to load, show an error message
        return render_template('error.html',
                               message="Mood detection system (model) failed to initialize. Please check server logs.")


@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """API endpoint to explicitly start the webcam detection."""
    global detector
    if detector:
        success = detector.start_webcam_detection()
        if success:
            return jsonify({"status": "success", "message": "Webcam detection started."})
        else:
            return jsonify({"status": "error", "message": "Failed to start webcam."}), 500
    else:
        return jsonify({"status": "error", "message": "Mood Detector not initialized."}), 500


@app.route('/survey')
def survey_page():
    """Renders the mood survey page."""
    return render_template('survey.html')


@app.route('/submit_survey', methods=['POST'])
def submit_survey():
    """Processes survey responses and infers mood using an LLM."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    survey_responses = request.get_json()

    # Construct a prompt for the LLM based on survey responses
    prompt_parts = []
    for question, answer in survey_responses.items():
        if answer:  # Only include questions with answers
            prompt_parts.append(f"{question}: {answer}")

    full_prompt = "Based on the following survey responses, what is the predominant mood (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)? Just give one word, the mood. If uncertain, default to Neutral.\n\n" + "\n".join(
        prompt_parts)

    try:
        # Prepare the payload for the LLM API call
        payload_contents = [{"role": "user", "parts": [{"text": full_prompt}]}]

        # Define the schema for the structured response
        payload = {
            "contents": payload_contents,
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "mood": {"type": "STRING"},
                        "reason": {"type": "STRING"}
                    },
                    "propertyOrdering": ["mood", "reason"]
                }
            }
        }

        # Retrieve API key from environment variable
        apiKey = os.environ.get("GEMINI_API_KEY", "")
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"

        # Make the API call using requests library
        response = requests.post(apiUrl, json=payload)

        if not response.ok:
            print(f"LLM API Error: {response.status_code} - {response.text}")
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

        result = response.json()  # Parse JSON response

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0][
            "content"].get("parts"):
            llm_response_text = result["candidates"][0]["content"]["parts"][0]["text"]

            parsed_llm_response = json.loads(llm_response_text)
            inferred_mood = parsed_llm_response.get("mood", "Neutral")
            inferred_reason = parsed_llm_response.get("reason", "Inferred from survey responses.")

            # Ensure the inferred mood is one of the valid labels
            if detector and inferred_mood not in detector.EMOTION_LABELS:
                inferred_mood = "Neutral"  # Default to neutral if LLM gives an unknown mood

            # Log the inferred mood (confidence can be fixed for survey)
            log_mood(inferred_mood, 90.0)  # Assign a high confidence for survey-based moods

            # Get a suggestion for the inferred mood
            suggestion = get_suggestion(inferred_mood)

            return jsonify({
                "status": "success",
                "mood": inferred_mood,
                "confidence": 90.0,
                "suggestion": suggestion,
                "reason": inferred_reason
            })
        else:
            print("LLM response structure unexpected:", result)
            raise Exception("LLM returned an unexpected response structure.")

    except requests.exceptions.RequestException as req_e:
        print(f"Network or API request error: {req_e}")
        log_mood("Uncertain", 50.0)  # Log as uncertain on error
        return jsonify({
            "status": "error",
            "message": f"Network or API error: {req_e}",
            "mood": "Uncertain",
            "confidence": 50.0,
            "suggestion": "We had trouble inferring your mood due to a connection issue.",
            "reason": "Network/API error"
        }), 500
    except json.JSONDecodeError as json_e:
        print(f"JSON decoding error: {json_e}")
        log_mood("Uncertain", 50.0)  # Log as uncertain on error
        return jsonify({
            "status": "error",
            "message": f"Server response not valid JSON: {json_e}",
            "mood": "Uncertain",
            "confidence": 50.0,
            "suggestion": "We had trouble understanding the server's response.",
            "reason": "Invalid server response"
        }), 500
    except Exception as e:
        print(f"Error inferring mood from survey: {e}")
        log_mood("Uncertain", 50.0)  # Log as uncertain on error
        return jsonify({
            "status": "error",
            "message": f"Failed to infer mood from survey: {e}",
            "mood": "Uncertain",
            "confidence": 50.0,
            "suggestion": "We had trouble inferring your mood. Perhaps try the webcam detection?",
            "reason": "Processing error"
        }), 500


@app.route('/history')
def history():
    """Renders the mood history page with data for the chart."""
    try:
        raw_data = read_mood_log()
        # Convert to list of dicts for JSON serialization
        mood_data = [{
            'timestamp': entry.timestamp,
            'mood': entry.mood,
            'confidence': entry.confidence
        } for entry in raw_data]
        return render_template("history.html", mood_data=mood_data)
    except Exception as e:
        print(f"Error loading history: {e}")
        return render_template("history.html", mood_data=[])

# --- API and Streaming Endpoints ---

def frame_generator():
    """
    A generator function that yields camera frames for the video stream.
    It gets the latest annotated frame from the MoodDetector.
    """
    global detector

    # Loop to continuously send frames. It will send placeholder frames
    # until the detector is fully running, then switch to live frames.
    while True:
        frame_bytes = None
        if detector and detector.is_running():
            # If detector is running, get the live frame
            frame_bytes = detector.get_frame()
        else:
            # If detector is not running, get the placeholder frame
            # (which MoodDetector's get_frame handles by returning a static image)
            # or a basic black frame if detector itself is None
            if detector:
                frame_bytes = detector.get_frame()
            else:
                # Fallback: a completely black JPEG if detector is not initialized at all
                _, buffer = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8),
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()

        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Sleep to control FPS, regardless of whether it's a live or placeholder frame
        time.sleep(1.0 / 30)  # Aim for 30 FPS


@app.route('/video_feed')
def video_feed():
    """Streams video frames to the client."""
    return Response(frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def mood_event_stream():
    """
    A generator function that uses Server-Sent Events (SSE) to push mood updates.
    It reads from the detector's queue and sends data to the client.
    """
    global detector
    if not detector:
        error_data = json.dumps({"error": "Camera or model could not be initialized."})
        yield f"data: {error_data}\n\n"
        return

    # Wait for the detector to actually start running before trying to get events
    while not detector.is_running():
        time.sleep(0.5)  # Wait for a bit for webcam to start if chosen

    print("Client connected to mood stream.")
    try:
        while True:
            mood_data = detector.get_mood_event()
            if mood_data:
                log_mood(mood_data['mood'], mood_data['confidence'])
                mood_data['suggestion'] = get_suggestion(mood_data['mood'])
                yield f"data: {json.dumps(mood_data)}\n\n"
    except GeneratorExit:
        print("Client disconnected from mood stream.")


@app.route('/mood_stream')
def mood_stream():
    """Endpoint for the Server-Sent Events (SSE) mood stream."""
    return Response(mood_event_stream(), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)

