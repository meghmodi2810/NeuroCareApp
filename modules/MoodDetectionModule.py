import os
import cv2
import numpy as np
import tensorflow as tf
from threading import Thread, Lock
from queue import Queue, Empty
import time


class MoodDetector:
    """
    A class to handle mood detection from a webcam feed in a separate thread.
    It uses a queue to communicate results back to the main application thread.
    """

    def __init__(self):
        # --- Constants ---
        self.EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        # Model expects 160x160 input with 3 channels, based on the specific model used.
        self.MODEL_INPUT_SIZE = (160, 160)

        # Determine the correct path to the model file
        script_dir = os.path.dirname(__file__)
        # This path assumes 'best_mobilenet_model.h5' is in a 'static' folder
        # that is a sibling to the 'modules' folder (i.e., NeuroCareApp/static/).
        # Adjust if your model file is named differently or in another location.
        model_path = os.path.join(script_dir, 'static', 'fina;_mobilenet_model.h5')

        # --- Camera Setup - Initialize to None, will be set in start_webcam_detection ---
        self.cap = None

        # --- Model Loading ---
        print(f"Loading emotion detection model from: {model_path}...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            # Print the model's expected input shape for confirmation.
            print(f"Model's expected input shape: {self.model.input_shape}")
            print("Model loaded successfully.")
        except Exception as e:
            print(f"!!! FATAL: Could not load model from {model_path}. Error: {e}")
            self.model = None  # Mark model as unusable if loading fails

        # --- Threading and Communication ---
        self.latest_frame = None
        self.latest_result = {"mood": "Initializing...", "confidence": 0.0}
        self.lock = Lock()
        self.running = False  # Flag to control the update thread; initially False.
        self.event_queue = Queue()  # Queue for sending mood events to the main thread.
        self.thread = None  # The background processing thread; initialized to None.

        print("MoodDetector initialized, webcam detection not started yet.")

    def start_webcam_detection(self):
        """
        Initializes the camera and starts the mood detection background thread.
        This method should be called explicitly when the user chooses webcam detection.
        Returns True if successful, False otherwise.
        """
        if self.running:
            print("Webcam detection already running.")
            return True

        # Check if the model was loaded successfully during initialization.
        if self.model is None:
            print("Cannot start webcam detection: Model failed to load during MoodDetector initialization.")
            return False

        print("Initializing camera for webcam detection...")
        # Attempt to open the default webcam (camera index 0).
        self.cap = cv2.VideoCapture(0)
        time.sleep(1.0)  # Give the camera hardware a moment to initialize.

        if not self.cap.isOpened():
            print(
                "!!! FATAL: Cannot open webcam. Please check if camera is connected and not in use by another application.")
            self.cap = None  # Explicitly set to None if camera fails to open.
            return False
        else:
            print("Camera initialized successfully for webcam detection.")

        # Set specific frame width and height for consistency, though model input size is separate.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True  # Set the running flag to True before starting the thread.
        # Create and start the background thread that continuously updates frames and predictions.
        self.thread = Thread(target=self.update,
                             daemon=True)  # daemon=True allows the thread to exit when main program exits.
        self.thread.start()
        print("Mood detection thread started.")
        return True

    def is_running(self):
        """
        Checks if the MoodDetector is actively running webcam detection.
        This includes checking if the camera is open, the model is loaded, and the processing thread is alive.
        """
        return (self.cap is not None and self.cap.isOpened() and
                self.model is not None and
                self.thread is not None and self.thread.is_alive())

    def _preprocess_frame(self, frame):
        """
        Prepares a single video frame for input into the TensorFlow model.
        It converts the frame to RGB, resizes it, normalizes pixel values,
        and adds a batch dimension as expected by the model.
        """
        try:
            if frame is None:
                # Log a warning if an empty frame is passed for preprocessing.
                print("Warning: Received None frame for preprocessing.")
                return None

            # Convert frame from BGR (OpenCV default) to RGB, as most models expect RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the RGB frame to the specific input dimensions required by the model.
            resized_frame = cv2.resize(rgb_frame, self.MODEL_INPUT_SIZE)

            # Normalize pixel values from [0, 255] to [-1, 1], common for pre-trained models like MobileNet.
            normalized_frame = (resized_frame / 127.5) - 1.0

            # Add a batch dimension (at axis 0) to make the shape (1, height, width, channels),
            # which is the format TensorFlow models expect for a single image prediction.
            processed_input = np.expand_dims(normalized_frame, axis=0)

            return processed_input
        except cv2.error as e:
            print(f"OpenCV error during preprocessing: {e}")
            return None
        except Exception as e:
            print(f"Error during frame preprocessing: {e}")
            return None

    def update(self):
        """
        This method runs in a dedicated background thread. It continuously
        captures frames from the webcam, preprocesses them, and uses the
        loaded model to predict mood. Mood changes are pushed to a queue.
        """
        frame_count = 0
        # Critical check: Ensure camera and model are ready before starting the loop.
        if self.cap is None or self.model is None or not self.cap.isOpened():
            print("Camera or model not available for update loop. Stopping.")
            self.running = False
            return

        while self.running:
            # Read a frame from the camera.
            success, frame = self.cap.read()
            if not success:
                print("Failed to grab frame from camera. Stopping thread.")
                self.running = False  # Signal to stop the thread if frame reading fails.
                break

            # Store the latest raw frame for the get_frame() method to retrieve and annotate.
            with self.lock:
                self.latest_frame = frame.copy()

            frame_count += 1
            # Process only every 5th frame to reduce CPU load and allow for real-time performance.
            if frame_count % 5 != 0:
                continue

            # Preprocess the captured frame for model input.
            processed_frame = self._preprocess_frame(frame)
            if processed_frame is None:
                continue  # Skip prediction if preprocessing failed for the current frame.

            try:
                # Perform mood prediction using the loaded TensorFlow model.
                predictions = self.model.predict(processed_frame, verbose=0)[0]
                # Get the index of the highest probability, which corresponds to the detected emotion.
                emotion_idx = np.argmax(predictions)
                mood = self.EMOTION_LABELS[emotion_idx]
                # Calculate confidence as the highest probability percentage.
                confidence = np.max(predictions) * 100
                current_result = {"mood": mood, "confidence": round(float(confidence), 2)}

                with self.lock:
                    # Only put a new mood event on the queue if the detected mood has changed.
                    # This prevents flooding the queue with redundant events.
                    if self.latest_result['mood'] != current_result['mood']:
                        self.event_queue.put(current_result)
                    self.latest_result = current_result  # Update the latest result regardless of change.

            except Exception as e:
                # Log any errors during model prediction but do not stop the thread
                # as it might be a transient issue. Add a small delay to prevent
                # the loop from spinning too fast on repeated errors.
                print(f"An error occurred during model prediction: {e}")
                time.sleep(0.01)  # Small delay to prevent rapid error looping.

    def get_frame(self):
        """
        Retrieves the latest captured frame, annotates it with the detected mood
        and confidence, and encodes it as a JPEG byte stream suitable for Flask
        video streaming. Returns a placeholder image if webcam is not active.
        """
        # If the camera is not active (either not started or failed to open),
        # return a static placeholder image with an informative message.
        if self.cap is None or not self.cap.isOpened():
            placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame.
            text = "Webcam not active/available. Choose an option."
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            # Center the text on the placeholder frame.
            text_x = (placeholder_frame.shape[1] - text_width) // 2
            text_y = (placeholder_frame.shape[0] + text_height) // 2
            cv2.putText(placeholder_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
            _, buffer = cv2.imencode('.jpg', placeholder_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buffer.tobytes()

        with self.lock:
            # If camera is running but no frames have been captured yet, create a blank frame
            # and then encode it to bytes. This fixes the ValueError.
            if self.latest_frame is None:
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return buffer.tobytes()

            frame = self.latest_frame.copy()  # Get a copy of the latest frame to avoid modification issues.
            frame = cv2.flip(frame, 1)
            result = self.latest_result.copy()  # Get a copy of the latest mood result.

        mood = result.get('mood', 'N/A')
        confidence = result.get('confidence', 0)
        text = f"{mood} ({confidence:.1f}%)"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        padding = 10
        rect_start = (padding, padding)
        rect_end = (padding + text_width + padding, padding + text_height + baseline + padding)
        text_origin = (padding * 2, padding * 2 + text_height)

        # Draw a black rectangle behind the text for better readability.
        cv2.rectangle(frame, rect_start, rect_end, (0, 0, 0), -1)
        # Put the mood and confidence text on the frame in white.
        cv2.putText(frame, text, text_origin, font, font_scale, (255, 255, 255), font_thickness)

        # Encode the annotated frame as a JPEG image.
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes()

    def get_mood_event(self):
        """
        Retrieves the next mood event from the internal queue.
        This method is non-blocking with a short timeout.
        """
        if not self.running:
            return None
        try:
            # Get an item from the queue, blocking for a maximum of 0.1 seconds.
            return self.event_queue.get(block=True, timeout=0.1)
        except Empty:
            return None  # Return None if no event is available within the timeout.

    def stop(self):
        """
        Gracefully stops the mood detection background thread and releases
        the webcam resource.
        """
        print("Stopping mood detector.")
        self.running = False  # Signal the update loop to terminate.
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)  # Wait for the thread to finish for up to 1 second.
            if self.thread.is_alive():
                print("Warning: Mood detection thread did not stop gracefully.")

        if self.cap and self.cap.isOpened():
            self.cap.release()  # Release the camera resource.
            self.cap = None  # Clear the camera object.
        print("Mood detector stopped.")

    def __del__(self):
        """
        Destructor to ensure resources are released when the MoodDetector object
        is garbage collected. Calls stop() if the detector is still running.
        """
        if hasattr(self, 'running') and self.running:
            self.stop()
