document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const video = document.getElementById('video');
    const scanBtn = document.getElementById('scanBtn');
    const resultsCard = document.getElementById('results');
    const moodElement = document.getElementById('mood');
    const confidenceElement = document.getElementById('confidence');
    const suggestionElement = document.getElementById('suggestion');
    const spinner = document.getElementById('spinner');

    let isScanning = false;
    let eventSource = null;

    // --- Mood Color Mapping ---
    const moodClasses = {
        'Happy': 'bg-success',
        'Sad': 'bg-primary',
        'Angry': 'bg-danger',
        'Surprise': 'bg-warning text-dark',
        'Fear': 'bg-dark',
        'Disgust': 'bg-brown', // You'll need to define this class in CSS
        'Neutral': 'bg-secondary',
        'Initializing...': 'bg-secondary',
    };

    /**
     * Updates the UI with the new mood data.
     * @param {object} data - The mood data from the server.
     */
    function updateMoodUI(data) {
        if (data.error) {
            console.error("Server Error:", data.error);
            moodElement.textContent = "Error";
            suggestionElement.textContent = data.error;
            confidenceElement.textContent = "";
            moodElement.className = 'badge fs-5 bg-danger';
            stopScan(); // Stop everything if there's an error
            return;
        }

        console.log("Mood data received:", data);
        moodElement.textContent = data.mood;
        confidenceElement.textContent = `(${data.confidence}%)`;
        suggestionElement.textContent = data.suggestion;

        // Update badge color
        const moodClass = moodClasses[data.mood] || 'bg-secondary';
        moodElement.className = `badge fs-5 ${moodClass}`;
    }


    /**
     * Starts the mood detection stream.
     */
    function startScan() {
        // Show loading spinner and hide results initially
        spinner.classList.remove('d-none');
        resultsCard.classList.add('d-none');

        // Set video feed source
        video.src = "/video_feed";
        video.style.display = 'block'; // Ensure video is visible

        // --- Initialize Server-Sent Events (SSE) ---
        if (eventSource) {
            eventSource.close();
        }
        eventSource = new EventSource('/mood_stream');

        eventSource.onopen = () => {
            console.log("Connection to mood stream opened.");
        };

        eventSource.onmessage = (event) => {
            // Hide spinner and show results on first message
            spinner.classList.add('d-none');
            resultsCard.classList.remove('d-none');

            const data = JSON.parse(event.data);
            updateMoodUI(data);
        };

        eventSource.onerror = (err) => {
            console.error("EventSource failed:", err);
            updateMoodUI({ error: "Connection to server lost. Please try again." });
            eventSource.close();
        };

        // Update button UI
        scanBtn.innerHTML = '<i class="bi bi-stop-circle-fill me-2"></i> Stop Scan';
        scanBtn.classList.replace('btn-success', 'btn-danger');
        isScanning = true;
    }

    /**
     * Stops the mood detection stream.
     */
    function stopScan() {
        if (eventSource) {
            eventSource.close();
            console.log("Connection to mood stream closed.");
        }

        // Clear video feed
        video.src = "";
        video.style.display = 'none';

        // Hide results and spinner
        resultsCard.classList.add('d-none');
        spinner.classList.add('d-none');


        // Reset button UI
        scanBtn.innerHTML = '<i class="bi bi-camera-video me-2"></i> Start Live Scan';
        scanBtn.classList.replace('btn-danger', 'btn-success');
        isScanning = false;
    }

    /**
     * Toggles the scanning state.
     */
    function toggleScan() {
        if (isScanning) {
            stopScan();
        } else {
            startScan();
        }
    }

    // --- Event Listener ---
    if (scanBtn) {
        scanBtn.addEventListener('click', toggleScan);
    }
});
