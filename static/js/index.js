const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => { video.srcObject = stream; })
  .catch(err => { console.error("Webcam access error:", err); });

function detectMood() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL('image/jpeg');

  fetch('/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById('moodResult').innerText =
      `Mood: ${data.mood} | Confidence: ${data.confidence}%`;
  })
  .catch(err => {
    console.error('Detection failed:', err);
    document.getElementById('moodResult').innerText = "Error detecting mood.";
  });
}
