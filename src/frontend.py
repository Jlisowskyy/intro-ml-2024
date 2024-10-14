
frontend_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intro ML 2024 Project!</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            text-align: center;
        }
        button {
            margin: 10px;
            padding: 10px;
            font-size: 16px;
        }
        #result {
            margin-top: 20px;
            font-size: 18px;
            color: green;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Intro ML 2024 Project! </h1>
        <button id="recordButton">Record Audio (5 seconds)</button>
        <button id="loadButton">Load Audio File</button>
        <input type="file" id="audioInput" accept="audio/*" style="display:none;">
        <div id="result"></div>
        <h2>Authors:</h2>
        <p>Jakub Lisowski</p>
        <p>Michal Kwiatkowski</p>
        <p>Lukasz Kryczka</p>
        <p>Kuba Pietrzak</p>
        <p>Tomasz Mycielski</p>
    </div>

    <script>
        const recordButton = document.getElementById('recordButton');
        const loadButton = document.getElementById('loadButton');
        const audioInput = document.getElementById('audioInput');
        const resultDiv = document.getElementById('result');
        const apiUrl = 'http://127.0.0.1:8000/run/model';

        recordButton.addEventListener('click', async () => {
            try {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    alert('Your browser does not support audio recording.');
                    return;
                }
    
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const mediaRecorder = new MediaRecorder(stream);
                let audioChunks = [];
    
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
    
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    sendAudio(audioBlob);
                };
    
                mediaRecorder.start();
    
                setTimeout(() => {
                    mediaRecorder.stop();
                }, 5000);
            } catch (error) {
                console.error('Error:', error);
                resultDiv.innerText = `Error recording audio: ${error.message}`;
            }
        });

        loadButton.addEventListener('click', () => {
            audioInput.click();
        });

        audioInput.addEventListener('change', async (event) => {
            const file = event.target.files[0];
            if (file) {
                sendAudio(file);
            }
        });

        async function sendAudio(audioBlob) {
            const formData = new FormData();
            formData.append('file', audioBlob, 'audio.wav');

            try {
                const response = await fetch(apiUrl, {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    resultDiv.innerText = `Response: ${result.response}`;
                } else {
                    resultDiv.innerText = 'Error uploading audio.';
                }
            } catch (error) {
                console.error('Error:', error);
                resultDiv.innerText = 'Error uploading audio.';
            }
        }
    </script>
</body>
</html>
"""