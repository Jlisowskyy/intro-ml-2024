const recordButton = document.getElementById('recordButton');
const loadButton = document.getElementById('loadButton');
const audioInput = document.getElementById('audioInput');
const resultDiv = document.getElementById('result');
const wavUrl = 'http://127.0.0.1:8000/run/model/wav';

let audioContext;

function convertToWav(audioBuffer) {
    const numOfChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const length = audioBuffer.length * numOfChannels;
    const buffer = new Float32Array(length);

    for (let channel = 0; channel < numOfChannels; channel++) {
        const channelData = audioBuffer.getChannelData(channel);
        for (let i = 0; i < audioBuffer.length; i++) {
            buffer[i * numOfChannels + channel] = channelData[i];
        }
    }

    const wavBuffer = new ArrayBuffer(44 + buffer.length * 2);
    const view = new DataView(wavBuffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + buffer.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numOfChannels * 2, true);
    view.setUint16(32, numOfChannels * 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, buffer.length * 2, true);

    const length16 = buffer.length;
    const index = 44;
    for (let i = 0; i < length16; i++) {
        view.setInt16(index + i * 2, buffer[i] * 0x7FFF, true);
    }

    return new Blob([wavBuffer], {type: 'audio/wav'});
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

recordButton.addEventListener('click', async () => {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Your browser does not support audio recording.');
        }

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const mediaRecorder = new MediaRecorder(stream);
        const audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks);
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            const wavBlob = convertToWav(audioBuffer);

            await sendAudio(wavBlob);
            stream.getTracks().forEach(track => track.stop());
        };

        recordButton.disabled = true;
        recordButton.textContent = 'Recording...';
        recordButton.classList.add('recording');

        mediaRecorder.start();

        setTimeout(() => {
            mediaRecorder.stop();
            recordButton.disabled = false;
            recordButton.textContent = 'Record Audio (3 seconds)';
            recordButton.classList.remove('recording');
        }, 3000);
    } catch (error) {
        console.error('Error:', error);
        showResult(`Error: ${error.message}`, false);
    }
});

loadButton.addEventListener('click', () => {
    audioInput.click();
});

audioInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        await sendAudio(file);
    }
});

async function sendAudio(wavBlob) {
    const formData = new FormData();
    formData.append('file', wavBlob, 'audio.wav');

    try {
        resultDiv.textContent = 'Uploading audio...';
        const response = await fetch(wavUrl, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            showResult(result.response, true);
        } else {
            showResult('Error uploading audio.', false);
        }
    } catch (error) {
        console.error('Error:', error);
        showResult('Error uploading audio.', false);
    }
}

function showResult(message, isSuccess) {
    resultDiv.textContent = message;
    resultDiv.className = isSuccess ? 'success' : 'error';
}