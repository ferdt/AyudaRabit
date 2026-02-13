const fileUpload = document.getElementById('file-upload');
const btnTest = document.getElementById('btn-test');
const imgOriginal = document.getElementById('img-original');
const imgProcessed = document.getElementById('img-processed');
const textResult = document.getElementById('text-result');
const executionTimeDisplay = document.getElementById('execution-time');

let currentBase64Image = null;

// Handle File Upload
fileUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        currentBase64Image = event.target.result;
        imgOriginal.src = currentBase64Image;
        imgProcessed.src = ""; // Clear previous
        textResult.textContent = "";

        // Auto-trigger test on new image load
        runTest();
    };
    reader.readAsDataURL(file);
});

// Handle Test Button
btnTest.addEventListener('click', runTest);

// Auto-trigger on input changes
const inputs = document.querySelectorAll('.controls-panel input');
inputs.forEach(input => {
    input.addEventListener('change', runTest);
    // For sliders/numbers, maybe 'input' event is better but 'change' is safer for now to avoid flood
});

async function runTest() {
    if (!currentBase64Image) {
        alert("Por favor, sube una imagen primero.");
        return;
    }

    // UI Loading state
    btnTest.textContent = "Procesando...";
    btnTest.disabled = true;
    document.body.style.cursor = "wait";

    try {
        // Collect Options
        const options = {
            apply_deskew: document.getElementById('opt-deskew').checked,
            apply_gray: document.getElementById('opt-gray').checked,
            apply_threshold: document.getElementById('opt-threshold').checked,
            threshold_block_size: parseInt(document.getElementById('opt-threshold-block').value) || 15,
            threshold_c: parseInt(document.getElementById('opt-threshold-c').value) || 5,
            apply_sharpening: document.getElementById('opt-sharpening').checked,
            apply_contrast: document.getElementById('opt-contrast').checked,
            contrast_alpha: parseFloat(document.getElementById('opt-contrast-alpha').value) || 1.5,
            contrast_beta: parseFloat(document.getElementById('opt-contrast-beta').value) || 0,
            apply_resize: document.getElementById('opt-resize').checked,
            resize_height: parseInt(document.getElementById('opt-resize-height').value) || 1024
        };

        const response = await fetch("http://localhost:5000/ocr", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image: currentBase64Image,
                ocr_engine: 'rapid', // Use rapid for testing prep
                preprocessing: options
            })
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.statusText}`);
        }

        const result = await response.json();

        // Update UI
        if (result.processed_image) {
            imgProcessed.src = result.processed_image;
        }
        textResult.textContent = result.text || "No text detected";
        executionTimeDisplay.textContent = result.execution_time ? result.execution_time.toFixed(3) : "N/A";

    } catch (err) {
        console.error(err);
        textResult.textContent = "Error: " + err.message;
    } finally {
        btnTest.textContent = "Probar Configuración";
        btnTest.disabled = false;
        document.body.style.cursor = "default";
    }
}
