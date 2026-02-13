const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const btnCamera = document.getElementById('btn-camera');
const btnCapture = document.getElementById('btn-capture');
const fileUploadStandard = document.getElementById('file-upload-standard');
const fileUploadDebug = document.getElementById('file-upload-debug');
const statusSection = document.getElementById('status-section');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');
const resultsSection = document.getElementById('results-section');
const resultsTableBody = document.querySelector('#results-table tbody');
const rawTextOutput = document.getElementById('raw-text');
const errorMessage = document.getElementById('error-message');
const btnReset = document.getElementById('btn-reset');

let stream = null;

// Queue State
let processingQueue = [];
let allProcessedData = [];
let totalFilesToProcess = 0;
let processedFilesCount = 0;
let isDebugMode = false;

// ... (processImageWithLocalServer remains same) ...

// Event Listeners
btnCamera.addEventListener('click', startCamera);
btnCapture.addEventListener('click', capturePhoto);

fileUploadStandard.addEventListener('change', (e) => {
    isDebugMode = false;
    handleFileUpload(e);
});

fileUploadDebug.addEventListener('change', (e) => {
    isDebugMode = true;
    handleFileUpload(e);
});

btnReset.addEventListener('click', resetApp);

// Drag & Drop
const dropZone = document.querySelector('.video-container');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (files.length > 0) {
            processFiles(files);
        } else {
            alert('Por favor, sube archivos de imagen válidos.');
        }
    }
});

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' } // Prefer rear camera on mobile
        });
        video.srcObject = stream;
        btnCamera.style.display = 'none'; // Hide start button
        btnCapture.disabled = false;
        btnCapture.classList.remove('btn-secondary');
        btnCapture.classList.add('btn-primary');
        errorMessage.classList.add('hidden');
    } catch (err) {
        console.error("Error al acceder a la cámara:", err);
        showError("No se pudo acceder a la cámara. Por favor, asegúrate de dar permisos o usa la opción de subir imagen.");
    }
}

function capturePhoto() {
    if (!stream) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Stop camera to save battery/resources
    stopCamera();

    // Process image as a single item in queue
    // We treat the canvas dataURL as a "virtual" file content
    const dataUrl = canvas.toDataURL('image/png');

    // Reset queue state for single capture
    processingQueue = [dataUrl];
    totalFilesToProcess = 1;
    processedFilesCount = 0;
    allProcessedData = [];

    processQueue();
}

function handleFileUpload(e) {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
        processFiles(files);
    }
}

function processFiles(files) {
    // Sort files by name to ensure correct order
    files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));

    // Read all files as Data URLs first
    const fileReaders = files.map(file => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (event) => resolve(event.target.result);
            reader.onerror = (error) => reject(error);
            reader.readAsDataURL(file);
        });
    });

    Promise.all(fileReaders).then(dataUrls => {
        processingQueue = dataUrls;
        totalFilesToProcess = dataUrls.length;
        processedFilesCount = 0;
        allProcessedData = [];
        processQueue();
    }).catch(err => {
        console.error("Error reading files:", err);
        showError("Error al leer los archivos.");
    });
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

async function processQueue() {
    if (processingQueue.length === 0) {
        // Queue finished!
        finishProcessing();
        return;
    }

    const currentImageSrc = processingQueue.shift();
    processedFilesCount++;

    // Update UI status
    document.getElementById('camera-section').classList.add('hidden');
    statusSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    statusText.innerText = `Procesando imagen ${processedFilesCount} de ${totalFilesToProcess}...`;
    const progressPerc = Math.round(((processedFilesCount - 1) / totalFilesToProcess) * 100);
    progressFill.style.width = `${progressPerc}%`;
    errorMessage.classList.add('hidden');

    try {
        const data = await processSingleImage(currentImageSrc);
        if (data && data.length > 0) {
            allProcessedData.push(...data);
        }

        // Next!
        processQueue();

    } catch (err) {
        console.error(err);
        showError(`Error al procesar la imagen ${processedFilesCount}: ` + err.message);
        // Continue queue even if one fails? Or stop? 
        // Let's continue to try to get as much data as possible.
        processQueue();
    }
}


async function processSingleImage(imageSrc) {
    // Call Local Server
    const result = await processImageWithLocalServer(imageSrc);

    if (result.error) {
        throw new Error(result.error);
    }

    if (!result.text) {
        throw new Error("No text found in response.");
    }

    // Show processed image from server (only showing the last one for now in debug)
    if (result.processed_image) {
        const processedImg = document.getElementById('processed-image-display');
        processedImg.src = result.processed_image;

        const originalImg = document.getElementById('original-image');
        originalImg.src = imageSrc;
    }

    const text = result.text;
    const executionTime = result.execution_time ? result.execution_time.toFixed(3) : "N/A";

    const timeDisplay = document.getElementById('processing-time');
    if (timeDisplay) {
        timeDisplay.textContent = `Tiempo de proceso (último): ${executionTime}s`;
    }

    // For debug raw text, we append
    rawTextOutput.textContent += `\n--- Imagen ${processedFilesCount} [${executionTime}s] ---\n` + text;

    return detectAndParse(text);
}

function detectAndParse(text) {
    // Try both modes
    const dataDistTime = parseTextToData(text, 'dist_time');
    const dataTimeDist = parseTextToData(text, 'time_dist');

    // Heuristic: Choose the one with MORE valid rows
    let selectedData = [];
    let detectedMode = "";

    if (dataTimeDist.length > dataDistTime.length) {
        selectedData = dataTimeDist;
        detectedMode = "Tiempo | Distancia";
    } else {
        selectedData = dataDistTime;
        detectedMode = "Distancia | Tiempo";
    }

    // Update UI
    const orderDisplay = document.getElementById('detected-order');
    if (orderDisplay) {
        orderDisplay.textContent = detectedMode;
    }

    console.log(`Auto-detect: Selected '${detectedMode}' (DT: ${dataDistTime.length}, TD: ${dataTimeDist.length})`);
    return selectedData;
}


function finishProcessing() {
    progressFill.style.width = "100%";
    statusText.innerText = "Finalizado!";

    displayResults(allProcessedData);
}


function displayResults(data) {
    statusSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    // 1. Global Sort by Distance
    data.sort((a, b) => a.distance - b.distance);

    // 2. Global Velocity Calculation
    for (let i = 0; i < data.length; i++) {
        if (i === 0) {
            data[i].velocity = 0;
        } else {
            const curr = data[i];
            const prev = data[i - 1];

            const deltaDist = curr.distance - prev.distance;
            const deltaTime = curr.time - prev.time;

            if (deltaTime > 0.00001) {
                data[i].velocity = deltaDist / deltaTime;
            } else {
                data[i].velocity = 0;
            }
        }
    }

    // Display Parsed Data
    const parsedTextOutput = document.getElementById('parsed-text');
    parsedTextOutput.textContent = JSON.stringify(data, null, 2);

    renderTable(data);
    renderChangesTable(data);
}


function renderChangesTable(data) {
    const changesTableBody = document.querySelector('#changes-table tbody');
    changesTableBody.innerHTML = '';

    if (data.length === 0) return;

    // Filter data: Show the LAST row of each constant speed segment
    const changes = [];
    if (data.length > 0) {
        for (let i = 0; i < data.length; i++) {
            const curr = data[i];

            // If it's the last item, include it
            if (i === data.length - 1) {
                changes.push(curr);
                continue;
            }

            // Dynamic tolerance calculation (0.1s rounding error)
            const timeError = 0.1 / 3600; // 0.1 seconds in hours

            // Helper: Calculate speed uncertainty for an interval
            const getUncertainty = (p1, p2) => {
                if (!p1 || !p2) return 0;
                const dt = p2.time - p1.time;
                const dd = p2.distance - p1.distance;
                if (dt < 0.00001) return 100.0; // High uncertainty if delta T is near zero

                // How much velocity changes if time changes by +0.1s
                const v = p2.velocity;
                const v_alt = dd / (dt + timeError);
                return Math.abs(v - v_alt);
            };

            const prev = (i > 0) ? data[i - 1] : null;
            const next = (i < data.length - 1) ? data[i + 1] : null;

            if (!next) {
                continue;
            }

            // Use MAX uncertainty of current or next segment to be conservative against noise
            const uncCurr = getUncertainty(prev, curr);
            const uncNext = getUncertainty(curr, next);
            const dynamicTol = Math.max(uncCurr, uncNext);

            // Check if change is greater than dynamic tolerance (plus a minimal base baseline)
            if (Math.abs(next.velocity - curr.velocity) > Math.max(dynamicTol, 0.5)) {
                changes.push(curr);
            }
        }
    }

    if (changes.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="3" style="text-align:center;">No hay cambios de velocidad.</td>`;
        changesTableBody.appendChild(tr);
        return;
    }

    let prevDistance = data[0].distance; // Start of first segment

    changes.forEach((item, index) => {
        const tr = document.createElement('tr');
        const timeStr = item.timeDisplay ? item.timeDisplay : item.time.toFixed(4);

        // "Distance From" is the end of the previous segment (or start of table)
        const distFrom = prevDistance;
        // "Distance To" is the current item
        const distTo = item.distance;

        // Update prevDistance for next loop
        prevDistance = item.distance;

        tr.innerHTML = `
            <td>${distFrom.toFixed(3)}</td>
            <td>${distTo.toFixed(3)}</td>
            <td><strong>${item.velocity.toFixed(2)}</strong></td>
            <td>${timeStr}</td>
        `;
        changesTableBody.appendChild(tr);
    });
}

function parseTextToData(text, columnMode = 'dist_time') {
    // 1. Clean and normalize text
    // Replace common OCR errors if needed (e.g. 'O' -> '0')
    // Split into tokens by whitespace
    const tokens = text.replace(/\n/g, ' ').split(/\s+/);

    const data = [];
    let bufferDistance = null;
    let bufferTime = null;

    // Regex definitions
    // Distance: 0,000 or 1,200 or 10.5 (comma or dot decimal)
    const distRegex = /^(\d+)[,.](?:\d{1,3})$/;
    // Time: mm:ss,d or mm:ss.d (e.g. 00:14,4)
    const timeRegex = /^(\d{1,2}):(\d{2})[,.](\d{1,2})$/;

    for (const token of tokens) {
        // Clean token
        const cleanToken = token.trim();
        if (!cleanToken) continue;

        // Check for Time format
        const timeMatch = timeRegex.exec(cleanToken);
        if (timeMatch) {
            // Parse time
            const minutes = parseInt(timeMatch[1], 10);
            const seconds = parseInt(timeMatch[2], 10);
            let decimalPart = timeMatch[3];
            const decimals = parseFloat("0." + decimalPart);
            const totalHours = (minutes / 60) + (seconds / 3600) + (decimals / 3600);

            const timeObj = {
                val: totalHours,
                display: `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')},${decimalPart}`
            };

            if (columnMode === 'dist_time') {
                // Mode: Distance | Time
                // We expect Distance to be buffered
                if (bufferDistance !== null) {
                    data.push({
                        distance: bufferDistance,
                        time: timeObj.val,
                        timeDisplay: timeObj.display,
                        velocity: 0
                    });
                    bufferDistance = null;
                } else {
                    console.warn("Found time without preceding distance (Dist|Time mode):", cleanToken);
                }
            } else {
                // Mode: Time | Distance
                // We store Time and wait for Distance
                bufferTime = timeObj;
            }
            continue;
        }

        // Check for Distance format
        const distMatch = distRegex.exec(cleanToken);
        if (distMatch) {
            const distVal = parseFloat(cleanToken.replace(',', '.'));

            if (columnMode === 'time_dist') {
                // Mode: Time | Distance
                // We expect Time to be buffered
                if (bufferTime !== null) {
                    data.push({
                        distance: distVal,
                        time: bufferTime.val,
                        timeDisplay: bufferTime.display,
                        velocity: 0
                    });
                    bufferTime = null;
                } else {
                    console.warn("Found distance without preceding time (Time|Dist mode):", cleanToken);
                }
            } else {
                // Mode: Distance | Time
                // We store Distance and wait for Time
                bufferDistance = distVal;
            }
            continue;
        }

        // Ignore noise
    }

    // Sort data by distance to handle multi-column blocks correctly (per image)
    data.sort((a, b) => a.distance - b.distance);

    return data;
}

function renderTable(data) {
    resultsTableBody.innerHTML = '';

    if (data.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="3" style="text-align:center;">No se detectaron datos válidos. Intenta mejorar la iluminación o el enfoque.</td>`;
        resultsTableBody.appendChild(tr);
        return;
    }

    data.forEach(item => {
        const tr = document.createElement('tr');
        // Use custom display if available, otherwise fallback to fixed hours
        const timeStr = item.timeDisplay ? item.timeDisplay : item.time.toFixed(4);

        tr.innerHTML = `
            <td>${item.distance.toFixed(2)}</td>
            <td>${timeStr}</td>
            <td><strong>${item.velocity.toFixed(2)}</strong></td>
        `;
        resultsTableBody.appendChild(tr);
    });
}

function resetApp() {
    resultsSection.classList.add('hidden');
    document.getElementById('camera-section').classList.remove('hidden');
    btnCamera.style.display = 'inline-block';

    // Reset video if needed
    video.srcObject = null;
    startCamera(); // Restart camera automatically for convenience
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove('hidden');
}

// Initial setup
// initWorker(); // No longer needed
