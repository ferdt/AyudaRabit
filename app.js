const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const btnCamera = document.getElementById('btn-camera');
const btnCapture = document.getElementById('btn-capture');
const fileUploadStandard = document.getElementById('file-upload-standard');
const fileUploadDebug = document.getElementById('file-upload-debug');
const statusSection = document.getElementById('status-section');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');

const resultsSectionStandard = document.getElementById('results-section-standard');
const resultsSectionDebug = document.getElementById('results-section-debug');

const rawTextOutput = document.getElementById('raw-text');
const errorMessage = document.getElementById('error-message-debug'); // Used in debug mainly
const btnResetStd = document.getElementById('btn-reset-std');
const btnResetDebug = document.getElementById('btn-reset-debug');

let stream = null;

// Queue State
let processingQueue = [];
let allProcessedData = [];
let totalFilesToProcess = 0;
let processedFilesCount = 0;
let isDebugMode = false;

// Local PaddleOCR Server logic
async function processImageWithLocalServer(base64Image) {
    const engineSelect = document.getElementById('ocr-engine');
    const engine = engineSelect ? engineSelect.value : 'rapid';

    // Collect Preprocessing Options
    // Since we moved controls to another page, we rely on server-side config.json
    // or we could implement a way to pass them via URL/Storage if persistence is needed.
    // For now, let's allow server defaults.
    const preprocessingOptions = {};

    const response = await fetch("http://localhost:5000/ocr", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            image: base64Image,
            ocr_engine: engine,
            preprocessing: preprocessingOptions
        })
    });

    if (!response.ok) {
        throw new Error(`Server Error: ${response.statusText}`);
    }

    const result = await response.json();
    return result;
}

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



btnResetStd.addEventListener('click', resetApp);
btnResetDebug.addEventListener('click', resetApp);

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
    resultsSectionStandard.classList.add('hidden');
    resultsSectionDebug.classList.add('hidden');

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

    const timeDisplayStd = document.getElementById('processing-time-std');
    if (timeDisplayStd) timeDisplayStd.textContent = `Tiempo de proceso: ${executionTime}s`;

    const timeDisplayDebug = document.getElementById('processing-time-debug');
    if (timeDisplayDebug) timeDisplayDebug.textContent = `Tiempo de proceso: ${executionTime}s`;

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

    const orderDisplayDebug = document.getElementById('detected-order-debug');
    if (orderDisplayDebug) {
        orderDisplayDebug.textContent = detectedMode;
    }

    const orderDisplayStd = document.getElementById('detected-order-std');
    if (orderDisplayStd) {
        orderDisplayStd.textContent = detectedMode;
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

    if (isDebugMode) {
        resultsSectionDebug.classList.remove('hidden');
        resultsSectionStandard.classList.add('hidden');
    } else {
        resultsSectionStandard.classList.remove('hidden');
        resultsSectionDebug.classList.add('hidden');
    }

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

            // Store delta time for display (convert to seconds for readability if needed, or keep as hours/min)
            // Let's store nicely formatted string
            data[i].deltaTimeVal = deltaTime;

            if (deltaTime > 0.00001) {
                data[i].velocity = deltaDist / deltaTime;
            } else {
                data[i].velocity = 0;
            }
        }
    }

    // Display Parsed Data (only in debug)
    const parsedTextOutput = document.getElementById('parsed-text');
    if (parsedTextOutput) parsedTextOutput.textContent = JSON.stringify(data, null, 2);

    // Render Tables
    // Debug view gets full table
    if (isDebugMode) {
        renderTable(data, 'results-table-debug');
        renderChangesTable(data, 'changes-table-debug');
    } else {
        // Standard view gets changes table
        renderChangesTable(data, 'changes-table-std');
    }
}


function renderChangesTable(data, tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;
    const tbody = table.querySelector('tbody');
    tbody.innerHTML = '';

    if (data.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="4" style="text-align:center;">No se detectaron datos válidos.</td>`;
        tbody.appendChild(tr);
        return;
    }

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

                // How much velocity changes if time changes by +/- 0.1s
                const v = p2.velocity;
                const v_high = dd / (dt + timeError);

                let v_low = v;
                if (dt > timeError) {
                    v_low = dd / (dt - timeError);
                } else {
                    // If interval is smaller than error, velocity is extremely uncertain
                    v_low = 10000; // Large number
                }

                const diff1 = Math.abs(v - v_high);
                const diff2 = Math.abs(v - v_low);

                return Math.max(diff1, diff2);
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
            if (Math.abs(next.velocity - curr.velocity) > dynamicTol) {
                changes.push(curr);
            }
        }
    }

    if (changes.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="4" style="text-align:center;">No hay cambios de velocidad significativos.</td>`;
        tbody.appendChild(tr);
        return;
    }

    let prevDistance = data[0].distance; // Start of first segment
    let prevTime = data[0].time;         // Start of first segment time

    changes.forEach((item, index) => {
        const tr = document.createElement('tr');
        const timeStr = item.timeDisplay ? item.timeDisplay : item.time.toFixed(4);

        // "Distance From" is the end of the previous segment (or start of table)
        const distFrom = prevDistance;
        // "Distance To" is the current item
        const distTo = item.distance;

        // Calculate Average Velocity for this segment
        const timeFrom = prevTime;
        const timeTo = item.time;

        let intervalVelocity = 0;
        const deltaDist = distTo - distFrom;
        const deltaTime = timeTo - timeFrom;

        if (deltaTime > 0.00001) {
            intervalVelocity = deltaDist / deltaTime;
        }

        // Update prevs for next loop
        prevDistance = item.distance;
        prevTime = item.time;

        tr.innerHTML = `
            <td>${distFrom.toFixed(3)}</td>
            <td>${distTo.toFixed(3)}</td>
            <td><strong>${intervalVelocity.toFixed(2)}</strong></td>
            <td>${timeStr}</td>
        `;
        tbody.appendChild(tr);
    });
}

function parseTextToData(text, columnMode) {
    // 1. Clean and normalize text
    // Replace common OCR errors if needed (e.g. 'O' -> '0')
    // Split into tokens by whitespace
    const tokens = text.replace(/\n/g, ' ').split(/\s+/);

    const data = [];
    let bufferDistance = null;
    let bufferTime = null;

    // Regex definitions
    // Distance: 0,000 or 1,200 or 10.5 (comma or dot decimal) OR pure integer
    // User Request: "Any value greater or equal of 100 km divide by 1000" handled in post-processing
    const distRegex = /^(\d+)(?:[,.]\d{1,3})?$/;
    // Time: mm:ss,d or mm:ss.d (e.g. 00:14,4)
    const timeRegex = /^(\d{1,2}):(\d{2})[,.](\d{1,2})$/;

    // Debugging: Log the first few decisions to see what's happening
    let logCount = 0;
    const maxLogs = 20;

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
                    if (logCount < maxLogs) console.warn(`[Parse] Found TIME '${cleanToken}' without DISTANCE buffer.`);
                }
            } else {
                // Mode: Time | Distance
                // We store Time and wait for Distance
                bufferTime = timeObj;
            }
            continue;
        }

        // Check for Distance format (Numeric)
        const distMatch = distRegex.exec(cleanToken);
        if (distMatch) {
            if (logCount < maxLogs) {
                console.log(`[Parse] Token '${cleanToken}' -> Numeric match. Mode: ${columnMode}`);
                logCount++;
            }

            const numericVal = parseFloat(cleanToken.replace(',', '.'));

            // Context-Aware Parsing
            if (columnMode === 'time_dist') {
                // Mode: Time | Distance
                // We expect Time FIRST
                if (bufferTime === null) {
                    // Treat as Time (Decimal or Unit-less)
                    // Start with Hours assumption since Velocity is usually km/h
                    const timeObj = {
                        val: numericVal,
                        display: cleanToken // Keep original string
                    };
                    bufferTime = timeObj;
                    if (logCount < maxLogs) console.log(`[Parse] Treated '${cleanToken}' as TIME (bufferTime set).`);
                } else {
                    // We have Time, so this MUST be Distance
                    data.push({
                        distance: numericVal,
                        time: bufferTime.val,
                        timeDisplay: bufferTime.display,
                        velocity: 0
                    });
                    bufferTime = null;
                    if (logCount < maxLogs) console.log(`[Parse] Treated '${cleanToken}' as DISTANCE. Row created.`);
                }
            } else {
                // Mode: Distance | Time
                // We expect Distance FIRST
                if (bufferDistance === null) {
                    // Treat as Distance
                    bufferDistance = numericVal;
                    if (logCount < maxLogs) console.log(`[Parse] Treated '${cleanToken}' as DISTANCE (bufferDistance set).`);
                } else {
                    // We have Distance, so this MUST be Time
                    const timeObj = {
                        val: numericVal,
                        display: cleanToken
                    };
                    data.push({
                        distance: bufferDistance,
                        time: timeObj.val,
                        timeDisplay: timeObj.display,
                        velocity: 0
                    });
                    bufferDistance = null;
                    if (logCount < maxLogs) console.log(`[Parse] Treated '${cleanToken}' as TIME. Row created.`);
                }
            }
            continue;
        }

        // Ignore noise
    }

    // --- Post-Processing: Normalize Distances ---
    // Heuristic: "Steps should be around 100 meters (0.1 km)"
    // 1. Calculate Median Distance
    if (data.length > 0) {
        const distances = data.map(d => d.distance);
        distances.sort((a, b) => a - b);
        const medianDist = distances[Math.floor(distances.length / 2)];

        let globalScale = 1.0;

        // If Median > 500, assume the whole table is in Meters (e.g. 500m, 1000m, 12000m)
        // Convert to KM
        if (medianDist > 500) {
            globalScale = 0.001;
            console.log(`[Normalization] Median distance ${medianDist} > 500. Assuming METERS. Scaling by 0.001.`);
        } else {
            console.log(`[Normalization] Median distance ${medianDist} <= 500. Assuming KM.`);
        }

        // Apply Scale and Fix Outliers
        data.forEach(d => {
            let original = d.distance;

            // Base conversion
            d.distance = original * globalScale;

            // 2. Fix Individual Outliers (e.g. missing decimal point in OCR: 4775 instead of 4.775)
            // User Request: Any value >= 100 should be divided by 1000 (likely meters or missing dot)
            if (globalScale === 1.0 && d.distance >= 100) {
                d.distance /= 1000;
                console.log(`[Normalization] Outlier detected: ${original} -> ${d.distance} (>= 100 rule)`);
            }
        });
    }

    // Sort data by distance to handle multi-column blocks correctly (per image)
    data.sort((a, b) => a.distance - b.distance);

    return data;
}

function renderTable(data, tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;
    const tbody = table.querySelector('tbody');
    tbody.innerHTML = '';

    if (data.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="3" style="text-align:center;">No se detectaron datos válidos. Intenta mejorar la iluminación o el enfoque.</td>`;
        tbody.appendChild(tr);
        return;
    }

    data.forEach((item, index) => {
        // Feature: Ignore first line if distance is 0
        if (index === 0 && item.distance === 0) {
            return;
        }

        const tr = document.createElement('tr');
        // Use custom display if available, otherwise fallback to fixed hours
        const timeStr = item.timeDisplay ? item.timeDisplay : item.time.toFixed(4);

        // Format Delta Time
        let diffStr = "-";
        if (item.deltaTimeVal !== undefined) {
            // Convert to total seconds, round to 1 decimal place
            const totalSecondsRounded = Math.round((item.deltaTimeVal * 3600) * 10) / 10;

            const mm = Math.floor(totalSecondsRounded / 60);
            const ss = (totalSecondsRounded % 60); // Float remainder but rounded to 1 decimal precision naturally

            const mmStr = mm.toString().padStart(2, '0');
            const ssStr = ss.toFixed(1).padStart(4, '0'); // e.g. "04.2" or "15.0"

            diffStr = `+${mmStr}:${ssStr}`;
        }

        tr.innerHTML = `
            <td>${item.distance.toFixed(2)}</td>
            <td>${timeStr}</td>
            <td>${diffStr}</td>
            <td><strong>${item.velocity.toFixed(2)}</strong></td>
        `;
        tbody.appendChild(tr);
    });
}

function resetApp() {
    resultsSectionStandard.classList.add('hidden');
    resultsSectionDebug.classList.add('hidden');
    document.getElementById('camera-section').classList.remove('hidden');
    btnCamera.style.display = 'inline-block';

    // Reset video if needed
    video.srcObject = null;
    startCamera(); // Restart camera automatically for convenience
}

function showError(msg) {
    if (isDebugMode) {
        const errDebug = document.getElementById('error-message-debug');
        if (errDebug) {
            errDebug.textContent = msg;
            errDebug.classList.remove('hidden');
            resultsSectionDebug.classList.remove('hidden'); // Ensure section is visible to see error
        }
    } else {
        alert("Error: " + msg); // Fallback for standard mode for now
    }
}

// Initial setup
// initWorker(); // No longer needed
