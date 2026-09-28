/**
 * AI Vision - Browser-Based Object Detection
 * Redesigned with the Wise Design System & Bootstrap Icons
 * 
 * Strict Vanilla JavaScript (ES6+), Zero Build Process, Zero Server Dependencies.
 * Deployable statically to GitHub Pages.
 */

'use strict';

(function () {
  // =========================================================================
  // Application State
  // =========================================================================
  const state = {
    currentFile: null,
    currentImage: null,
    confidenceThreshold: 0.50,
    model: null,
    isModelLoading: false,
    isModelReady: false,
    isDetecting: false,
    hasRunInference: false,
    rawDetections: [] // Real predictions from model.detect()
  };

  // Supported MIME types and maximum file size (15 MB)
  const SUPPORTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024;

  // Wise Design System Color Palette for Object Detections
  const PALETTE = [
    '#9fe870', // Wise Green (Primary)
    '#2ead4b', // Positive Green
    '#38c8ff', // Accent Cyan
    '#ffc091', // Accent Orange
    '#ffd11a', // Warning Yellow
    '#d03238', // Negative Red
    '#a855f7', // Purple
    '#f43f5e', // Rose
    '#10b981', // Emerald
    '#f97316', // Bright Orange
    '#818cf8', // Indigo
    '#06b6d4'  // Cyan
  ];

  // =========================================================================
  // DOM Elements
  // =========================================================================
  const elements = {
    // Status & Alerts
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    alertBox: document.getElementById('alert-box'),
    alertMessage: document.getElementById('alert-message'),
    alertClose: document.getElementById('alert-close'),

    // Upload & Canvas Area
    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    canvasContainer: document.getElementById('canvas-container'),
    canvas: document.getElementById('detection-canvas'),
    imageDimensions: document.getElementById('image-dimensions'),
    imageFilesize: document.getElementById('image-filesize'),
    imageFilename: document.getElementById('image-filename'),

    // Controls & Action Buttons
    thresholdSlider: document.getElementById('threshold-slider'),
    thresholdVal: document.getElementById('threshold-val'),
    btnDetect: document.getElementById('btn-detect'),
    btnClear: document.getElementById('btn-clear'),
    btnDownload: document.getElementById('btn-download'),

    // Summary & Details
    statTotal: document.getElementById('stat-total'),
    statTypes: document.getElementById('stat-types'),
    statThreshold: document.getElementById('stat-threshold'),
    breakdownList: document.getElementById('breakdown-list'),
    detailsList: document.getElementById('details-list'),
    detailsCount: document.getElementById('details-count')
  };

  const canvasContext = elements.canvas.getContext('2d');

  // =========================================================================
  // UI & Alert Helpers
  // =========================================================================

  /**
   * Display an alert notification to the user
   * @param {string} message 
   * @param {'error'|'info'|'success'} type 
   */
  function showAlert(message, type = 'error') {
    elements.alertMessage.textContent = message;
    elements.alertBox.className = `alert-box alert-${type}`;
    elements.alertBox.classList.remove('alert-hidden');
  }

  /**
   * Hide the alert notification
   */
  function hideAlert() {
    elements.alertBox.classList.add('alert-hidden');
  }

  /**
   * Update the status badge text and visual appearance
   * @param {'ready'|'busy'|'error'} mode 
   * @param {string} text 
   */
  function updateStatus(mode, text) {
    elements.statusDot.className = `status-indicator status-${mode}`;
    elements.statusText.textContent = text;
  }

  /**
   * Format file size in human-readable units (KB / MB)
   * @param {number} bytes 
   * @returns {string}
   */
  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  /**
   * Capitalize first letter of string
   * @param {string} str 
   * @returns {string}
   */
  function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  /**
   * Convert hex color string to rgba string with transparency
   * @param {string} hex 
   * @param {number} alpha 
   * @returns {string}
   */
  function hexToRgba(hex, alpha) {
    let cleanHex = hex.replace('#', '');
    if (cleanHex.length === 3) {
      cleanHex = cleanHex.split('').map(c => c + c).join('');
    }
    const num = parseInt(cleanHex, 16);
    const r = (num >> 16) & 255;
    const g = (num >> 8) & 255;
    const b = num & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  /**
   * Calculate contrast text color (near-black #0e0f0c vs white #ffffff) based on color luminance
   * @param {string} hex 
   * @returns {string}
   */
  function getContrastTextColor(hex) {
    let cleanHex = hex.replace('#', '');
    if (cleanHex.length === 3) {
      cleanHex = cleanHex.split('').map(c => c + c).join('');
    }
    const num = parseInt(cleanHex, 16);
    const r = (num >> 16) & 255;
    const g = (num >> 8) & 255;
    const b = num & 255;
    // Relative luminance formula
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return luminance > 0.55 ? '#0e0f0c' : '#ffffff';
  }

  /**
   * Deterministically assign a distinct color to an object class name
   * @param {string} className 
   * @returns {string} Hex color
   */
  function getColorForClass(className) {
    let hash = 0;
    for (let i = 0; i < className.length; i++) {
      hash = className.charCodeAt(i) + ((hash << 5) - hash);
    }
    const index = Math.abs(hash) % PALETTE.length;
    return PALETTE[index];
  }

  // =========================================================================
  // Model Loading (TensorFlow.js + COCO-SSD)
  // =========================================================================

  /**
   * Initialize pretrained COCO-SSD model in the browser
   */
  async function initializeModel() {
    state.isModelLoading = true;
    updateStatus('busy', 'Loading model...');

    // Verify CDN scripts loaded
    if (typeof cocoSsd === 'undefined') {
      state.isModelLoading = false;
      updateStatus('error', '⚠ Model failed to load');
      showAlert('Failed to connect to the AI model library. Please check your internet connection and refresh.', 'error');
      return;
    }

    try {
      // Load COCO-SSD with lightweight MobileNet backend for fast browser inference
      const loadedModel = await cocoSsd.load({
        base: 'lite_mobilenet_v2'
      });

      state.model = loadedModel;
      state.isModelReady = true;
      state.isModelLoading = false;
      updateStatus('ready', '✓ AI model ready');

      // If user uploaded an image while model was downloading, enable detection button now
      if (state.currentImage) {
        elements.btnDetect.disabled = false;
      }
    } catch (error) {
      state.isModelLoading = false;
      state.isModelReady = false;
      console.error('Model initialization error:', error);
      updateStatus('error', '⚠ Model failed to load');
      showAlert('Could not load pretrained AI model. Please ensure modern browser compatibility and internet access.', 'error');
    }
  }

  // =========================================================================
  // File Validation & Handling
  // =========================================================================

  /**
   * Validate uploaded file type and size
   * @param {File} file 
   * @returns {boolean}
   */
  function validateFile(file) {
    if (!file) {
      showAlert('Please select an image file.');
      return false;
    }

    const extension = file.name.split('.').pop().toLowerCase();
    const isValidType = SUPPORTED_TYPES.includes(file.type) || 
      ['jpg', 'jpeg', 'png', 'webp'].includes(extension);

    if (!isValidType) {
      showAlert('Please upload a JPG, PNG or WEBP image.');
      return false;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      showAlert('The selected image exceeds 15MB. Please choose a smaller image.');
      return false;
    }

    hideAlert();
    return true;
  }

  /**
   * Handle the selected image file
   * @param {File} file 
   */
  function handleImageFile(file) {
    if (!validateFile(file)) return;

    state.currentFile = file;
    state.hasRunInference = false;
    state.rawDetections = [];

    const reader = new FileReader();
    reader.onload = function (event) {
      const img = new Image();
      img.onload = function () {
        state.currentImage = img;
        renderImageToCanvas(img);
        updateMetadata(file, img);
        resetSummaryViews();

        // Enable buttons based on model readiness
        elements.btnClear.disabled = false;
        elements.btnDownload.disabled = true;

        if (state.isModelReady) {
          elements.btnDetect.disabled = false;
        } else if (state.isModelLoading) {
          elements.btnDetect.disabled = true;
          showAlert('AI model is still loading. Please wait a moment.', 'info');
        }
      };
      img.onerror = function () {
        showAlert('Failed to read image file. Please try another image.');
      };
      img.src = event.target.result;
    };
    reader.onerror = function () {
      showAlert('Error reading file from storage.');
    };
    reader.readAsDataURL(file);
  }

  /**
   * Render image to canvas preserving original aspect ratio and dimensions
   * @param {HTMLImageElement} img 
   */
  function renderImageToCanvas(img) {
    elements.canvas.width = img.naturalWidth;
    elements.canvas.height = img.naturalHeight;

    canvasContext.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    canvasContext.drawImage(img, 0, 0, elements.canvas.width, elements.canvas.height);

    elements.uploadArea.style.display = 'none';
    elements.canvasContainer.classList.remove('canvas-hidden');
  }

  /**
   * Update image dimensions and filename meta tags
   * @param {File} file 
   * @param {HTMLImageElement} img 
   */
  function updateMetadata(file, img) {
    elements.imageDimensions.textContent = `${img.naturalWidth} \u00D7 ${img.naturalHeight} px`;
    elements.imageFilesize.textContent = formatBytes(file.size);
    elements.imageFilename.textContent = file.name;
    elements.imageFilename.title = file.name;
  }

  // =========================================================================
  // Object Detection Inference
  // =========================================================================

  /**
   * Execute real client-side object detection inference using the loaded model
   */
  async function runDetection() {
    if (!state.isModelReady) {
      showAlert('AI model is still loading. Please wait.', 'info');
      return;
    }

    if (!state.currentImage) {
      showAlert('Please upload an image before running detection.', 'error');
      return;
    }

    if (state.isDetecting) return;

    state.isDetecting = true;
    updateStatus('busy', 'Running detection...');
    hideAlert();

    // Disable action buttons and show spin animation using Bootstrap Icons
    elements.btnDetect.disabled = true;
    elements.btnDetect.innerHTML = `
      <i class="bi bi-arrow-repeat spin btn-icon"></i>
      <span>Detecting...</span>
    `;
    elements.btnClear.disabled = true;
    elements.btnDownload.disabled = true;

    // Yield control to browser thread to repaint UI status before heavy inference begins
    setTimeout(async () => {
      try {
        // Run REAL browser inference via COCO-SSD
        // minScore 0.05 allows client-side slider filtering down to 10% without re-running the model
        const predictions = await state.model.detect(elements.canvas, 50, 0.05);

        state.rawDetections = predictions;
        state.hasRunInference = true;
        updateStatus('ready', '✓ Detection complete');

        // Apply confidence threshold and visualize results
        applyDetectionFiltering();

        // Enable download of annotated canvas
        elements.btnDownload.disabled = false;
      } catch (err) {
        console.error('Detection inference failed:', err);
        updateStatus('error', '⚠ Object detection failed');
        showAlert('Object detection failed. Please try another image.', 'error');
      } finally {
        state.isDetecting = false;
        elements.btnDetect.disabled = false;
        elements.btnClear.disabled = false;
        elements.btnDetect.innerHTML = `
          <i class="bi bi-play-fill btn-icon"></i>
          <span>Detect Objects</span>
        `;
      }
    }, 50);
  }

  // =========================================================================
  // Bounding Box Visualization & Confidence Filtering
  // =========================================================================

  /**
   * Filter detections by confidence threshold and update canvas & summary panels
   * Avoids rerunning the model when only the threshold changes.
   */
  function applyDetectionFiltering() {
    if (!state.currentImage) return;

    // Reset canvas to original image before drawing active bounding boxes
    canvasContext.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    canvasContext.drawImage(state.currentImage, 0, 0, elements.canvas.width, elements.canvas.height);

    if (!state.hasRunInference) return;

    // Filter raw detections according to current threshold
    const filtered = state.rawDetections.filter(d => d.score >= state.confidenceThreshold);

    // Render bounding boxes & labels
    drawAnnotations(filtered);

    // Update statistical counters & breakdown panels
    updateSummaryViews(filtered);
  }

  /**
   * Draw bounding boxes, labels, and confidence percentages on the HTML canvas
   * Follows the Wise visual softness with pill labels and clean typography.
   * @param {Array} detections 
   */
  function drawAnnotations(detections) {
    if (!detections || detections.length === 0) return;

    const canvasWidth = elements.canvas.width;
    const canvasHeight = elements.canvas.height;

    // Dynamic line width and font size proportional to image resolution
    const lineWidth = Math.max(3, Math.round(canvasWidth / 300));
    const fontSize = Math.max(13, Math.min(24, Math.round(canvasWidth / 42)));

    detections.forEach(detection => {
      const [rawX, rawY, rawW, rawH] = detection.bbox;
      const x = Math.max(0, Math.round(rawX));
      const y = Math.max(0, Math.round(rawY));
      const width = Math.min(canvasWidth - x, Math.round(rawW));
      const height = Math.min(canvasHeight - y, Math.round(rawH));

      const className = detection.class;
      const confidence = detection.score;
      const classColor = getColorForClass(className);
      const textColor = getContrastTextColor(classColor);

      // 1. Draw Bounding Box Fill & Stroke
      canvasContext.lineWidth = lineWidth;
      canvasContext.strokeStyle = classColor;
      canvasContext.fillStyle = hexToRgba(classColor, 0.14);

      canvasContext.fillRect(x, y, width, height);
      canvasContext.strokeRect(x, y, width, height);

      // 2. Draw Readable Pill Label Badge
      const labelText = `${capitalize(className)} ${Math.round(confidence * 100)}%`;
      canvasContext.font = `700 ${fontSize}px "Inter", -apple-system, sans-serif`;

      const textMetrics = canvasContext.measureText(labelText);
      const paddingH = Math.round(fontSize * 0.55);
      const paddingV = Math.round(fontSize * 0.35);
      const badgeWidth = textMetrics.width + paddingH * 2;
      const badgeHeight = fontSize + paddingV * 2;
      const cornerRadius = Math.min(8, Math.round(badgeHeight / 2));

      // Position label above box if possible; otherwise place inside top edge
      let badgeY = y - badgeHeight - 2;
      if (badgeY < 0) {
        badgeY = y + 2;
      }

      // Constrain horizontally inside canvas boundaries
      let badgeX = x;
      if (badgeX + badgeWidth > canvasWidth) {
        badgeX = canvasWidth - badgeWidth - 2;
      }

      // Draw rounded label badge
      canvasContext.fillStyle = classColor;
      if (typeof canvasContext.roundRect === 'function') {
        canvasContext.beginPath();
        canvasContext.roundRect(badgeX, badgeY, badgeWidth, badgeHeight, cornerRadius);
        canvasContext.fill();
      } else {
        canvasContext.fillRect(badgeX, badgeY, badgeWidth, badgeHeight);
      }

      // Draw high-contrast label text
      canvasContext.fillStyle = textColor;
      canvasContext.textBaseline = 'top';
      canvasContext.fillText(labelText, badgeX + paddingH, badgeY + paddingV);
    });
  }

  // =========================================================================
  // Detection Summary & Details Panels
  // =========================================================================

  /**
   * Update Summary metrics, class breakdown, and details accordion
   * @param {Array} detections 
   */
  function updateSummaryViews(detections) {
    const totalCount = detections.length;
    const thresholdPct = Math.round(state.confidenceThreshold * 100) + '%';

    // Count detections by class
    const countsByClass = {};
    detections.forEach(d => {
      const cls = capitalize(d.class);
      countsByClass[cls] = (countsByClass[cls] || 0) + 1;
    });

    const uniqueClasses = Object.keys(countsByClass);

    // Update Metric Cards
    elements.statTotal.textContent = totalCount;
    elements.statTypes.textContent = uniqueClasses.length;
    elements.statThreshold.textContent = thresholdPct;
    elements.detailsCount.textContent = `${totalCount} item${totalCount === 1 ? '' : 's'}`;

    // Update Breakdown List (Sorted by count descending)
    if (totalCount === 0) {
      elements.breakdownList.innerHTML = `
        <div class="empty-state-notice">
          <i class="bi bi-info-circle"></i>
          <span>No objects meet the ${thresholdPct} confidence threshold. Try lowering the slider.</span>
        </div>
      `;
    } else {
      const sortedClasses = Object.entries(countsByClass)
        .sort((a, b) => b[1] - a[1]);

      elements.breakdownList.innerHTML = sortedClasses.map(([clsName, count]) => {
        const color = getColorForClass(clsName.toLowerCase());
        return `
          <div class="breakdown-row">
            <span class="breakdown-class">
              <span class="breakdown-color-tag" style="background-color: ${color};"></span>
              ${clsName}
            </span>
            <span class="breakdown-count">${count}</span>
          </div>
        `;
      }).join('');
    }

    // Update Detection Details List
    if (totalCount === 0) {
      elements.detailsList.innerHTML = `
        <div class="empty-state-notice">
          <i class="bi bi-bounding-box"></i>
          <span>No detections to display at the current threshold.</span>
        </div>
      `;
    } else {
      elements.detailsList.innerHTML = detections.map(d => {
        const clsName = capitalize(d.class);
        const scorePct = Math.round(d.score * 100) + '%';
        const [x, y, w, h] = d.bbox.map(n => Math.round(n));

        return `
          <div class="detail-item">
            <div class="detail-top">
              <span class="detail-name">${clsName}</span>
              <span class="detail-confidence">${scorePct}</span>
            </div>
            <div class="detail-coords">
              <span>x: ${x}</span>
              <span>y: ${y}</span>
              <span>width: ${w}</span>
              <span>height: ${h}</span>
            </div>
          </div>
        `;
      }).join('');
    }
  }

  /**
   * Reset summary and details to placeholder states
   */
  function resetSummaryViews() {
    const thresholdPct = Math.round(state.confidenceThreshold * 100) + '%';
    elements.statTotal.textContent = '0';
    elements.statTypes.textContent = '0';
    elements.statThreshold.textContent = thresholdPct;
    elements.detailsCount.textContent = '0 items';

    elements.breakdownList.innerHTML = `
      <div class="empty-state-notice">
        <i class="bi bi-info-circle"></i>
        <span>No objects detected yet. Click "Detect Objects" to analyze.</span>
      </div>
    `;

    elements.detailsList.innerHTML = `
      <div class="empty-state-notice">
        <i class="bi bi-bounding-box"></i>
        <span>Detailed bounding box coordinates and scores will appear after running detection.</span>
      </div>
    `;
  }

  // =========================================================================
  // Download Result
  // =========================================================================

  /**
   * Export the annotated canvas as a high-resolution PNG image directly in browser
   */
  function downloadAnnotatedImage() {
    if (!state.currentImage || !state.hasRunInference) {
      showAlert('Please detect objects before downloading results.', 'info');
      return;
    }

    try {
      const dataUrl = elements.canvas.toDataURL('image/png');
      const downloadAnchor = document.createElement('a');
      downloadAnchor.href = dataUrl;
      downloadAnchor.download = 'ai-vision-result.png';
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      document.body.removeChild(downloadAnchor);
    } catch (err) {
      console.error('Download export failed:', err);
      showAlert('Could not export annotated image. Please try again.', 'error');
    }
  }

  // =========================================================================
  // Reset / Clear Function
  // =========================================================================

  /**
   * Completely reset application state back to initial upload screen
   */
  function clearApplicationState() {
    state.currentFile = null;
    state.currentImage = null;
    state.rawDetections = [];
    state.hasRunInference = false;
    state.confidenceThreshold = 0.50;

    // Reset controls
    elements.thresholdSlider.value = 0.50;
    elements.thresholdVal.textContent = '50%';
    elements.statThreshold.textContent = '50%';

    // Reset buttons
    elements.btnDetect.disabled = true;
    elements.btnClear.disabled = true;
    elements.btnDownload.disabled = true;

    // Reset canvas & upload views
    elements.fileInput.value = '';
    canvasContext.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    elements.canvasContainer.classList.add('canvas-hidden');
    elements.uploadArea.style.display = 'flex';

    // Reset summary & details
    elements.statTotal.textContent = '0';
    elements.statTypes.textContent = '0';
    elements.detailsCount.textContent = '0 items';

    elements.breakdownList.innerHTML = `
      <div class="empty-state-notice">
        <i class="bi bi-info-circle"></i>
        <span>No objects detected yet. Upload an image to preview.</span>
      </div>
    `;

    elements.detailsList.innerHTML = `
      <div class="empty-state-notice">
        <i class="bi bi-bounding-box"></i>
        <span>Detailed bounding box coordinates and scores will appear after running detection.</span>
      </div>
    `;

    hideAlert();
    if (state.isModelReady) {
      updateStatus('ready', '✓ AI model ready');
    }
  }

  // =========================================================================
  // Event Listeners
  // =========================================================================

  function initializeEventListeners() {
    // Alert close button
    elements.alertClose.addEventListener('click', hideAlert);

    // Upload area click & keyboard activation
    elements.uploadArea.addEventListener('click', function () {
      elements.fileInput.click();
    });

    elements.uploadArea.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        elements.fileInput.click();
      }
    });

    // File input change
    elements.fileInput.addEventListener('change', function (e) {
      if (e.target.files && e.target.files.length > 0) {
        handleImageFile(e.target.files[0]);
      }
    });

    // Drag-and-drop interactions
    ['dragenter', 'dragover'].forEach(eventName => {
      elements.uploadArea.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        elements.uploadArea.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      elements.uploadArea.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        elements.uploadArea.classList.remove('dragover');
      });
    });

    elements.uploadArea.addEventListener('drop', function (e) {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        handleImageFile(dt.files[0]);
      }
    });

    // Confidence threshold slider
    elements.thresholdSlider.addEventListener('input', function (e) {
      const value = parseFloat(e.target.value);
      state.confidenceThreshold = value;
      const percentage = Math.round(value * 100) + '%';
      elements.thresholdVal.textContent = percentage;
      elements.statThreshold.textContent = percentage;

      // Filter existing detections instantaneously without re-running model inference
      if (state.hasRunInference) {
        applyDetectionFiltering();
      }
    });

    // Action buttons
    elements.btnDetect.addEventListener('click', runDetection);
    elements.btnClear.addEventListener('click', clearApplicationState);
    elements.btnDownload.addEventListener('click', downloadAnnotatedImage);
  }

  // =========================================================================
  // Initialization
  // =========================================================================
  function init() {
    initializeEventListeners();
    initializeModel();
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
