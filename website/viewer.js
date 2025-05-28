
function formatRaHms(raDeg) {
  const hours = raDeg / 15;
  const h = Math.floor(hours);
  const m = Math.floor((hours - h) * 60);
  const s = (((hours - h) * 60 - m) * 60).toFixed(1);
  return `${h}h ${m}m ${s}s`;
}

function formatDecDms(decDeg) {
  const sign = decDeg >= 0 ? '+' : '−';
  const abs = Math.abs(decDeg);
  const d = Math.floor(abs);
  const m = Math.floor((abs - d) * 60);
  const s = (((abs - d) * 60 - m) * 60).toFixed(1);
  return `${sign}${d}° ${m}' ${s}"`;
}
function waitForAladin(timeout = 5000) {
    return new Promise((resolve, reject) => {
        const start = Date.now();
        function check() {
            if (window.A && typeof A.aladin === 'function') {
                resolve(A);
            } else if (Date.now() - start > timeout) {
                reject(new Error('Aladin Lite library not loaded'));
            } else {
                setTimeout(check, 50);
            }
        }
        check();
    });
}

/**
 * Enhanced PulseHunter Detection Viewer
 * Optimized for performance and user experience
 */

class PulseHunterViewer {
formatRaHms(raDeg) {
        const ra = raDeg / 15;
        const h = Math.floor(ra);
        const m = Math.floor((ra - h) * 60);
        const s = (((ra - h) * 60 - m) * 60).toFixed(1);
        return `${h}h ${m}m ${s}s`;
    }

    formatDecDms(decDeg) {
        const sign = decDeg >= 0 ? '+' : '−';
        const abs = Math.abs(decDeg);
        const d = Math.floor(abs);
        const m = Math.floor((abs - d) * 60);
        const s = (((abs - d) * 60 - m) * 60).toFixed(1);
        return `${sign}${d}° ${m}' ${s}"`;
    }
    constructor() {
        this.cache = new Map();
        this.detections = [];
        this.aladin = null;
        this.catalogs = new Map();
        this.loadingState = false;
        this.retryCount = 0;
        this.maxRetries = 3;

        // Performance monitoring
        this.performanceMetrics = {
            loadStart: 0,
            loadEnd: 0,
            renderStart: 0,
            renderEnd: 0
        };

        this.init();
    }

    async init() {
        console.log('🚀 Initializing PulseHunter Viewer...');
        this.performanceMetrics.loadStart = performance.now();

        try {
            this.showLoadingState(true);
            await this.initializeAladin();
            await this.loadAllDetections();
            this.renderVisualization();
            this.populateTable();
            this.updateStatistics();
            this.setupEventListeners();

            this.performanceMetrics.loadEnd = performance.now();
            console.log(`✅ Viewer initialized in ${(this.performanceMetrics.loadEnd - this.performanceMetrics.loadStart).toFixed(2)}ms`);

        } catch (error) {
            console.error('❌ Failed to initialize viewer:', error);
            this.handleError(error);
        } finally {
            this.showLoadingState(false);
        }
    }

    showLoadingState(show) {
        const loadingElements = document.querySelectorAll('.loading, .spinner');
        const contentElements = document.querySelectorAll('#aladin-lite-div, #detectionTable');

        loadingElements.forEach(el => {
            el.style.display = show ? 'flex' : 'none';
        });

        contentElements.forEach(el => {
            el.style.opacity = show ? '0.3' : '1';
            el.style.pointerEvents = show ? 'none' : 'auto';
        });

        this.loadingState = show;
    }

    async initializeAladin() {
        return new Promise((resolve, reject) => {
            try {
                if (typeof A === 'undefined') {
                    throw new Error('Aladin Lite library not loaded');
                }

                this.aladin = A.aladin('#aladin-lite-div', {
                    survey: 'P/DSS2/color',
                    fov: 180,
                    target: '12 00 +41 00',
                    showReticle: false,
                    showZoomControl: true,
                    showFullscreenControl: true,
                    showLayersControl: true,
                    showGotoControl: true,
                    showShareControl: false,
                    showCatalogControl: true,
                    showFrame: true,
                    showCooFrame: true
                });

                // Wait for Aladin to be ready
                setTimeout(() => {
                    if (this.aladin) {
                        console.log('✅ Aladin initialized successfully');
                        logDebug('Aladin initialized ✔');
                        resolve();
                    } else {
                        reject(new Error('Aladin failed to initialize'));
                    }
                }, 1000);

            } catch (error) {
                console.error('❌ Aladin initialization failed:', error);
                reject(error);
            }
        });
    }

    async loadAllDetections() {
        try {
            const reportFiles = await this.fetchReportFileList();
            console.log(`📁 Found ${reportFiles.length} report files`);

            if (reportFiles.length === 0) {
                throw new Error('No report files found');
            }

            // Load files in batches to prevent overwhelming the server
            const batchSize = 5;
            const batches = this.chunkArray(reportFiles, batchSize);

            for (let i = 0; i < batches.length; i++) {
                const batch = batches[i];
                console.log(`📊 Loading batch ${i + 1}/${batches.length} (${batch.length} files)`);

                const promises = batch.map(file => this.loadSingleReport(file));
                const results = await Promise.allSettled(promises);

                results.forEach((result, index) => {
                    if (result.status === 'fulfilled') {
                        this.detections.push(...result.value);
                    } else {
                        console.warn(`⚠️ Failed to load ${batch[index]}:`, result.reason);
                    }
                });

                // Small delay between batches to be nice to the server
                if (i < batches.length - 1) {
                    await this.delay(100);
                }
            }

            console.log(`✅ Successfully loaded ${this.detections.length} total detections`);
            logDebug(`Loaded ${this.detections.length} detections`);

            if (this.detections.length === 0) {
                throw new Error('No detections found in any report files');
            }

        } catch (error) {
            console.error('❌ Failed to load detections:', error);

            if (this.retryCount < this.maxRetries) {
                this.retryCount++;
                console.log(`🔄 Retrying... (attempt ${this.retryCount}/${this.maxRetries})`);
                await this.delay(1000 * this.retryCount); // Exponential backoff
                return this.loadAllDetections();
            }

            throw error;
        }
    }

    async fetchReportFileList() {
        const cacheKey = 'report-file-list';
        const cacheExpiry = 5 * 60 * 1000; // 5 minutes

        // Check cache first
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < cacheExpiry) {
                return cached.data;
            }
        }

        try {
            const response = await fetch(`reports/?t=${Date.now()}`, {
                headers: {
                    'Cache-Control': 'no-cache'
                }
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

            const html = await response.text();
            const fileMatches = [...html.matchAll(/href="(report_\d{8}_\d{6}\.json)"/g)];
            const files = fileMatches.map(match => match[1]).sort().reverse();

            if (files.length > 0) {
                this.cache.set(cacheKey, {
                    data: files,
                    timestamp: Date.now()
                });
                return files;
            } else {
                throw new Error('No report files found in directory listing');
            }

        } catch (error) {
            console.warn('⚠️ Directory listing failed or empty, falling back to reports.json');

            try {
                const fallbackResponse = await fetch(`reports/reports.json?t=${Date.now()}`, {
                    headers: { 'Cache-Control': 'no-cache' }
                });

                if (!fallbackResponse.ok) throw new Error(`HTTP ${fallbackResponse.status}`);

                const reports = await fallbackResponse.json();

                const filenames = reports.map(r => r.filename).sort().reverse();

                this.cache.set(cacheKey, {
                    data: filenames,
                    timestamp: Date.now()
                });

                return filenames;

            } catch (fallbackError) {
                console.error('❌ Fallback to reports.json also failed:', fallbackError);
                throw fallbackError;
            }
        }
    }

    async loadSingleReport(filename) {
        const cacheKey = `report-${filename}`;

        // Check cache first
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        try {
            const response = await fetch(`reports/${filename}`, {
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'max-age=300' // 5 minutes
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status} for ${filename}`);
            }

            const data = await response.json();
            const detections = Array.isArray(data.detections) ? data.detections : [];

            // Validate and clean detection data
            const validDetections = detections.filter(det => this.validateDetection(det));

            // Cache the result
            this.cache.set(cacheKey, validDetections);

            return validDetections;

        } catch (error) {
            console.warn(`Failed to load ${filename}:`, error);
            return []; // Return empty array instead of throwing
        }
    }

    validateDetection(detection) {
        // Basic validation to ensure detection has required fields
        return (
            detection &&
            typeof detection === 'object' &&
            typeof detection.ra_deg === 'number' &&
            typeof detection.dec_deg === 'number' &&
            !isNaN(detection.ra_deg) &&
            !isNaN(detection.dec_deg) &&
            detection.ra_deg >= -180 && detection.ra_deg <= 360 &&
            detection.dec_deg >= -90 && detection.dec_deg <= 90
        );
    }

    renderVisualization() {
        if (!this.aladin || this.detections.length === 0) {
            logDebug('⚠️ No detections found — nothing to render.');
            console.warn('Cannot render visualization: missing Aladin or detections');
            return;
        }

        this.performanceMetrics.renderStart = performance.now();

        try {
            // Clear existing catalogs
            this.clearCatalogs();

            // Group detections by confidence level for better visualization
            const groupedDetections = this.groupDetectionsByConfidence();

            // Create catalogs for each confidence level
            Object.entries(groupedDetections).forEach(([level, detections]) => {
                if (detections.length === 0) return;

                const config = this.getCatalogConfig(level);
                const catalog = A.catalog({
                    name: config.name,
                    sourceSize: config.size,
                    color: config.color,
                    shape: 'circle'
                });

                // Add sources to catalog
                detections.forEach(detection => {
                    const source = A.source(detection.ra_deg, detection.dec_deg, {
                        name: this.formatSourceName(detection),
                        popupTitle: this.formatPopupTitle(detection),
                        popupDesc: this.formatPopupDescription(detection)
                    });

                    catalog.addSources([source]);
                });

                this.aladin.addCatalog(catalog);
                this.catalogs.set(level, catalog);
            });

            // Set initial view to show interesting region
            this.setOptimalView();

            this.performanceMetrics.renderEnd = performance.now();
            logDebug('Sky map rendered ✔');
            console.log(`🎨 Visualization rendered in ${(this.performanceMetrics.renderEnd - this.performanceMetrics.renderStart).toFixed(2)}ms`);

        } catch (error) {
            console.error('❌ Failed to render visualization:', error);
            this.handleVisualizationError(error);
        }
    }

    groupDetectionsByConfidence() {
        return {
            high: this.detections.filter(d => (d.confidence || 0) >= 0.8),
            medium: this.detections.filter(d => (d.confidence || 0) >= 0.5 && (d.confidence || 0) < 0.8),
            low: this.detections.filter(d => (d.confidence || 0) < 0.5)
        };
    }

    getCatalogConfig(level) {
        const configs = {
            high: { name: 'High Confidence (≥80%)', size: 12, color: '#00c851' },
            medium: { name: 'Medium Confidence (50-80%)', size: 10, color: '#ffbb33' },
            low: { name: 'Low Confidence (<50%)', size: 8, color: '#ff4444' }
        };
        return configs[level] || configs.low;
    }

    formatSourceName(detection) {
        const observer = detection.observer || 'Unknown';
        const confidence = ((detection.confidence || 0) * 100).toFixed(0);
        return `${observer} - ${confidence}%`;
    }

    formatPopupTitle(detection) {
        return `Detection by ${detection.observer || 'Unknown Observer'}`;
    }

    formatPopupDescription(detection) {
        const lines = [];
        lines.push(`RA: ${this.formatRaHms(detection.ra_deg)}<br>DEC: ${this.formatDecDms(detection.dec_deg)}`);


        if (detection.confidence) {
            lines.push(`Confidence: ${(detection.confidence * 100).toFixed(1)}%`);
        }

        if (detection.timestamp_utc) {
            const date = new Date(detection.timestamp_utc).toLocaleDateString();
            lines.push(`Date: ${date}`);
        }

        if (detection.match_name) {
            lines.push(`Match: ${detection.match_name}`);
        }

        if (detection.dimming) {
            lines.push('Type: Dimming Event');
            if (detection.exo_match) {
                lines.push(`Exoplanet: ${detection.exo_match.planet}`);
            }
        } else {
            lines.push('Type: Brightening Event');
        }

        return lines.join('<br>');
    }

    setOptimalView() {
        if (this.detections.length === 0) return;

        // Calculate center point from all detections
        const avgRA = this.detections.reduce((sum, d) => sum + d.ra_deg, 0) / this.detections.length;
        const avgDec = this.detections.reduce((sum, d) => sum + d.dec_deg, 0) / this.detections.length;

        // Calculate appropriate field of view
        const raRange = Math.max(...this.detections.map(d => d.ra_deg)) - Math.min(...this.detections.map(d => d.ra_deg));
        const decRange = Math.max(...this.detections.map(d => d.dec_deg)) - Math.min(...this.detections.map(d => d.dec_deg));
        const fov = Math.max(Math.max(raRange, decRange) * 1.2, 10); // Add 20% padding, minimum 10°

        this.aladin.gotoRaDec(avgRA, avgDec);
        this.aladin.setFoV(Math.min(fov, 180)); // Cap at 180°
    }

    populateTable() {
        const tbody = document.querySelector('#detectionTable tbody');
        if (!tbody) {
            console.warn('Detection table not found');
            return;
        }

        // Clear existing rows
        tbody.innerHTML = '';

        // Sort detections by confidence (highest first), then by timestamp
        const sortedDetections = [...this.detections].sort((a, b) => {
            const confA = a.confidence || 0;
            const confB = b.confidence || 0;

            if (confA !== confB) {
                return confB - confA;
            }

            // Secondary sort by timestamp (most recent first)
            const timeA = a.timestamp_utc ? new Date(a.timestamp_utc).getTime() : 0;
            const timeB = b.timestamp_utc ? new Date(b.timestamp_utc).getTime() : 0;
            return timeB - timeA;
        });

        // Limit to top 100 detections for performance
        const displayDetections = sortedDetections.slice(0, 100);

        displayDetections.forEach((detection, index) => {
            const row = this.createTableRow(detection, index);
            tbody.appendChild(row);
        });

        // Add note if we're showing limited results
        if (sortedDetections.length > 100) {
            const noteRow = document.createElement('tr');
            noteRow.innerHTML = `
                <td colspan="6" style="text-align: center; color: var(--text-secondary); font-style: italic; padding: 1rem;">
                    Showing top 100 detections (of ${sortedDetections.length} total)
                </td>
            `;
            tbody.appendChild(noteRow);
        }
    }

    createTableRow(detection, index) {
        const row = document.createElement('tr');
        row.style.animationDelay = `${index * 0.05}s`;

        const confidence = (detection.confidence || 0) * 100;
        const confidenceColor = this.getConfidenceColor(confidence);

        const coords = `RA: ${this.formatRaHms(detection.ra_deg)}, DEC: ${this.formatDecDms(detection.dec_deg)}`;
        const date = detection.timestamp_utc ?
            new Date(detection.timestamp_utc).toLocaleDateString() : 'Unknown';
        const observer = detection.observer || 'Unknown';
        const match = detection.match_name || '—';

        // Add type indicator
        let typeIndicator = detection.dimming ? '🌑 Dimming' : '✨ Brightening';
        if (detection.exo_match) {
            typeIndicator += ' (Exoplanet Candidate)';
        }

        row.innerHTML = `
            <td>${observer}</td>
            <td title="${coords}">${coords}</td>
            <td>${date}</td>
            <td><span style="color: ${confidenceColor}; font-weight: bold;">${confidence.toFixed(0)}%</span></td>
            <td>${typeIndicator}</td>
            <td title="${match}" style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                ${match}
            </td>
        `;

        // Add click handler to center map on detection
        row.style.cursor = 'pointer';
        row.addEventListener('click', () => {
            this.centerOnDetection(detection);
            row.style.background = 'rgba(0, 212, 255, 0.2)';
            setTimeout(() => {
                row.style.background = '';
            }, 2000);
        });

        return row;
    }

    centerOnDetection(detection) {
        if (this.aladin) {
            this.aladin.gotoRaDec(detection.ra_deg, detection.dec_deg);
            this.aladin.setFoV(5); // Zoom in to 5 degrees
        }
    }

    getConfidenceColor(confidence) {
        if (confidence >= 80) return '#00c851';
        if (confidence >= 50) return '#ffbb33';
        return '#ff4444';
    }

    updateStatistics() {
        const stats = this.calculateStatistics();

        // Update stat counters with animation
        Object.entries(stats).forEach(([key, value]) => {
            const element = document.getElementById(key);
            if (element) {
                this.animateCounter(element, value);
            }
        });
    }

    calculateStatistics() {
        const totalDetections = this.detections.length;
        const uniqueObservers = new Set(this.detections.map(d => d.observer || 'Unknown')).size;
        const exoplanetCandidates = this.detections.filter(d => d.exo_match).length;
        const highConfidence = this.detections.filter(d => (d.confidence || 0) >= 0.8).length;

        return {
            'total-detections': totalDetections,
            'active-observers': uniqueObservers,
            'exoplanet-candidates': exoplanetCandidates,
            'high-confidence': highConfidence
        };
    }

    animateCounter(element, targetValue) {
        const startValue = 0;
        const duration = 1500; // 1.5 seconds
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function for smooth animation
            const easeOutCubic = 1 - Math.pow(1 - progress, 3);
            const currentValue = Math.floor(startValue + (targetValue - startValue) * easeOutCubic);

            element.textContent = currentValue.toLocaleString();

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    setupEventListeners() {
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'r':
                        e.preventDefault();
                        this.refreshData();
                        break;
                    case 'f':
                        e.preventDefault();
                        this.toggleFullscreen();
                        break;
                }
            }
        });

        // Window resize handler
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                if (this.aladin) {
                    this.aladin.view.requestRedraw();
                }
            }, 250);
        });

        // Visibility change handler (pause/resume when tab is hidden/visible)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseUpdates();
            } else {
                this.resumeUpdates();
            }
        });

        // Add refresh button functionality
        const refreshBtn = document.createElement('button');
        refreshBtn.textContent = '🔄 Refresh Data';
        refreshBtn.className = 'btn';
        refreshBtn.style.cssText = 'position: fixed; top: 100px; right: 20px; z-index: 1000;';
        refreshBtn.addEventListener('click', () => this.refreshData());
        document.body.appendChild(refreshBtn);
    }

    async refreshData() {
        console.log('🔄 Refreshing detection data...');
        this.cache.clear(); // Clear cache to force fresh data
        this.retryCount = 0;

        try {
            this.showLoadingState(true);
            this.detections = [];
            await this.loadAllDetections();
            this.renderVisualization();
            this.populateTable();
            this.updateStatistics();
            console.log('✅ Data refreshed successfully');
        } catch (error) {
            console.error('❌ Failed to refresh data:', error);
            this.handleError(error);
        } finally {
            this.showLoadingState(false);
        }
    }

    toggleFullscreen() {
        const mapDiv = document.getElementById('aladin-lite-div');
        if (!document.fullscreenElement) {
            mapDiv.requestFullscreen?.() || mapDiv.webkitRequestFullscreen?.();
        } else {
            document.exitFullscreen?.() || document.webkitExitFullscreen?.();
        }
    }

    pauseUpdates() {
        console.log('⏸️ Pausing updates (tab hidden)');
        // Could implement auto-refresh pausing here
    }

    resumeUpdates() {
        console.log('▶️ Resuming updates (tab visible)');
        // Could implement auto-refresh resuming here
    }

    clearCatalogs() {
        this.catalogs.forEach(catalog => {
            try {
                this.aladin.removeCatalog(catalog);
            } catch (error) {
                console.warn('Failed to remove catalog:', error);
            }
        });
        this.catalogs.clear();
    }

    handleError(error) {
        const errorContainer = this.createErrorContainer();
        errorContainer.innerHTML = `
            <div class="error-content">
                <h3>⚠️ Unable to Load Detection Data</h3>
                <p>We encountered an issue loading the detection data:</p>
                <code>${error.message}</code>
                <div class="error-actions">
                    <button class="btn btn-primary" onclick="location.reload()">
                        🔄 Reload Page
                    </button>
                    <button class="btn btn-secondary" onclick="this.closest('.error-container').remove()">
                        ✕ Dismiss
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(errorContainer);
    }

    handleVisualizationError(error) {
        const mapDiv = document.getElementById('aladin-lite-div');
        if (mapDiv) {
            mapDiv.innerHTML = `
                <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; background: var(--surface); color: var(--text-primary); text-align: center; padding: 2rem;">
                    <h3 style="color: var(--error); margin-bottom: 1rem;">🔧 Visualization Error</h3>
                    <p style="margin-bottom: 1rem;">Failed to render sky map visualization.</p>
                    <button class="btn" onclick="window.viewer.renderVisualization()">
                        🔄 Retry Visualization
                    </button>
                </div>
            `;
        }
    }

    createErrorContainer() {
        const container = document.createElement('div');
        container.className = 'error-container';
        container.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 2000;
            padding: 2rem;
        `;

        const style = document.createElement('style');
        style.textContent = `
            .error-content {
                background: var(--surface);
                border-radius: 15px;
                padding: 2rem;
                max-width: 500px;
                width: 100%;
                border: 1px solid var(--error);
                text-align: center;
            }
            .error-content h3 {
                color: var(--error);
                margin-bottom: 1rem;
            }
            .error-content code {
                background: var(--surface-light);
                padding: 0.5rem;
                border-radius: 4px;
                display: block;
                margin: 1rem 0;
                font-family: monospace;
                font-size: 0.9rem;
            }
            .error-actions {
                display: flex;
                gap: 1rem;
                justify-content: center;
                margin-top: 1.5rem;
            }
        `;
        document.head.appendChild(style);

        return container;
    }

    // Utility functions
    chunkArray(array, chunkSize) {
        const chunks = [];
        for (let i = 0; i < array.length; i += chunkSize) {
            chunks.push(array.slice(i, i + chunkSize));
        }
        return chunks;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Performance monitoring
    getPerformanceReport() {
        const loadTime = this.performanceMetrics.loadEnd - this.performanceMetrics.loadStart;
        const renderTime = this.performanceMetrics.renderEnd - this.performanceMetrics.renderStart;

        return {
            totalDetections: this.detections.length,
            loadTimeMs: loadTime.toFixed(2),
            renderTimeMs: renderTime.toFixed(2),
            cacheHits: Array.from(this.cache.keys()).length,
            memoryUsage: this.getMemoryUsage()
        };
    }

    getMemoryUsage() {
        if (performance.memory) {
            return {
                used: (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2) + ' MB',
                total: (performance.memory.totalJSHeapSize / 1024 / 1024).toFixed(2) + ' MB'
            };
        }
        return 'Not available';
    }

    // Debug function
    debug() {
        console.group('🔍 PulseHunter Viewer Debug Info');
        console.log('Performance:', this.getPerformanceReport());
        console.log('Cache keys:', Array.from(this.cache.keys()));
        console.log('Detections sample:', this.detections.slice(0, 3));
        console.log('Catalogs:', Array.from(this.catalogs.keys()));
        console.groupEnd();
    }
}

// Global functions and initialization
async function loadDetections() {
    console.warn('loadDetections() is deprecated. Use PulseHunterViewer class instead.');
    if (!window.viewer) {
        window.viewer = new PulseHunterViewer();
    }
}

function logDebug(message) {
    console.log(message);
}

function createDebugOverlay() {
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.bottom = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.maxHeight = '150px';
    overlay.style.overflowY = 'auto';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.75)';
    overlay.style.color = 'lime';
    overlay.style.fontFamily = 'monospace';
    overlay.style.fontSize = '12px';
    overlay.style.padding = '4px';
    overlay.style.zIndex = '9999';
    overlay.id = 'debugOverlay';
    document.body.appendChild(overlay);
}

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", function () {
    console.log('🌟 Starting PulseHunter Detection Viewer');

    waitForAladin().then(() => {
        window.viewer = new PulseHunterViewer();

        // Add debug access
        window.debugViewer = () => window.viewer.debug();
    }).catch(err => {
        console.error("❌ Aladin initialization failed:", err);

        // Try to initialize without Aladin for debugging
        console.log('🔧 Attempting to initialize without Aladin...');
        window.viewer = new PulseHunterViewer();
        window.debugViewer = () => window.viewer.debug();
    });
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    event.preventDefault();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PulseHunterViewer;
}
