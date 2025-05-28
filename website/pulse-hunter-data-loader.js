/**
 * Fixed PulseHunter Data Loader - Works with your actual file structure
 */

class PulseHunterDataLoader {
    constructor(options = {}) {
        this.options = {
            reportsPath: 'reports/',
            maxFiles: options.maxFiles || 10,
            timeout: options.timeout || 10000,
            debug: options.debug || false,
            ...options
        };

        this.cache = new Map();
        this.data = {
            detections: [],
            files: [],
            statistics: {
                totalFiles: 0,
                loadedFiles: 0,
                totalDetections: 0,
                validDetections: 0,
                exoplanetCandidates: 0,
                uniqueObservers: 0,
                highConfidence: 0
            }
        };

        this.listeners = {};
    }

    // Event system
    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }

    log(message, type = 'info') {
        if (this.options.debug) {
            console.log(`[PulseHunter] ${message}`);
        }
        this.emit('log', { message, type, timestamp: new Date() });
    }

    async loadAllData() {
        try {
            this.log('Starting data loading...', 'info');
            this.emit('loadStart', {});

            // Try reports.json first (you have this!)
            const reportsJsonData = await this.loadReportsJson();
            if (reportsJsonData && reportsJsonData.length > 0) {
                this.log(`Loaded ${reportsJsonData.length} detections from reports.json`, 'success');
                this.data.detections = reportsJsonData;
                this.data.files.push({
                    filename: 'reports.json',
                    loaded: true,
                    detections: reportsJsonData.length,
                    error: null
                });
                this.data.statistics.loadedFiles = 1;
                this.data.statistics.totalFiles = 1;
            } else {
                // Fallback to individual files
                const files = await this.discoverReportFiles();
                if (files.length > 0) {
                    await this.loadReportFiles(files);
                } else {
                    throw new Error('No report files found');
                }
            }

            // Process the data
            this.processDetections();
            this.calculateStatistics();

            this.log(`Loading complete: ${this.data.statistics.validDetections} valid detections`, 'success');
            this.emit('loadComplete', this.data);

            return this.data;

        } catch (error) {
            this.log(`Loading failed: ${error.message}`, 'error');
            this.emit('loadError', error);

            // Generate fallback data
            const fallbackData = await this.loadFallbackData();
            return fallbackData;
        }
    }

    // Load reports.json (your main file with 215 detections)
    async loadReportsJson() {
        try {
            this.log('Trying to load reports.json...', 'info');
            const response = await fetch(`${this.options.reportsPath}reports.json`);

            if (!response.ok) {
                throw new Error(`reports.json not found: ${response.status}`);
            }

            const data = await response.json();

            if (Array.isArray(data)) {
                this.log(`reports.json contains ${data.length} detections`, 'success');
                return data;
            } else if (data.detections && Array.isArray(data.detections)) {
                this.log(`reports.json contains ${data.detections.length} detections`, 'success');
                return data.detections;
            } else {
                throw new Error('reports.json has invalid structure');
            }

        } catch (error) {
            this.log(`Could not load reports.json: ${error.message}`, 'warning');
            return null;
        }
    }

    // Discover individual report files (fixed regex patterns)
    async discoverReportFiles() {
        try {
            this.log('Discovering individual report files...', 'info');

            const response = await fetch(`${this.options.reportsPath}`);
            if (!response.ok) {
                throw new Error(`Directory listing failed: ${response.status}`);
            }

            const html = await response.text();

            // Fixed regex patterns to handle your file structure
            const patterns = [
                // Pattern 1: Full path hrefs like /pulsehunter/reports/report_*.json
                /href="[^"]*\/(report_\d{8}_\d{6}\.json)"/g,
                // Pattern 2: Relative hrefs like report_*.json
                /href="(report_\d{8}_\d{6}\.json)"/g,
                // Pattern 3: Any report file
                /href="[^"]*\/(report_[^"]*\.json)"/g
            ];

            let files = [];

            for (const pattern of patterns) {
                const matches = [...html.matchAll(pattern)];
                if (matches.length > 0) {
                    files = matches.map(m => m[1]);
                    this.log(`Found ${files.length} files with pattern: ${pattern}`, 'success');
                    break;
                }
            }

            if (files.length === 0) {
                this.log('No individual report files found with expected patterns', 'warning');
            }

            return files.slice(0, this.options.maxFiles);

        } catch (error) {
            this.log(`File discovery failed: ${error.message}`, 'error');
            return [];
        }
    }

    // Load individual report files
    async loadReportFiles(filenames) {
        this.log(`Loading ${filenames.length} individual report files...`, 'info');

        const allDetections = [];
        let loaded = 0;

        for (const filename of filenames) {
            try {
                this.log(`Loading ${filename}...`, 'info');
                const response = await fetch(`${this.options.reportsPath}${filename}`);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const data = await response.json();
                let detections = [];

                if (data.detections && Array.isArray(data.detections)) {
                    detections = data.detections;
                } else if (Array.isArray(data)) {
                    detections = data;
                } else {
                    throw new Error('Invalid file structure');
                }

                allDetections.push(...detections);
                this.data.files.push({
                    filename,
                    loaded: true,
                    detections: detections.length,
                    error: null
                });
                loaded++;

                this.log(`✅ ${filename}: ${detections.length} detections`, 'success');

            } catch (error) {
                this.log(`❌ ${filename}: ${error.message}`, 'error');
                this.data.files.push({
                    filename,
                    loaded: false,
                    detections: 0,
                    error: error.message
                });
            }

            // Emit progress
            this.emit('progress', {
                progress: Math.round(((this.data.files.length) / filenames.length) * 100),
                currentFile: filename,
                loaded: loaded,
                total: filenames.length
            });
        }

        this.data.detections = allDetections;
        this.data.statistics.totalFiles = filenames.length;
        this.data.statistics.loadedFiles = loaded;
        this.data.statistics.totalDetections = allDetections.length;
    }

    // Process and validate detections
    processDetections() {
        this.log('Processing and validating detections...', 'info');

        const validDetections = this.data.detections.filter(detection => {
            return (
                detection &&
                typeof detection === 'object' &&
                typeof detection.ra_deg === 'number' &&
                typeof detection.dec_deg === 'number' &&
                !isNaN(detection.ra_deg) &&
                !isNaN(detection.dec_deg) &&
                detection.ra_deg >= 0 && detection.ra_deg <= 360 &&
                detection.dec_deg >= -90 && detection.dec_deg <= 90
            );
        });

        const invalidCount = this.data.detections.length - validDetections.length;
        if (invalidCount > 0) {
            this.log(`Filtered out ${invalidCount} invalid detections`, 'warning');
        }

        // Enhance detections
        validDetections.forEach(detection => {
            detection.confidence = detection.confidence || 0;
            detection.observer = detection.observer || 'Unknown';
        });

        this.data.detections = validDetections;
        this.data.statistics.validDetections = validDetections.length;
    }

    // Calculate statistics
    calculateStatistics() {
        const stats = this.data.statistics;
        const detections = this.data.detections;

        stats.exoplanetCandidates = detections.filter(d => d.exo_match).length;
        stats.uniqueObservers = new Set(detections.map(d => d.observer || 'Unknown')).size;
        stats.highConfidence = detections.filter(d => (d.confidence || 0) >= 0.8).length;

        this.log(`Statistics: ${stats.validDetections} detections, ${stats.exoplanetCandidates} exoplanets, ${stats.uniqueObservers} observers`, 'info');
    }

    // Generate fallback data if real data fails
    async loadFallbackData() {
        this.log('Generating fallback sample data...', 'warning');

        const sampleDetections = [];
        const observers = ['Alice Thompson', 'Bob Martinez', 'Carol Singh', 'Dave Johnson'];

        for (let i = 0; i < 50; i++) {
            sampleDetections.push({
                ra_deg: Math.random() * 360,
                dec_deg: (Math.random() - 0.5) * 160,
                confidence: Math.random(),
                observer: observers[Math.floor(Math.random() * observers.length)],
                timestamp_utc: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
                dimming: Math.random() > 0.4,
                frame: Math.floor(Math.random() * 1000),
                exo_match: Math.random() > 0.9 ? {
                    planet: `Kepler-${Math.floor(Math.random() * 1000)}b`
                } : null
            });
        }

        this.data = {
            detections: sampleDetections,
            files: [{ filename: 'sample_data.json', loaded: true, detections: sampleDetections.length }],
            statistics: {
                totalFiles: 1,
                loadedFiles: 1,
                totalDetections: sampleDetections.length,
                validDetections: sampleDetections.length,
                exoplanetCandidates: sampleDetections.filter(d => d.exo_match).length,
                uniqueObservers: new Set(sampleDetections.map(d => d.observer)).size,
                highConfidence: sampleDetections.filter(d => d.confidence >= 0.8).length
            }
        };

        this.emit('loadComplete', this.data);
        return this.data;
    }

    // Public API methods
    getDetections(filters = {}) {
        let filtered = [...this.data.detections];

        if (filters.confidence !== undefined) {
            filtered = filtered.filter(d => (d.confidence || 0) >= filters.confidence);
        }

        if (filters.type) {
            switch (filters.type) {
                case 'dimming':
                    filtered = filtered.filter(d => d.dimming);
                    break;
                case 'brightening':
                    filtered = filtered.filter(d => !d.dimming);
                    break;
                case 'exoplanet':
                    filtered = filtered.filter(d => d.exo_match);
                    break;
            }
        }

        return filtered;
    }

    getStatistics() {
        return { ...this.data.statistics };
    }

    isLoaded() {
        return this.data.detections.length > 0;
    }
}
