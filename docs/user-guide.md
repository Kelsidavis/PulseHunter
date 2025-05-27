# PulseHunter User Guide

Welcome to PulseHunter! This guide will help you get started with detecting optical transients and contributing to the search for extraterrestrial intelligence.

## Table of Contents
- [Getting Started](#getting-started)
- [Understanding the Interface](#understanding-the-interface)
- [Observing Strategy](#observing-strategy)
- [Data Processing](#data-processing)
- [Interpreting Results](#interpreting-results)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Your First Session

1. **Launch PulseHunter**
```bash
python pulse_gui.py
```

2. **Set up Calibration** (recommended)
   - Click "Calibration Setup"
   - Select your camera type (Monochrome/OSC)
   - Enter your observer name
   - Create master calibration frames
   - Set ASTAP path for plate solving

3. **Select Your Data**
   - Click "Select FITS Folder"
   - Choose folder with your FITS files
   - Ensure files are from the same session

4. **Configure Detection**
   - Set detection threshold (start with 6.0σ)
   - Higher values = fewer false positives
   - Lower values = more sensitivity

5. **Run Detection**
   - Click "Start Detection"
   - Monitor progress and log output
   - Review results when complete

### What You Need

**Essential:**
- FITS files from astronomical observations
- Consistent exposure times within a session
- Minimum 20-50 frames for statistics

**Recommended:**
- Calibration frames (bias, dark, flat)
- Plate solving capability (ASTAP)
- Star-rich fields for better results

## Understanding the Interface

### Main Window Components

**Control Panel:**
- **Dataset Folder**: Path to your FITS files
- **Detection Threshold**: Sensitivity setting (σ)
- **Calibration Setup**: Configure observatory settings
- **Start Detection**: Begin processing

**Progress Section:**
- **Processing Status**: Real-time progress bar
- **Log Output**: Detailed processing information

**Results Panel:**
- **Detection List**: All found candidates
- **Preview Image**: Cutout of selected detection
- **Filter Options**: Show only unmatched detections

**Export Section:**
- **Generate & Upload Report**: Create comprehensive output

### Detection List Columns

- **Observer**: Your name/observatory identifier
- **Coordinates**: RA/Dec in degrees (if plate solved)
- **Date**: Observation timestamp
- **Confidence**: Detection reliability (0-100%)
- **Type**: Brightening ✨ or Dimming 🌑 event
- **GAIA Match**: Known catalog object identification

### Confidence Levels

- **High (80-100%)**: 🟢 Strong candidates, likely genuine
- **Medium (50-79%)**: 🟡 Possible detections, need review
- **Low (<50%)**: 🔴 Uncertain, may be noise

## Observing Strategy

### Target Selection

**Good Targets:**
- Star-rich fields (Milky Way regions)
- Open clusters
- Known variable star fields
- Exoplanet host stars

**Avoid:**
- Empty sky regions
- Areas with bright planets
- Light-polluted fields
- Regions with many galaxies (for SETI)

### Observation Parameters

**Timing:**
- Minimum 1-2 hours per session
- Consistent cadence (same exposure interval)
- Monitor same field across multiple nights

**Exposure Settings:**
- Balance SNR vs. temporal resolution
- Typical: 30-120 seconds per frame
- Avoid saturation of bright stars
- Maintain consistent exposure times

**Environmental Conditions:**
- Stable atmospheric conditions preferred
- Avoid windy nights (tracking issues)
- Good seeing helps with photometry
- Dark sky locations yield better results

### Field Selection Strategy

**For SETI Detection:**
- Target nearby star systems
- Focus on Sun-like stars
- Monitor known exoplanet hosts
- Choose fields observable for long periods

**For Exoplanet Transits:**
- Known transit systems for confirmation
- TESS candidates needing follow-up
- Systems with favorable geometry
- Bright host stars (magnitude < 12)

## Data Processing

### Calibration Workflow

**Master Bias Creation:**
```
20-50 bias frames → Master Bias
- Shortest possible exposure
- Same temperature as science frames
- Camera shutter closed
```

**Master Dark Creation:**
```
10-20 dark frames → Master Dark
- Same exposure as science frames
- Same temperature as science frames
- Camera shutter closed
```

**Master Flat Creation:**
```
10-20 flat frames → Master Flat
- Twilight sky or light panel
- Same filter as science frames
- Uniform illumination across field
```

### Detection Process

1. **Image Calibration**
   - Subtract master bias from all frames
   - Subtract master dark from all frames
   - Divide by normalized master flat

2. **Astrometric Calibration**
   - ASTAP plate solving for WCS
   - Enables coordinate determination
   - Required for catalog crossmatching

3. **Statistical Analysis**
   - Compute mean and standard deviation images
   - Calculate z-scores for each pixel
   - Identify outliers above threshold

4. **Candidate Filtering**
   - Reject detections near image edges
   - Filter elongated features (cosmic rays, satellites)
   - Apply confidence scoring

5. **Catalog Crossmatching**
   - Query GAIA DR3 for known stars
   - Check NASA Exoplanet Archive
   - Flag known variable stars

### Quality Control

**Automatic Checks:**
- Statistical significance testing
- Morphology filtering
- Edge rejection
- Catalog verification

**Manual Review:**
- Inspect high-confidence detections
- Check cutout images for artifacts
- Verify light curve behavior
- Cross-reference with observing logs

## Interpreting Results

### Understanding Detections

**Brightening Events ✨:**
- Possible optical flashes
- Stellar flares
- Asteroid occultation events
- Instrumental artifacts

**Dimming Events 🌑:**
- Potential exoplanet transits
- Variable star minima
- Asteroid occultations
- Cloud interference

### Light Curve Analysis

**Good Detections Show:**
- Clear signal above noise
- Consistent behavior across frames
- Reasonable photometric values
- Expected temporal evolution

**Suspect Detections Show:**
- Single-frame spikes
- Extreme brightness values
- Irregular patterns
- Edge-of-field locations

### GAIA Cross-matching

**Known Star Match:**
- Likely instrumental or atmospheric effect
- Could be stellar variability
- Check known variable star catalogs
- May still be scientifically interesting

**No Match Found:**
- More interesting for SETI
- Could be genuine transient
- Warrants careful investigation
- Prime candidate for follow-up

### Exoplanet Candidates

**Matched Exoplanets:**
- Transit confirmation opportunities
- Verify timing and depth
- Contribute to ephemeris updates
- Valuable for citizen science

**Unmatched Dimming:**
- Potential new discoveries
- Check for periodic behavior
- Calculate transit parameters
- Report to exoplanet databases

## Best Practices

### Data Quality

**Pre-Processing:**
- Always use calibration frames
- Ensure stable tracking/guiding
- Monitor focus throughout session
- Record detailed observing logs

**Processing Parameters:**
- Start with conservative thresholds (6σ)
- Process test data first
- Validate with known sources
- Document parameter choices

### Scientific Rigor

**Documentation:**
- Record all observing conditions
- Note any equipment issues
- Keep detailed processing logs
- Save all intermediate results

**Validation:**
- Cross-check suspicious detections
- Compare with other observers
- Verify coordinates independently
- Follow up promising candidates

### Community Participation

**Data Sharing:**
- Upload results to network database
- Share interesting discoveries
- Participate in coordinated campaigns
- Contribute to follow-up observations

**Collaboration:**
- Join observer forums and chats
- Share processing techniques
- Coordinate observations
- Mentor new participants

## Troubleshooting

### Common Issues

**No Detections Found:**
- Check detection threshold (try 4-5σ)
- Verify data quality and calibration
- Ensure sufficient frames for statistics
- Review field selection

**Too Many False Positives:**
- Increase detection threshold
- Improve calibration frames
- Check for systematic issues
- Review edge masking parameters

**Poor Photometry:**
- Verify flat field calibration
- Check for focus drift
- Ensure stable tracking
- Monitor atmospheric conditions

**Plate Solving Failures:**
- Verify ASTAP installation
- Check star catalog availability
- Ensure sufficient field stars
- Try manual initial coordinates

### Performance Issues

**Slow Processing:**
- Reduce image size or batch size
- Close other applications
- Use SSD storage for temporary files
- Consider processing overnight

**Memory Errors:**
- Process smaller batches
- Increase virtual memory
- Use 64-bit Python
- Monitor system resources

**GUI Responsiveness:**
- Let processing complete
- Check system specifications
- Update graphics drivers
- Restart application if needed

### Getting Help

**Self-Help Resources:**
- Check log output for error messages
- Review FAQ documentation
- Search GitHub issues
- Try with sample data

**Community Support:**
- Post in user forums
- Contact other observers
- Share problem datasets
- Ask for processing help

**Developer Contact:**
- Report bugs on GitHub
- Email technical issues
- Request new features
- Contribute improvements

## Advanced Topics

### Custom Processing

**Batch Processing:**
```bash
python -m pulsehunter.batch --input-dir /path/to/data --threshold 5.0
```

**Parameter Tuning:**
- Adjust detection thresholds for your data
- Optimize cutout sizes
- Customize edge margins
- Fine-tune confidence scoring

**Integration with Other Tools:**
- Export data for external analysis
- Import results from other surveys
- Coordinate with professional facilities
- Contribute to research projects

### Research Applications

**Publication Guidelines:**
- Acknowledge PulseHunter in papers
- Cite relevant methodology references
- Share code modifications
- Report significant discoveries

**Data Archiving:**
- Preserve original FITS files
- Archive processing parameters
- Document all modifications
- Enable reproducible research

## Conclusion

PulseHunter empowers amateur astronomers to contribute meaningfully to cutting-edge research. By following these guidelines and best practices, you'll maximize your chances of making genuine discoveries while contributing valuable data to the global search for extraterrestrial intelligence.

Remember: every observation contributes to our understanding of the universe. Your participation in this citizen science network helps expand humanity's search for life among the stars!

**Happy hunting! 🌟**

---

*For additional support, visit our [GitHub repository](https://github.com/Kelsidavis/PulseHunter) or contact [pulsehunter@geekastro.dev](mailto:pulsehunter@geekastro.dev)*
