#!/usr/bin/env python3
"""
Quick PulseHunter Results Viewer
View your 43 detections without the GUI
"""

import json
import os
from pathlib import Path


def view_results():
    # Try to find the results file
    possible_files = [
        r"F:\astrophotography\2024-07-13 - bortle2 - sthelens\pulsehunter_results.json",
        "pulsehunter_results.json",
        "../pulsehunter_results.json",
    ]

    results_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            results_file = file_path
            break

    if not results_file:
        print("❌ Could not find pulsehunter_results.json")
        print("Search locations:")
        for path in possible_files:
            print(f"  - {path}")
        return

    # Load and display results
    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        detections = data.get("detections", [])
        metadata = data.get("metadata", {})

        print("🌌 PulseHunter Detection Results")
        print("=" * 40)
        print(f"📁 File: {results_file}")
        print(f"📅 Generated: {metadata.get('generated_at', 'Unknown')}")
        print(f"🔍 Total Detections: {len(detections)}")
        print(f"⭐ High Confidence: {metadata.get('high_confidence_count', 0)}")
        print(f"🪐 Exoplanet Candidates: {metadata.get('exoplanet_candidates', 0)}")
        print(f"🌟 GAIA Matches: {metadata.get('gaia_matches', 0)}")

        print("\n🎯 Top 15 Detections by Confidence:")
        print("-" * 80)
        print(
            f"{'#':<3} {'Frame':<6} {'Z-Score':<8} {'Conf%':<6} {'Type':<12} {'GAIA Match':<15}"
        )
        print("-" * 80)

        # Sort by confidence
        sorted_detections = sorted(
            detections, key=lambda x: x.get("confidence", 0), reverse=True
        )

        for i, det in enumerate(sorted_detections[:15]):
            frame = det.get("frame", "?")
            z_score = det.get("z_score", 0)
            confidence = det.get("confidence", 0) * 100
            det_type = "Dimming" if det.get("dimming") else "Brightening"
            gaia_match = "Yes" if det.get("match_name") else "No"

            print(
                f"{i+1:<3} {frame:<6} {z_score:<8.1f} {confidence:<6.0f} {det_type:<12} {gaia_match:<15}"
            )

        print("\n🔬 Analysis Suggestions:")
        brightening = sum(1 for d in detections if not d.get("dimming", True))
        dimming = len(detections) - brightening

        print(f"  • {brightening} brightening events (potential SETI signals)")
        print(f"  • {dimming} dimming events (potential exoplanet transits)")
        print(f"  • Focus on high confidence detections (>80%)")
        print(f"  • Check unmatched sources (No GAIA match)")
        print(f"  • Your Bortle 2 site gives excellent sensitivity!")

        print("\n🚀 Next Steps:")
        print("  1. Run the fixed GUI: python pulse_gui.py")
        print("  2. File → Open Project → Select your JSON file")
        print("  3. Explore detections in Results tab")
        print("  4. Export high-confidence candidates for further analysis")

    except Exception as e:
        print(f"❌ Error reading results: {e}")


if __name__ == "__main__":
    view_results()
