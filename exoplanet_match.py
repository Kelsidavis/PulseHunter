"""
Exoplanet Transit Matching Module for PulseHunter
Updated to use NASA Exoplanet Archive PS (Planetary Systems) table
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive


def match_transits_with_exoplanets(detections, radius_arcsec=5.0):
    """
    Cross-match detections with known exoplanets from NASA Exoplanet Archive

    Updated to use the new 'ps' (Planetary Systems) table instead of
    the deprecated 'exoplanets' table.

    Args:
        detections (list): List of detection dictionaries
        radius_arcsec (float): Search radius in arcseconds

    Returns:
        list: Updated detections with exoplanet matches
    """
    try:
        print("Querying NASA Exoplanet Archive (PS table)...")

        # Query the new PS (Planetary Systems) table
        catalog = NasaExoplanetArchive.query_criteria(
            table="ps",  # Updated from deprecated "exoplanets" table
            select="pl_name,ra,dec,pl_orbper,pl_trandep,hostname,sy_dist,pl_rade,pl_masse,disc_year",
            where="pl_trandep IS NOT NULL AND ra IS NOT NULL AND dec IS NOT NULL",
        )

        print(
            f"Successfully loaded {len(catalog)} exoplanets from NASA Exoplanet Archive (PS table)"
        )

    except Exception as e:
        print("Failed to fetch exoplanet archive:", e)
        print("This might be due to network issues or changes in the archive API")
        return detections

    matched = []
    match_count = 0

    for det in detections:
        # Skip detections without coordinates or non-dimming events
        if (
            det.get("ra_deg") is None
            or det.get("dec_deg") is None
            or not det.get("dimming")
        ):
            matched.append(det)
            continue

        coord = SkyCoord(
            ra=det["ra_deg"] * u.deg, dec=det["dec_deg"] * u.deg, frame="icrs"
        )

        found_match = False
        best_match = None
        best_separation = float("inf")

        # Find the closest match within the search radius
        for row in catalog:
            try:
                ep_coord = SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg, frame="icrs")
                sep = coord.separation(ep_coord).arcsecond

                if sep <= radius_arcsec and sep < best_separation:
                    best_separation = sep
                    best_match = row
                    found_match = True

            except Exception as e:
                # Skip rows with invalid coordinates
                continue

        if found_match and best_match is not None:
            # Create enhanced exoplanet match information
            exo_match = {
                "host": str(best_match["hostname"])
                if best_match["hostname"]
                else "Unknown",
                "planet": str(best_match["pl_name"])
                if best_match["pl_name"]
                else "Unknown",
                "sep_arcsec": round(best_separation, 2),
                "period_days": (
                    round(float(best_match["pl_orbper"]), 2)
                    if best_match["pl_orbper"]
                    and not np.isnan(float(best_match["pl_orbper"]))
                    else None
                ),
                "depth_ppm": (
                    round(float(best_match["pl_trandep"]) * 1e6, 1)
                    if best_match["pl_trandep"]
                    and not np.isnan(float(best_match["pl_trandep"]))
                    else None
                ),
            }

            # Add additional parameters if available
            if best_match["sy_dist"] and not np.isnan(float(best_match["sy_dist"])):
                exo_match["distance_pc"] = round(float(best_match["sy_dist"]), 1)

            if best_match["pl_rade"] and not np.isnan(float(best_match["pl_rade"])):
                exo_match["radius_earth"] = round(float(best_match["pl_rade"]), 2)

            if best_match["pl_masse"] and not np.isnan(float(best_match["pl_masse"])):
                exo_match["mass_earth"] = round(float(best_match["pl_masse"]), 2)

            if best_match["disc_year"] and not np.isnan(float(best_match["disc_year"])):
                exo_match["discovery_year"] = int(best_match["disc_year"])

            det["exo_match"] = exo_match
            match_count += 1

            print(
                f"Exoplanet match found: {exo_match['planet']} "
                f'(separation: {best_separation:.1f}")'
            )
        else:
            det["exo_match"] = None

        matched.append(det)

    dimming_count = len([d for d in detections if d.get("dimming")])
    print(
        f"Exoplanet cross-matching complete: {match_count} matches found out of {dimming_count} dimming events"
    )
    return matched


def get_exoplanet_stats():
    """
    Get statistics about the exoplanet catalog

    Returns:
        dict: Statistics about available exoplanets
    """
    try:
        print("Querying exoplanet statistics from PS table...")

        # Query basic statistics from the PS table
        catalog = NasaExoplanetArchive.query_criteria(
            table="ps",
            select="pl_name,pl_trandep,pl_orbper,disc_year,hostname",
            where="pl_trandep IS NOT NULL",
        )

        total_transiting = len(catalog)
        recent_discoveries = len(
            [row for row in catalog if row["disc_year"] and row["disc_year"] >= 2020]
        )

        # Count unique host stars
        unique_hosts = len(set([row["hostname"] for row in catalog if row["hostname"]]))

        # Get discovery year range
        years = [row["disc_year"] for row in catalog if row["disc_year"]]
        min_year = int(min(years)) if years else None
        max_year = int(max(years)) if years else None

        return {
            "total_transiting_planets": total_transiting,
            "unique_host_stars": unique_hosts,
            "recent_discoveries_2020_plus": recent_discoveries,
            "discovery_year_range": f"{min_year}-{max_year}"
            if min_year and max_year
            else "Unknown",
            "table_version": "ps (Planetary Systems)",
            "last_updated": "Live query from NASA Exoplanet Archive",
        }

    except Exception as e:
        return {
            "error": f"Could not retrieve statistics: {e}",
            "table_version": "ps (Planetary Systems)",
            "status": "Query failed",
        }


def test_exoplanet_query():
    """
    Test function to verify the exoplanet query is working with the new PS table
    """
    print("Testing NASA Exoplanet Archive PS table query...")

    try:
        # Test a simple query
        test_query = NasaExoplanetArchive.query_criteria(
            table="ps",
            select="pl_name,hostname,disc_year,ra,dec,pl_trandep,pl_orbper",
            where="disc_year > 2020 AND pl_trandep IS NOT NULL",
        )

        print(
            f"✓ Query successful! Found {len(test_query)} recent transiting exoplanets"
        )

        if len(test_query) > 0:
            sample = test_query[0]
            print(f"Sample entry: {sample['pl_name']} around {sample['hostname']}")
            print(f"  Discovery year: {sample['disc_year']}")
            print(f"  Coordinates: RA={sample['ra']:.4f}°, Dec={sample['dec']:.4f}°")
            print(f"  Transit depth: {sample['pl_trandep']:.6f}")
            if sample["pl_orbper"]:
                print(f"  Orbital period: {sample['pl_orbper']:.2f} days")

        return True

    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False


def search_known_exoplanets_by_coordinates(ra_deg, dec_deg, radius_arcsec=30.0):
    """
    Search for known exoplanets near given coordinates

    Args:
        ra_deg (float): Right Ascension in degrees
        dec_deg (float): Declination in degrees
        radius_arcsec (float): Search radius in arcseconds

    Returns:
        list: List of nearby exoplanets with details
    """
    try:
        # Query PS table for nearby transiting exoplanets
        catalog = NasaExoplanetArchive.query_criteria(
            table="ps",
            select="pl_name,hostname,ra,dec,pl_orbper,pl_trandep,sy_dist,pl_rade,pl_masse,disc_year",
            where="pl_trandep IS NOT NULL AND ra IS NOT NULL AND dec IS NOT NULL",
        )

        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        nearby_exoplanets = []

        for row in catalog:
            try:
                ep_coord = SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg, frame="icrs")
                sep = coord.separation(ep_coord).arcsecond

                if sep <= radius_arcsec:
                    exoplanet_info = {
                        "planet_name": str(row["pl_name"]),
                        "host_star": str(row["hostname"]),
                        "separation_arcsec": round(sep, 2),
                        "ra_deg": round(float(row["ra"]), 6),
                        "dec_deg": round(float(row["dec"]), 6),
                        "period_days": round(float(row["pl_orbper"]), 2)
                        if row["pl_orbper"]
                        else None,
                        "transit_depth": float(row["pl_trandep"])
                        if row["pl_trandep"]
                        else None,
                        "distance_pc": round(float(row["sy_dist"]), 1)
                        if row["sy_dist"]
                        else None,
                        "radius_earth": round(float(row["pl_rade"]), 2)
                        if row["pl_rade"]
                        else None,
                        "mass_earth": round(float(row["pl_masse"]), 2)
                        if row["pl_masse"]
                        else None,
                        "discovery_year": int(row["disc_year"])
                        if row["disc_year"]
                        else None,
                    }
                    nearby_exoplanets.append(exoplanet_info)

            except Exception:
                continue

        # Sort by separation
        nearby_exoplanets.sort(key=lambda x: x["separation_arcsec"])

        return nearby_exoplanets

    except Exception as e:
        print(f"Error searching for nearby exoplanets: {e}")
        return []


def get_recent_exoplanet_discoveries(years_back=5):
    """
    Get recently discovered transiting exoplanets

    Args:
        years_back (int): Number of years back to search

    Returns:
        list: List of recent exoplanet discoveries
    """
    try:
        from datetime import datetime

        current_year = datetime.now().year
        min_year = current_year - years_back

        catalog = NasaExoplanetArchive.query_criteria(
            table="ps",
            select="pl_name,hostname,disc_year,ra,dec,pl_orbper,pl_trandep,sy_dist",
            where=f"disc_year >= {min_year} AND pl_trandep IS NOT NULL AND ra IS NOT NULL AND dec IS NOT NULL",
        )

        recent_discoveries = []

        for row in catalog:
            discovery = {
                "planet_name": str(row["pl_name"]),
                "host_star": str(row["hostname"]),
                "discovery_year": int(row["disc_year"]) if row["disc_year"] else None,
                "ra_deg": round(float(row["ra"]), 6),
                "dec_deg": round(float(row["dec"]), 6),
                "period_days": round(float(row["pl_orbper"]), 2)
                if row["pl_orbper"]
                else None,
                "transit_depth": float(row["pl_trandep"])
                if row["pl_trandep"]
                else None,
                "distance_pc": round(float(row["sy_dist"]), 1)
                if row["sy_dist"]
                else None,
            }
            recent_discoveries.append(discovery)

        # Sort by discovery year (most recent first)
        recent_discoveries.sort(key=lambda x: x["discovery_year"] or 0, reverse=True)

        return recent_discoveries

    except Exception as e:
        print(f"Error getting recent discoveries: {e}")
        return []


if __name__ == "__main__":
    # Test the updated exoplanet query
    print("=" * 60)
    print("NASA EXOPLANET ARCHIVE PS TABLE TEST")
    print("=" * 60)

    if test_exoplanet_query():
        print("\n" + "=" * 60)
        print("EXOPLANET STATISTICS")
        print("=" * 60)

        stats = get_exoplanet_stats()
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        print("\n" + "=" * 60)
        print("RECENT DISCOVERIES (Last 3 years)")
        print("=" * 60)

        recent = get_recent_exoplanet_discoveries(3)
        if recent:
            print(f"Found {len(recent)} recent transiting exoplanets:")
            for i, discovery in enumerate(recent[:5]):  # Show first 5
                print(
                    f"{i+1}. {discovery['planet_name']} ({discovery['discovery_year']})"
                )
                print(f"   Host: {discovery['host_star']}")
                if discovery["period_days"]:
                    print(f"   Period: {discovery['period_days']:.2f} days")
                if discovery["distance_pc"]:
                    print(f"   Distance: {discovery['distance_pc']} pc")
                print()
        else:
            print("No recent discoveries found")

        print("=" * 60)
        print("✅ Migration to PS table completed successfully!")
        print("✅ PulseHunter can now access the latest exoplanet data")
        print("=" * 60)

    else:
        print("\n" + "=" * 60)
        print("❌ Migration test failed")
        print("Please check your internet connection and astroquery installation")
        print("=" * 60)
