import astropy.units as u
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
        # Query the new PS (Planetary Systems) table
        catalog = NasaExoplanetArchive.query_criteria(
            table="ps",  # Updated from deprecated "exoplanets" table
            select="pl_name,ra,dec,pl_orbper,pl_trandep,hostname,sy_dist,pl_rade,pl_masse",
            where="pl_trandep IS NOT NULL AND ra IS NOT NULL AND dec IS NOT NULL",
        )
        
        print(f"Successfully loaded {len(catalog)} exoplanets from NASA Exoplanet Archive (PS table)")
        
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
        best_separation = float('inf')
        
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
                "host": str(best_match["hostname"]) if best_match["hostname"] else "Unknown",
                "planet": str(best_match["pl_name"]) if best_match["pl_name"] else "Unknown",
                "sep_arcsec": round(best_separation, 2),
                "period_days": (
                    round(float(best_match["pl_orbper"]), 2) 
                    if best_match["pl_orbper"] and not np.isnan(float(best_match["pl_orbper"])) 
                    else None
                ),
                "depth_ppm": (
                    round(float(best_match["pl_trandep"]) * 1e6, 1) 
                    if best_match["pl_trandep"] and not np.isnan(float(best_match["pl_trandep"])) 
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
            
            det["exo_match"] = exo_match
            match_count += 1
            
            print(f"Exoplanet match found: {exo_match['planet']} "
                  f"(separation: {best_separation:.1f}\")")
        else:
            det["exo_match"] = None

        matched.append(det)

    print(f"Exoplanet cross-matching complete: {match_count} matches found out of {len([d for d in detections if d.get('dimming')])} dimming events")
    return matched


def get_exoplanet_stats():
    """
    Get statistics about the exoplanet catalog
    
    Returns:
        dict: Statistics about available exoplanets
    """
    try:
        # Query basic statistics from the PS table
        catalog = NasaExoplanetArchive.query_criteria(
            table="ps",
            select="pl_name,pl_trandep,pl_orbper,disc_year",
            where="pl_trandep IS NOT NULL"
        )
        
        total_transiting = len(catalog)
        recent_discoveries = len([row for row in catalog if row["disc_year"] and row["disc_year"] >= 2020])
        
        return {
            "total_transiting_planets": total_transiting,
            "recent_discoveries_2020_plus": recent_discoveries,
            "table_version": "ps (Planetary Systems)",
            "last_updated": "Live query from NASA Exoplanet Archive"
        }
        
    except Exception as e:
        return {
            "error": f"Could not retrieve statistics: {e}",
            "table_version": "ps (Planetary Systems)",
            "status": "Query failed"
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
            select="pl_name,hostname,disc_year,ra,dec,pl_trandep",
            where="disc_year > 2020 AND pl_trandep IS NOT NULL"
        )
        
        print(f"✓ Query successful! Found {len(test_query)} recent transiting exoplanets")
        
        if len(test_query) > 0:
            sample = test_query[0]
            print(f"Sample entry: {sample['pl_name']} around {sample['hostname']}")
            print(f"  Discovery year: {sample['disc_year']}")
            print(f"  Coordinates: RA={sample['ra']:.4f}°, Dec={sample['dec']:.4f}°")
            print(f"  Transit depth: {sample['pl_trandep']:.6f}")
            
        return True
        
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False


# Add numpy import for NaN checking
import numpy as np

if __name__ == "__main__":
    # Test the updated exoplanet query
    print("=" * 50)
    print("NASA EXOPLANET ARCHIVE PS TABLE TEST")
    print("=" * 50)
    
    if test_exoplanet_query():
        print("\n" + "=" * 50)
        print("EXOPLANET STATISTICS")
        print("=" * 50)
        
        stats = get_exoplanet_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
        print("\n✓ Migration to PS table completed successfully!")
        print("✓ PulseHunter can now access the latest exoplanet data")
    else:
        print("\n✗ Migration test failed")
        print("Please check your internet connection and astroquery installation")