from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.coordinates import SkyCoord
import astropy.units as u


def match_transits_with_exoplanets(detections, radius_arcsec=5.0):
    try:
        catalog = NasaExoplanetArchive.query_criteria(
            table="exoplanets",
            select="pl_name,ra,dec,pl_orbper,pl_trandep,hostname",
            where="pl_trandep IS NOT NULL AND ra IS NOT NULL AND dec IS NOT NULL"
        )
    except Exception as e:
        print("Failed to fetch exoplanet archive:", e)
        return detections

    matched = []
    for det in detections:
        if det.get("ra_deg") is None or det.get("dec_deg") is None or not det.get("dimming"):
            matched.append(det)
            continue

        coord = SkyCoord(ra=det["ra_deg"] * u.deg, dec=det["dec_deg"] * u.deg, frame="icrs")
        found_match = False
        for row in catalog:
            ep_coord = SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg)
            sep = coord.separation(ep_coord).arcsecond
            if sep <= radius_arcsec:
                det["exo_match"] = {
                    "host": row["hostname"],
                    "planet": row["pl_name"],
                    "sep_arcsec": round(sep, 2),
                    "period_days": round(row["pl_orbper"], 2) if row["pl_orbper"] else None,
                    "depth_ppm": round(row["pl_trandep"] * 1e6, 1) if row["pl_trandep"] else None
                }
                found_match = True
                break

        if not found_match:
            det["exo_match"] = None

        matched.append(det)

    return matched

