"""Specialist referral directory + Google Maps helpers.

Live results need a Google Places API (New) key:

    set GOOGLE_MAPS_API_KEY=...

Enable "Places API (New)" in the Google Cloud project and set the environment
variable before starting the server. The legacy Places API cannot be enabled
on new projects, so this module calls the current Text Search endpoint.

Without a key the app serves a curated Pune directory of real clinics that
offer ADHD / child-psychiatry assessments, clearly marked as a sample that
must be verified by phone. The map itself uses the public Google Maps embed,
so no key is required for the map view.
"""

from __future__ import annotations

import logging
import math
import os
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_FIELD_MASK = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.rating",
        "places.userRatingCount",
        "places.internationalPhoneNumber",
        "places.nationalPhoneNumber",
        "places.websiteUri",
        "places.googleMapsUri",
    ]
)

CURATED_PUNE = [
    {
        "id": "mansa",
        "name": "Dr. Vidya Ganapathy",
        "specialty": "Child & adolescent psychiatrist",
        "clinic": "Mansa Clinic",
        "address": "Parihar Chowk, Raviraj Apartment, DP Road, Aundh, Pune 411007",
        "city": "Pune",
        "phone": "+919175951157",
        "phone_display": "+91 91759 51157",
        "website": "https://mansaclinicpune.com",
        "tags": ["ADHD", "Child psychiatry", "NIMHANS-trained"],
        "lat": 18.5590,
        "lng": 73.8070,
        "note": "Child and adolescent psychiatry; ADHD assessments and family counselling.",
    },
    {
        "id": "pathfinder",
        "name": "Dr. Neville Misquitta",
        "specialty": "Child psychiatrist",
        "clinic": "Pathfinder Clinic",
        "address": "S-5, 2nd Floor, A-Wing, Destination Centre, Magarpatta City, Hadapsar, Pune 411028",
        "city": "Pune",
        "phone": "+917410781580",
        "phone_display": "+91 74107 81580",
        "website": "https://www.pathfinderclinic.com/child-psychiatrist-pune",
        "tags": ["ADHD", "Child psychiatry", "Learning disorders"],
        "lat": 18.5080,
        "lng": 73.9260,
        "note": "Child psychiatry assessments for attention, behaviour, and school difficulties.",
    },
    {
        "id": "smart",
        "name": "Smart Clinic — Child Guidance Unit",
        "specialty": "Child & adolescent psychiatry",
        "clinic": "Smart Clinic",
        "address": "209, 2nd Floor, Insignia Building, Wadia College Chowk, near Ruby Hall Metro Station, Pune 411001",
        "city": "Pune",
        "phone": "+917410051105",
        "phone_display": "+91 74100 51105",
        "website": "https://www.smartclinicpune.com",
        "tags": ["ADHD", "Ages 3–18", "Assessment"],
        "lat": 18.5280,
        "lng": 73.8770,
        "note": "Child guidance unit for attention, school, and emotional-regulation concerns.",
    },
    {
        "id": "mindful",
        "name": "Dr. (Maj) Sachin Dhanwade",
        "specialty": "Consultant psychiatrist",
        "clinic": "The Mindful Psychiatry Clinic",
        "address": "Shop 203, 2nd Floor, Gagan Avencia, Grand Road, Kharadi, Pune 411014",
        "city": "Pune",
        "phone": "+918329022350",
        "phone_display": "+91 83290 22350",
        "website": "https://www.mindfulpsychiatryclinics.in",
        "tags": ["ADHD", "Child & adolescent", "Teleconsultation"],
        "lat": 18.5510,
        "lng": 73.9400,
        "note": "Treats ADHD and child/adolescent psychiatric concerns; in-person and online.",
    },
    {
        "id": "psycognia",
        "name": "Psycognia Neuropsychiatry Clinic",
        "specialty": "Neuropsychiatry & child psychiatry",
        "clinic": "Psycognia",
        "address": "32, Viman Nagar Road, Clover Park, Viman Nagar, Pune 411014",
        "city": "Pune",
        "phone": "+919022867925",
        "phone_display": "+91 90228 67925",
        "website": "https://psycogniaclinic.com",
        "tags": ["ADHD", "Child & adolescent", "Neuropsychiatry"],
        "lat": 18.5670,
        "lng": 73.9140,
        "note": "ADHD and attention problems, child/adolescent psychiatry, and neuropsychiatry.",
    },
    {
        "id": "dmh",
        "name": "Deenanath Mangeshkar Hospital — Psychiatry",
        "specialty": "Child psychiatry department",
        "clinic": "Deenanath Mangeshkar Hospital",
        "address": "Near Mhatre Bridge, Erandwane, Pune 411004",
        "city": "Pune",
        "phone": "+912040151100",
        "phone_display": "+91 20 4015 1100",
        "website": "https://www.dmhospital.org/specialty-details/PSYCHIATRY",
        "tags": ["ADHD", "Hospital OPD", "Child psychiatry"],
        "lat": 18.5050,
        "lng": 73.8310,
        "note": "Hospital psychiatry department with child psychiatry services for ADHD.",
    },
    {
        "id": "bharati",
        "name": "Bharati Hospital — Psychiatry",
        "specialty": "Psychiatry department",
        "clinic": "Bharati Hospital & Research Centre",
        "address": "Bharati Vidyapeeth Campus, Dhankawadi, Pune 411043",
        "city": "Pune",
        "phone": "+912040555555",
        "phone_display": "+91 20 4055 5555",
        "website": "https://bharatihospital.com/center-of-excellence/psychiatry",
        "tags": ["ADHD", "Hospital OPD", "Psychiatry"],
        "lat": 18.4650,
        "lng": 73.8550,
        "note": "Hospital psychiatry department; call the helpline for the psychiatry OPD.",
    },
    {
        "id": "rubyhall",
        "name": "Ruby Hall Clinic — Psychiatry",
        "specialty": "Psychiatry & mental health",
        "clinic": "Ruby Hall Clinic",
        "address": "40 Sassoon Road, Sangamvadi, Pune 411001",
        "city": "Pune",
        "phone": "+912066455100",
        "phone_display": "+91 20 6645 5100",
        "website": "https://rubyhall.com/broad-specialities/psychiatry",
        "tags": ["ADHD", "Hospital OPD", "Counselling"],
        "lat": 18.5330,
        "lng": 73.8770,
        "note": "Psychiatry department with psychiatrists and clinical psychologists.",
    },
]


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def _maps_embed(query: str) -> str:
    return f"https://maps.google.com/maps?q={quote_plus(query)}&z=12&output=embed"


def _maps_link(query: str) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"


def _enrich(doctor: dict, lat: float | None, lng: float | None) -> dict:
    item = dict(doctor)
    item["maps_query"] = (
        f"{item.get('name', '')} {item.get('clinic', '')} {item.get('address', '')}"
    ).strip()
    item["maps_url"] = _maps_link(item["maps_query"])
    if (
        lat is not None
        and lng is not None
        and item.get("lat") is not None
        and item.get("lng") is not None
    ):
        item["distance_km"] = round(
            _haversine_km(lat, lng, item["lat"], item["lng"]), 1
        )
    return item


def _google_places(
    query: str,
    lat: float | None,
    lng: float | None,
    api_key: str,
    limit: int = 8,
) -> list[dict]:
    """Search live clinics with Places API (New) Text Search."""
    body: dict = {
        "textQuery": query,
        "maxResultCount": max(1, min(limit, 20)),
    }
    if lat is not None and lng is not None:
        body["locationBias"] = {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": 15000.0,
            }
        }

    response = requests.post(
        TEXT_SEARCH_URL,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": PLACES_FIELD_MASK,
        },
        json=body,
        timeout=15,
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Places API returned HTTP {response.status_code} with a non-JSON body"
        ) from exc

    if not response.ok:
        error = payload.get("error") if isinstance(payload, dict) else None
        message = (error or {}).get("message") or f"HTTP {response.status_code}"
        raise RuntimeError(f"Places API: {message}")

    doctors: list[dict] = []
    for result in (payload.get("places") or [])[:limit]:
        location = result.get("location") or {}
        name = (result.get("displayName") or {}).get("text") or "Specialist"
        phone = result.get("internationalPhoneNumber") or ""
        phone_display = result.get("nationalPhoneNumber") or phone

        item = {
            "id": result.get("id") or name,
            "name": name,
            "specialty": "Child psychiatrist / ADHD clinic",
            "clinic": name,
            "address": result.get("formattedAddress", ""),
            "city": "",
            "phone": phone,
            "phone_display": phone_display,
            "website": result.get("websiteUri", ""),
            "tags": ["ADHD / child psychiatry", "Google Places"],
            "rating": result.get("rating"),
            "reviews": result.get("userRatingCount"),
            "lat": location.get("latitude"),
            "lng": location.get("longitude"),
            "note": "Live Google Places result. Confirm specialty and timings by phone.",
        }
        enriched = _enrich(item, lat, lng)
        if result.get("googleMapsUri"):
            enriched["maps_url"] = result["googleMapsUri"]
        doctors.append(enriched)

    return doctors


def get_doctors(
    city: str = "Pune",
    lat: float | None = None,
    lng: float | None = None,
) -> dict:
    """Return specialist listings plus a map preview."""
    city = (city or "").strip()
    api_key = (
        os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_PLACES_API_KEY") or ""
    ).strip()
    live_error: str | None = None
    live_empty = False

    display_city = city or ("your area" if lat is not None else "Pune")
    map_query = (
        f"ADHD child psychiatrist near {lat},{lng}"
        if lat is not None and lng is not None and not city
        else f"ADHD child psychiatrist in {display_city}"
    )

    if api_key:
        try:
            query = (
                f"ADHD child psychiatrist in {display_city}"
                if city
                else "ADHD child psychiatrist"
            )
            doctors = _google_places(query, lat, lng, api_key)
            if doctors:
                return {
                    "city": display_city,
                    "source": "google_places",
                    "doctors": doctors,
                    "map_embed": _maps_embed(map_query),
                    "map_link": _maps_link(map_query),
                    "notice": (
                        "Live Google Places results. Confirm the clinician's "
                        "specialty, fees, and availability directly."
                    ),
                }
            live_empty = True
        except Exception as exc:  # pragma: no cover - network dependent
            live_error = str(exc)
            logger.warning("Live Google Places lookup failed: %s", exc)

    if live_error:
        fallback_note = (
            " Live Google Places lookup failed, so the fallback list is shown; "
            "check the server key and that Places API (New) is enabled."
        )
    elif live_empty:
        fallback_note = (
            " Google Places returned no matches, so the fallback list is shown."
        )
    elif not api_key:
        fallback_note = " Add GOOGLE_MAPS_API_KEY for live results in any city."
    else:
        fallback_note = ""

    near_pune = False
    if lat is not None and lng is not None:
        near_pune = _haversine_km(lat, lng, 18.5204, 73.8567) <= 80.0

    if city.lower() in {"", "pune", "pimpri", "pcmc", "pune, maharashtra"} or near_pune:
        doctors = [_enrich(item, lat, lng) for item in CURATED_PUNE]
        if lat is not None and lng is not None:
            doctors.sort(key=lambda item: item.get("distance_km", 999.0))
        return {
            "city": "Pune",
            "source": "curated",
            "doctors": doctors,
            "map_embed": _maps_embed("ADHD child psychiatrist in Pune"),
            "map_link": _maps_link("ADHD child psychiatrist in Pune"),
            "notice": (
                "Curated sample of real Pune clinics, not a live proximity "
                "search. Call before visiting." + fallback_note
            ),
        }

    return {
        "city": display_city,
        "source": "maps_search",
        "doctors": [],
        "map_embed": _maps_embed(map_query),
        "map_link": _maps_link(map_query),
        "notice": (
            "No curated directory for this city. Use the Google Map search "
            "below, and confirm the clinician's ADHD/child-psychiatry "
            "credentials by phone." + fallback_note
        ),
    }
