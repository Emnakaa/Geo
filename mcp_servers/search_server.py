"""MCP Search Server — Google Maps (SerpApi), TripAdvisor (Apify), DuckDuckGo.

Run as a standalone MCP server:
    python mcp_servers/search_server.py

Tools exposed:
    google_maps_search(entity, location) -> dict
    tripadvisor_search(entity, location) -> dict   ← Apify epctex/tripadvisor-scraper
    ddg_search(query, max_results)       -> list[dict]
"""

import sys
import os
import re
import time
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SERPAPI_KEYS, APIFY_API_TOKENS

# Key pools — try ALL keys on every request, rotate until one works
_serpapi_idx  = 0
_apify_idx    = 0

def _serpapi_key() -> str:
    return SERPAPI_KEYS[_serpapi_idx] if _serpapi_idx < len(SERPAPI_KEYS) else ""

def _apify_token() -> str:
    return APIFY_API_TOKENS[_apify_idx] if _apify_idx < len(APIFY_API_TOKENS) else ""

def _rotate_serpapi() -> bool:
    global _serpapi_idx
    if _serpapi_idx + 1 < len(SERPAPI_KEYS):
        _serpapi_idx += 1
        print(f"[search_server] SerpAPI rotated to key{_serpapi_idx}")
        return True
    return False

def _rotate_apify() -> bool:
    global _apify_idx
    if _apify_idx + 1 < len(APIFY_API_TOKENS):
        _apify_idx += 1
        print(f"[search_server] Apify rotated to token{_apify_idx}")
        return True
    return False

def _serpapi_call(params: dict) -> dict:
    """Try every SerpAPI key until one returns a non-error response."""
    from serpapi import GoogleSearch
    global _serpapi_idx
    tried = 0
    total = len(SERPAPI_KEYS)
    while tried < total:
        params["api_key"] = _serpapi_key()
        try:
            data = GoogleSearch(params).get_dict()
        except Exception as e:
            err = str(e)
            print(f"[search_server] SerpAPI key{_serpapi_idx} exception: {err[:80]}")
            if _rotate_serpapi():
                tried += 1
                continue
            return {"error": err}
        error = data.get("error", "")
        if error:
            print(f"[search_server] SerpAPI key{_serpapi_idx} error: {error[:80]}")
            if _rotate_serpapi():
                tried += 1
                continue
            return data  # all keys failed
        return data  # success
    return {"error": "all_keys_exhausted"}

def _is_quota_error(msg: str) -> bool:
    return any(w in msg.lower() for w in
               ("quota", "credit", "limit", "402", "429", "exceeded", "plan",
                "actor-is-not-rented", "not rented", "free trial", "403",
                "run out", "out of searches"))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("search-server")


def _delay(lo=0.3, hi=0.8):
    time.sleep(random.uniform(lo, hi))


def _key_ok(k):
    return bool(k and len(k) > 10)


@mcp.tool()
def google_maps_search(entity: str, location: str = "Tunisia") -> dict:
    """Search Google Maps via SerpApi for a restaurant entity.

    Returns structured venue data: name, rating, review_count, address,
    phone, website, category, status.
    """
    result = {
        "gm_name": None, "gm_rating": None, "gm_review_count": None,
        "gm_address": None, "gm_phone": None, "gm_website": None,
        "gm_category": None, "status": "no_results",
    }
    if not _key_ok(_serpapi_key()):
        result["status"] = "no_key"
        return result

    try:
        places = []
        for query in [f"{entity} restaurant {location}", f"{entity} {location}"]:
            params = {
                "engine": "google_maps",
                "q": query,
                "type": "search",
            }
            data = _serpapi_call(params)
            if data.get("error"):
                result["status"] = "no_quota"
                return result
            places = data.get("local_results", [])
            if not places:
                single = data.get("place_results")
                if single:
                    places = [single]
            if places:
                break

        if not places:
            return result

        p = places[0]
        result["gm_name"]         = p.get("title")
        result["gm_rating"]       = p.get("rating")
        result["gm_review_count"] = p.get("reviews")
        result["gm_address"]      = p.get("address")
        result["gm_phone"]        = p.get("phone")
        result["gm_website"]      = p.get("website")

        types_raw = p.get("types", [])
        if types_raw and isinstance(types_raw[0], dict):
            result["gm_category"] = ", ".join(t.get("name", "") for t in types_raw)[:80]
        elif types_raw and isinstance(types_raw[0], str):
            result["gm_category"] = ", ".join(types_raw)[:80]
        else:
            result["gm_category"] = p.get("type", "")

        result["status"] = "success"

    except Exception as e:
        result["status"] = f"error: {str(e)[:80]}"

    _delay()
    return result


def _apify_run_sync(actor_id: str, input_data: dict, timeout: int = 90) -> list:
    """Run an Apify actor synchronously and return dataset items.

    Uses the run-sync-get-dataset-items endpoint — blocks until the actor
    finishes (or times out) and returns items directly.
    """
    import requests as _req
    if not _key_ok(_apify_token()):
        return []
    url = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
    try:
        resp = _req.post(
            url,
            json=input_data,
            params={"token": _apify_token(), "timeout": timeout, "memory": 1024},
            timeout=timeout + 20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        if _is_quota_error(str(e)) and _rotate_apify():
            return _apify_run_sync(actor_id, input_data, timeout)  # retry next token
        return [{"_apify_error": str(e)[:120]}]


def _ddg_find_tripadvisor_url(entity: str, location: str) -> str | None:
    """Use DuckDuckGo to find the TripAdvisor restaurant URL for an entity."""
    try:
        from ddgs import DDGS
        query = f"{entity} {location} tripadvisor restaurant"
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=8))
        for r in results:
            href = r.get("href", "")
            # Must be a Restaurant_Review URL (not Hotel, Attraction, etc.)
            if "tripadvisor.com/Restaurant_Review" in href:
                return href
        # Fallback: accept any tripadvisor.com link
        for r in results:
            href = r.get("href", "")
            if "tripadvisor.com/" in href and "Review" in href:
                return href
    except Exception:
        pass
    return None


@mcp.tool()
def tripadvisor_search(entity: str, location: str = "Tunisia") -> dict:
    """Search TripAdvisor for a restaurant entity via Apify maxcopell/tripadvisor.

    Strategy:
    1. Use DDG to find the TripAdvisor restaurant URL for the entity.
    2. Pass the URL as startUrls to the Apify actor to get structured data.

    Returns: {ta_rating, ta_review_count, ta_url, ta_ranking, ta_price_level, status}
    """
    result = {
        "ta_rating": None, "ta_review_count": None, "ta_url": None,
        "ta_ranking": None, "ta_price_level": None, "status": "no_results",
    }

    if not _key_ok(_apify_token()):
        result["status"] = "no_key"
        return result

    try:
        # Step 1: find TripAdvisor URL via DDG
        _delay(0.8, 1.5)
        ta_url = _ddg_find_tripadvisor_url(entity, location)
        if not ta_url:
            result["status"] = "no_ta_url"
            return result

        result["ta_url"] = ta_url

        # Step 2: scrape structured data via Apify startUrls
        items = _apify_run_sync(
            "maxcopell~tripadvisor",
            {
                "startUrls":          [{"url": ta_url}],
                "language":           "en",
                "currency":           "USD",
                "includeRestaurants": True,
                "includeHotels":      False,
                "includeAttractions": False,
                "includeNearbyResults": False,
                "includeAiReviewsSummary": False,
                "includePriceOffers": False,
                "includeTags":        False,
                "maxPhotosPerPlace":  0,
            },
            timeout=120,
        )
        items = [it for it in items if not it.get("_apify_error") and not it.get("error")]

        if not items:
            result["status"] = "found_url"
            return result

        it = items[0]

        ta_rating = it.get("rating") or it.get("overallRating")
        if ta_rating is not None:
            try: result["ta_rating"] = float(ta_rating)
            except (ValueError, TypeError): pass

        reviews = it.get("numberOfReviews") or it.get("reviewsCount")
        if reviews is not None:
            try: result["ta_review_count"] = int(str(reviews).replace(",", ""))
            except (ValueError, TypeError): pass

        result["ta_url"]         = it.get("webUrl") or ta_url
        result["ta_ranking"]     = it.get("rankingString") or it.get("ranking")
        result["ta_price_level"] = it.get("priceLevel") or it.get("priceRange")
        result["status"]         = "success" if result["ta_rating"] is not None else "found_url"

    except Exception as e:
        result["status"] = f"error: {str(e)[:80]}"

    _delay()
    return result


@mcp.tool()
def ddg_search(query: str, max_results: int = 5) -> list:
    """General DuckDuckGo text search.

    Returns list of {title, href, body} dicts.
    Useful for finding Instagram/Facebook handles and general entity info.
    """
    _delay(1.0, 2.0)
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return [{"error": str(e)[:80]}]


if __name__ == "__main__":
    mcp.run(transport="stdio")
