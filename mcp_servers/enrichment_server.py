"""MCP Enrichment Server — Wikidata entity validation.

Run as a standalone MCP server:
    python mcp_servers/enrichment_server.py

Tools exposed:
    wikidata_lookup(entity)  -> dict  — structured entity facts (no API key)
"""

import sys
import os
import re
import time
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("enrichment-server")


def _delay(lo=0.5, hi=1.2):
    time.sleep(random.uniform(lo, hi))


# Food/restaurant Wikidata instance types (Q-ids)
_FOOD_QIDS = {
    "Q11707",   # restaurant
    "Q1299668", # fast food restaurant
    "Q4830453", # business (broad)
    "Q570116",  # tourist attraction
    "Q1414983", # café
    "Q2360219", # snack bar
    "Q37836",   # bistro
    "Q2044700", # brasserie
    "Q846875",  # patisserie
}

# Wikidata properties for restaurant entities
_WD_PROPS = {
    "P17":   "country",
    "P131":  "location",
    "P18":   "image",
    "P856":  "official_website",
    "P856":  "official_website",
    "P571":  "founded",
    "P576":  "dissolved",
    "P7153": "significant_place",
}


@mcp.tool()
def wikidata_lookup(entity: str) -> dict:
    """Look up a restaurant entity on Wikidata.

    Uses the Wikidata Action API (no API key required).
    Returns structured facts: entity_type, country, founded, official_website,
    wikipedia_link, wikidata_id, is_food_venue.

    This supplements Wikipedia by providing:
    - Explicit entity type classification (restaurant vs hotel vs person)
    - Structured coordinates and country
    - Official website from structured data
    """
    result = {
        "wd_id": None,
        "wd_label": None,
        "wd_entity_type": None,
        "wd_country": None,
        "wd_founded": None,
        "wd_official_website": None,
        "wd_wikipedia_link": None,
        "is_food_venue": False,
        "status": "not_found",
    }

    try:
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": "GEO-Pipeline/1.0 (research)"})

        # Step 1: Search for the entity by label
        search_url = "https://www.wikidata.org/w/api.php"
        search_params = {
            "action":   "wbsearchentities",
            "search":   entity,
            "language": "fr",
            "type":     "item",
            "limit":    5,
            "format":   "json",
        }
        resp = session.get(search_url, params=search_params, timeout=10)
        resp.raise_for_status()
        search_data = resp.json()

        candidates = search_data.get("search", [])
        if not candidates:
            # Try English search as fallback
            search_params["language"] = "en"
            resp = session.get(search_url, params=search_params, timeout=10)
            candidates = resp.json().get("search", [])

        if not candidates:
            return result

        # Use first candidate
        item_id = candidates[0]["id"]
        result["wd_id"]    = item_id
        result["wd_label"] = candidates[0].get("label", entity)

        # Step 2: Fetch entity details
        entity_url = "https://www.wikidata.org/wiki/Special:EntityData/{}.json".format(item_id)
        resp = session.get(entity_url, timeout=10)
        resp.raise_for_status()
        entity_data = resp.json()

        claims = entity_data.get("entities", {}).get(item_id, {}).get("claims", {})

        # Instance of (P31) — determines entity type
        p31_claims = claims.get("P31", [])
        instance_qids = set()
        for c in p31_claims:
            qid = c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
            if qid:
                instance_qids.add(qid)
                # Get label for this QID
                label_resp = session.get(search_url, params={
                    "action": "wbgetentities",
                    "ids": qid,
                    "props": "labels",
                    "languages": "en|fr",
                    "format": "json",
                }, timeout=8)
                label_data = label_resp.json()
                labels = label_data.get("entities", {}).get(qid, {}).get("labels", {})
                label_en = labels.get("en", {}).get("value", "")
                label_fr = labels.get("fr", {}).get("value", "")
                result["wd_entity_type"] = label_fr or label_en

        result["is_food_venue"] = bool(instance_qids & _FOOD_QIDS)

        # Country (P17)
        p17 = claims.get("P17", [])
        if p17:
            country_qid = p17[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
            if country_qid:
                c_resp = session.get(search_url, params={
                    "action": "wbgetentities",
                    "ids": country_qid,
                    "props": "labels",
                    "languages": "en|fr",
                    "format": "json",
                }, timeout=8)
                c_data = c_resp.json()
                c_labels = c_data.get("entities", {}).get(country_qid, {}).get("labels", {})
                result["wd_country"] = (
                    c_labels.get("fr", {}).get("value")
                    or c_labels.get("en", {}).get("value")
                )

        # Official website (P856)
        p856 = claims.get("P856", [])
        if p856:
            result["wd_official_website"] = (
                p856[0].get("mainsnak", {}).get("datavalue", {}).get("value")
            )

        # Founded (P571)
        p571 = claims.get("P571", [])
        if p571:
            time_val = (
                p571[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("time", "")
            )
            # Format: +1990-00-00T00:00:00Z
            m = re.match(r"[+-](\d{4})", time_val)
            if m:
                result["wd_founded"] = int(m.group(1))

        # Wikipedia sitelink
        sitelinks = entity_data.get("entities", {}).get(item_id, {}).get("sitelinks", {})
        for lang in ("frwiki", "enwiki", "arwiki"):
            sl = sitelinks.get(lang, {})
            if sl:
                title = sl.get("title", "").replace(" ", "_")
                wiki_lang = lang.replace("wiki", "")
                result["wd_wikipedia_link"] = (
                    f"https://{wiki_lang}.wikipedia.org/wiki/{title}"
                )
                break

        result["status"] = "found"

    except Exception as e:
        result["status"]        = f"error: {str(e)[:80]}"
        result["is_food_venue"] = False

    _delay()
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
