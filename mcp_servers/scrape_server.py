"""MCP Scrape Server — Apify (Instagram, Facebook), website social extraction.

Run as a standalone MCP server:
    python mcp_servers/scrape_server.py

Tools exposed:
    scrape_instagram(handle)          -> dict
    scrape_facebook(handle)           -> dict
    extract_website_socials(url)      -> dict
"""

import sys
import os
import re
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import APIFY_API_TOKENS

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scrape-server")

_apify_idx = 0


def _apify_token() -> str:
    return APIFY_API_TOKENS[_apify_idx] if _apify_idx < len(APIFY_API_TOKENS) else ""


def _rotate_apify() -> bool:
    global _apify_idx
    if _apify_idx + 1 < len(APIFY_API_TOKENS):
        _apify_idx += 1
        print(f"[scrape_server] Apify rotated to token {_apify_idx + 1}")
        return True
    return False


def _apify_run(actor_id: str, input_data: dict, timeout: int = 90) -> list:
    """Run an Apify actor, rotating tokens on quota/auth errors."""
    from apify_client import ApifyClient
    tried = 0
    total = len(APIFY_API_TOKENS)
    while tried < total:
        try:
            client = ApifyClient(_apify_token())
            run = client.actor(actor_id).call(run_input=input_data, timeout_secs=timeout)
            if run is None:
                if _rotate_apify():
                    tried += 1
                    continue
                return []
            items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
            return items
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ("quota", "limit", "unauthorized", "payment", "403", "402")):
                if _rotate_apify():
                    tried += 1
                    continue
            return []
    return []


def _delay(lo=0.3, hi=0.8):
    time.sleep(random.uniform(lo, hi))


def _key_ok(k):
    return bool(k and len(k) > 10)


@mcp.tool()
def scrape_instagram(handle: str | None = None) -> dict:
    """Scrape an Instagram profile via Apify.

    Returns: ig_followers, ig_posts, ig_engagement_rate, ig_bio, ig_scraper.
    """
    result = {
        "ig_followers": 0, "ig_posts": 0, "ig_engagement_rate": 0.0,
        "ig_bio": "", "ig_scraper": "none",
    }
    if not handle or not _key_ok(_apify_token()):
        return result

    try:
        items = _apify_run("apify/instagram-profile-scraper",
                           {"usernames": [handle], "resultsLimit": 1}, timeout=300)
        if items:
            r = items[0]
            result["ig_followers"]       = r.get("followersCount", 0)
            result["ig_posts"]           = r.get("postsCount", 0)
            result["ig_engagement_rate"] = r.get("engagementRate", 0.0)
            result["ig_bio"]             = (r.get("biography") or "")[:200]
            result["ig_scraper"]         = "apify"

    except Exception as e:
        result["ig_scraper"] = f"error: {str(e)[:80]}"

    _delay()
    return result


@mcp.tool()
def scrape_facebook(handle: str | None = None) -> dict:
    """Scrape a Facebook page via Apify.

    Returns: fb_page_likes, fb_post_engagement, fb_scraper.
    """
    result = {"fb_page_likes": 0, "fb_post_engagement": 0.0, "fb_scraper": "none"}
    if not handle or not _key_ok(_apify_token()):
        return result

    try:
        items = _apify_run("apify/facebook-pages-scraper", {
            "startUrls": [{"url": f"https://www.facebook.com/{handle}"}],
            "resultsLimit": 10,
            "maxPostComments": 0,
            "maxReviews": 0,
        }, timeout=180)
        if items:
            page  = items[0]
            posts = page.get("posts", []) or []
            result["fb_page_likes"] = page.get("likes", 0) or 0
            if posts:
                total = sum(
                    (p.get("likes", 0) or 0) + (p.get("comments", 0) or 0) +
                    (p.get("shares", 0) or 0) for p in posts
                )
                result["fb_post_engagement"] = round(total / len(posts), 1)
            result["fb_scraper"] = "apify"

    except Exception as e:
        result["fb_scraper"] = f"error: {str(e)[:80]}"

    _delay()
    return result


@mcp.tool()
def extract_website_socials(website_url: str | None = None) -> dict:
    """Fetch a website's HTML and extract Instagram and Facebook handles.

    Returns: website_ig, website_fb (handles as strings, or None).
    """
    socials = {"website_ig": None, "website_fb": None}
    if not website_url or not isinstance(website_url, str) or website_url.strip().lower() in ("null", "none", ""):
        return socials

    try:
        import httpx
        r = httpx.get(
            website_url, timeout=10, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        html = r.text

        ig = re.search(
            r'href=["\']?https?://(?:www\.)?instagram\.com/([A-Za-z0-9_.]{3,30})', html
        )
        fb = re.search(
            r'href=["\']?https?://(?:www\.)?facebook\.com/([A-Za-z0-9_.]{3,50})', html
        )

        if ig:
            socials["website_ig"] = ig.group(1).lower()
        if fb:
            socials["website_fb"] = fb.group(1).lower()

    except Exception as e:
        socials["error"] = str(e)[:80]

    return socials


if __name__ == "__main__":
    mcp.run(transport="stdio")
