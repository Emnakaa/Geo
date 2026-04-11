"""MCP Wikipedia Server — existence check for en/fr Wikipedia pages.

Run as a standalone MCP server:
    python mcp_servers/wiki_server.py

Tools exposed:
    wikipedia_lookup(entity) -> dict
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("wiki-server")

# Keywords that confirm the page is about a food/hospitality business
_FOOD_KEYWORDS = {
    "restaurant", "café", "cafe", "brasserie", "bistro", "bar", "hotel",
    "cuisine", "food", "dining", "eatery", "menu", "chef", "culinary",
    "tunisian", "moroccan", "mediterranean", "french cuisine",
    "michelin", "gastronomic", "patisserie",
}


def _is_food_related(page) -> bool:
    """Return True only if the Wikipedia page is clearly about a food/hospitality venue."""
    text = (page.summary or "").lower()
    cats = " ".join(page.categories.keys()).lower() if hasattr(page, "categories") else ""
    combined = text[:500] + " " + cats
    return any(kw in combined for kw in _FOOD_KEYWORDS)


@mcp.tool()
def wikipedia_lookup(entity: str) -> dict:
    """Check if a restaurant/business entity has a Wikipedia page (English or French).

    Only returns has_wikipedia=True if the page is clearly about a food or
    hospitality business — avoids false positives from unrelated namesakes.

    Returns: {has_wikipedia: bool, found_in: 'en'|'fr'|None, url: str|None}
    """
    result = {"has_wikipedia": False, "found_in": None, "url": None}
    try:
        import wikipediaapi
        for lang in ["en", "fr"]:
            wiki = wikipediaapi.Wikipedia(
                user_agent="GEO-Pipeline/1.0", language=lang
            )
            for title in [entity, entity.title()]:
                page = wiki.page(title)
                if page.exists() and _is_food_related(page):
                    result["has_wikipedia"] = True
                    result["found_in"]      = lang
                    result["url"]           = page.fullurl
                    return result
    except Exception as e:
        result["error"] = str(e)[:80]

    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
