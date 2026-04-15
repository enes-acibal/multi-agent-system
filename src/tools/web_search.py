import httpx
from bs4 import BeautifulSoup

async def web_search(query: str, num_results: int = 5) -> list:
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"
    }
   
    async with httpx.AsyncClient() as client:
        try:
           response = await client.post(url, data={"q": query}, headers=headers, timeout=10.0)
           response.raise_for_status()
        except httpx.HTTPError as e:
           return f"Search failed: {str(e)}"
    
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for result in soup.select(".result")[:num_results]:
        title_tag = result.select_one(".result__title") 
        snippet_tag = result.select_one(".result__snippet")
        link_tag = result.select_one(".result__url")

        title = title_tag.get_text(strip=True) if title_tag else "No Title"
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No Snippet"
        link = link_tag.get_text(strip=True) if link_tag else "No Link"

        results.append(f"**{title}**\n{snippet}\nURL: {link}")

    if not results:
        return "No search results found."
    
    return "\n\n---\n\n".join(results)

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current information. Use this when you need "
        "facts, recent events, documentation, or any information you don't "
        "have in your training data. Returns titles, snippets, and URLs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and concise."
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (1-10).",
                "default": 5
            }
        },
        "required": ["query"]
    }
}