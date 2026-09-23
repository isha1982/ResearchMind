from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic."""

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No search results found."

        out = []

        for r in results:
            out.append(
                f"Title: {r.get('title', '')}\n"
                f"URL: {r.get('href', '')}\n"
                f"Snippet: {r.get('body', '')[:300]}\n"
            )

        return "\n----\n".join(out)

    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""

    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if not text:
            return "No readable content found on this page."

        return text[:5000]

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"