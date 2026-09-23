import re
from types import SimpleNamespace

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url
from dotenv import load_dotenv


load_dotenv()


# ─────────────────────────────────────────────
# GEMINI MODEL
# Only Writer + Critic use Gemini
# ─────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    max_retries=3
)


# ─────────────────────────────────────────────
# DIRECT SEARCH AGENT
# No Gemini API call
# ─────────────────────────────────────────────

class DirectSearchAgent:

    def invoke(self, data):

        messages = data.get("messages", [])

        if not messages:
            query = ""
        else:
            message = messages[-1]

            if isinstance(message, tuple):
                query = message[1]
            else:
                query = str(message)

        # Remove unnecessary instruction text
        prefix = "Find recent, reliable and detailed information about:"

        if prefix in query:
            query = query.split(prefix, 1)[1].strip()

        try:
            result = web_search.invoke(query)

        except Exception as e:
            result = f"Search failed: {str(e)}"

        return {
            "messages": [
                SimpleNamespace(content=result)
            ]
        }


def build_search_agent():
    return DirectSearchAgent()


# ─────────────────────────────────────────────
# DIRECT READER AGENT
# Extract first URL and scrape directly
# No Gemini API call
# ─────────────────────────────────────────────

class DirectReaderAgent:

    def invoke(self, data):

        messages = data.get("messages", [])

        if not messages:
            text = ""
        else:
            message = messages[-1]

            if isinstance(message, tuple):
                text = message[1]
            else:
                text = str(message)

        # Find URLs from search results
        urls = re.findall(
            r'https?://[^\s<>"\']+',
            text
        )

        if not urls:
            return {
                "messages": [
                    SimpleNamespace(
                        content="No valid URL found in search results."
                    )
                ]
            }

        # Use first search result
        url = urls[0].rstrip(".,);]")

        try:
            content = scrape_url.invoke(url)

        except Exception as e:
            content = f"Could not scrape URL: {str(e)}"

        return {
            "messages": [
                SimpleNamespace(content=content)
            ]
        }


def build_reader_agent():
    return DirectReaderAgent()


# ─────────────────────────────────────────────
# WRITER CHAIN
# Uses Gemini
# ─────────────────────────────────────────────

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert research writer.

Write clear, structured, factual and insightful
research reports.

Use only the research information provided.

Do not invent facts, statistics, URLs or sources.

If some information is uncertain or unavailable,
state that clearly.
"""
    ),

    (
        "human",
        """
Write a detailed research report on the topic below.

Topic:
{topic}

Research Gathered:
{research}

Structure the report as:

# Introduction

Give a clear overview of the topic.

# Key Findings

Provide at least 3 detailed and well-explained
key findings based on the supplied research.

# Conclusion

Summarize the most important findings.

# Sources

List the URLs present in the supplied research.

Do not create fake URLs.

Write in a professional and readable style.
"""
    ),
])


writer_chain = (
    writer_prompt
    | llm
    | StrOutputParser()
)


# ─────────────────────────────────────────────
# CRITIC CHAIN
# Uses Gemini
# ─────────────────────────────────────────────

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a professional research critic.

Evaluate the report based on:

- factual clarity
- structure
- usefulness
- source quality
- completeness

Be concise, constructive and specific.
"""
    ),

    (
        "human",
        """
Review the research report below.

Report:

{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
...
"""
    ),
])


critic_chain = (
    critic_prompt
    | llm
    | StrOutputParser()
)