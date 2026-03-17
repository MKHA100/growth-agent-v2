"""
Social micro-agent — Reddit OAuth2 (asyncpraw) + HN Algolia API.

Provides two async functions:
- search_reddit: searches all subreddits via OAuth2
- search_hn: searches HackerNews via the public Algolia API
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator

import asyncpraw
import httpx

from schemas.findings import Post

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------


def _make_reddit() -> asyncpraw.Reddit:
    return asyncpraw.Reddit(
        client_id=os.environ.get("REDDIT_CLIENT_ID", ""),
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
        user_agent=os.environ.get("REDDIT_USER_AGENT", "growth-intelligence/1.0"),
    )


async def search_reddit(query: str, limit: int = 15) -> list[Post]:
    """Search Reddit across all subreddits and return structured Post objects.

    Fetches the top `limit` posts and includes up to 5 top-level comments per post.

    Args:
        query: Search query string.
        limit: Maximum number of posts to return.

    Returns:
        List of Post objects with titles, body text, URLs, and comments.
    """
    posts: list[Post] = []

    reddit_id = os.environ.get("REDDIT_CLIENT_ID", "")
    reddit_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not reddit_id or not reddit_secret:
        # Return empty list if Reddit credentials are not configured
        logger.info("[Social] Reddit credentials not configured, skipping")
        return posts

    try:
        async with _make_reddit() as reddit:
            subreddit = await reddit.subreddit("all")
            async for submission in subreddit.search(query, limit=limit):
                comments: list[str] = []
                try:
                    await submission.load()
                    await submission.comments.replace_more(limit=0)
                    for comment in list(submission.comments)[:5]:
                        if hasattr(comment, "body"):
                            comments.append(comment.body)
                except Exception:  # noqa: BLE001
                    pass

                posts.append(
                    Post(
                        title=submission.title,
                        body=submission.selftext or "",
                        url=f"https://reddit.com{submission.permalink}",
                        comments=comments,
                        source="reddit",
                    )
                )
    except Exception:  # noqa: BLE001
        pass

    return posts


# ---------------------------------------------------------------------------
# HackerNews Algolia
# ---------------------------------------------------------------------------

_HN_API_BASE = "https://hn.algolia.com/api/v1"


async def search_hn(query: str, limit: int = 15) -> list[Post]:
    """Search HackerNews stories via the public Algolia API.

    Args:
        query: Search query string.
        limit: Maximum number of results.

    Returns:
        List of Post objects with titles and URLs.
    """
    posts: list[Post] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{_HN_API_BASE}/search",
                params={
                    "query": query,
                    "tags": "story",
                    "hitsPerPage": limit,
                },
            )
            response.raise_for_status()
            data = response.json()

            for hit in data.get("hits", []):
                posts.append(
                    Post(
                        title=hit.get("title", ""),
                        body=hit.get("story_text") or "",
                        url=hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                        comments=[],
                        source="hackernews",
                    )
                )
    except Exception:  # noqa: BLE001
        pass

    return posts
