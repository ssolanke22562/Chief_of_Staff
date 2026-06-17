"""
engine.py - Email engine using the Gmail MCP server (MCP Python SDK).

Uses the official MCP Python SDK (mcp.client.session.ClientSession) to communicate
with the Gmail MCP server. Requests are batched to avoid Gmail API quota limits,
and quota errors are caught with a graceful fallback to basic thread info.
"""

import asyncio
import re
import os
import sys
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import AsyncIterator, Optional

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from mcp.types import CallToolResult, TextContent

# Add Gmail-MCP-Server directory to sys.path so we can import triage
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Gmail-MCP-Server"))
from triage import triage_inbox

# Path to the Gmail MCP server executable
MCP_SERVER_PARAMS = StdioServerParameters(
    command="node",
    args=["C:\\Users\\sarth\\cheif_of_staff\\Gmail-MCP-Server\\dist\\index.js"],
)

# ---------- quota / rate-limit helpers ----------

QUOTA_ERROR_KEYWORDS = [
    "quota", "rate limit", "429", "too many requests",
    "resource exhausted", "user-rate limit",
]

BATCH_SIZE = 5          # how many read_email calls per batch
BATCH_DELAY_SECONDS = 1.0  # delay between batches


def _is_quota_error(result: CallToolResult) -> bool:
    """Check if a tool call result indicates a quota / rate-limit error."""
    if result.isError:
        # Combine all text content to search for quota keywords
        text = " ".join(
            c.text for c in result.content
            if isinstance(c, TextContent)
        ).lower()
        for keyword in QUOTA_ERROR_KEYWORDS:
            if keyword in text:
                return True
    return False


# ---------- helpers for extracting tool results ----------

def _text_from_result(result: CallToolResult) -> str:
    """Extract all text content from a tool call result, joined."""
    parts = []
    for c in result.content:
        if isinstance(c, TextContent):
            parts.append(c.text)
        elif isinstance(c, str):
            parts.append(c)
    return "\n".join(parts)


# ---------- MCP session context manager ----------

@asynccontextmanager
async def _mcp_session() -> AsyncIterator[ClientSession]:
    """
    Async context manager that provides an initialized MCP client session.

    Usage:
        async with _mcp_session() as session:
            result = await session.call_tool(...)
    """
    async with stdio_client(MCP_SERVER_PARAMS) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


# ---------- fetching threads ----------

def _parse_email_block(text: str) -> Optional[dict]:
    """
    Parse a single email text block returned by search_emails.

    Format example:
        ID: 19ec66b54a9af8ab
        Subject: New device login detected in your Flipkart account
        From: "Flipkart.com" <noreply@rmo.flipkart.com>
        Date: Sun, 14 Jun 2026 13:56:19 +0000
    """
    lines = text.strip().split("\n")
    info = {}
    for line in lines:
        line = line.strip()
        if line.startswith("ID:"):
            info["thread_id"] = line[len("ID:"):].strip()
        elif line.startswith("Subject:"):
            info["subject"] = line[len("Subject:"):].strip()
        elif line.startswith("From:"):
            info["sender"] = line[len("From:"):].strip()
        elif line.startswith("Date:"):
            info["date"] = line[len("Date:"):].strip()

    if "thread_id" not in info:
        return None

    info.setdefault("subject", "(no subject)")
    info.setdefault("sender", "(unknown)")
    info.setdefault("date", "(unknown)")
    info.setdefault("snippet", info["subject"])

    return info


def _parse_search_results(text: str) -> list[dict]:
    """Parse the search_emails text output into a list of basic thread dicts."""
    threads = []
    email_blocks = text.strip().split("\n\n")
    for email_text in email_blocks:
        email_text = email_text.strip()
        if not email_text:
            continue
        info = _parse_email_block(email_text)
        if info:
            threads.append(info)

    # De-duplicate by thread_id, keeping first occurrence
    seen = set()
    unique = []
    for t in threads:
        if t["thread_id"] not in seen:
            seen.add(t["thread_id"])
            unique.append(t)
    return unique


async def _search_inbox(session: ClientSession, max_results: int = 20) -> list[dict]:
    """Search the inbox and return basic thread info."""
    result = await session.call_tool("search_emails", {
        "query": "in:inbox",
        "maxResults": max_results,
    })
    if result.isError:
        raise RuntimeError(f"search_emails failed: {_text_from_result(result)}")
    text = _text_from_result(result)
    return _parse_search_results(text)


def _extract_snippet(full_text: str, max_chars: int = 150) -> str:
    """Extract a readable snippet from the raw email body."""
    lines = full_text.split("\n")
    body_lines = []
    in_body = False
    for line in lines:
        if not in_body:
            stripped = line.strip()
            if stripped == "":
                in_body = True
            else:
                if any(stripped.startswith(h) for h in
                       ["Thread ID:", "Subject:", "From:", "To:", "Date:"]):
                    continue
                if stripped and len(body_lines) > 2:
                    body_lines.append(stripped)
        else:
            stripped = line.strip()
            if stripped:
                body_lines.append(stripped)

    if not body_lines:
        body_lines = [
            l.strip() for l in lines
            if l.strip() and not any(
                l.strip().startswith(h) for h in
                ["Thread ID:", "Subject:", "From:", "To:", "Date:", "ID:"]
            )
        ]

    snippet = " ".join(body_lines)
    snippet = re.sub(r'https?://\S+', '', snippet)
    snippet = re.sub(r'\s+', ' ', snippet).strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "..."
    return snippet if snippet else "(no preview)"


async def _read_email_with_quota_handling(
    session: ClientSession, message_id: str
) -> Optional[str]:
    """
    Read a single email via read_email.  Returns the full text on success,
    or None if a quota error occurs (fallback signal) or any other error.
    """
    try:
        result = await session.call_tool("read_email", {
            "messageId": message_id,
        })
    except Exception:
        # Treat any transport-level exception as a fallback
        return None

    if _is_quota_error(result):
        return None  # signal quota exhaustion

    if result.isError:
        # Some non-quota error — fall back to basic info for this thread
        return None

    return _text_from_result(result)


async def fetch_threads(max_results: int = 20) -> list[dict]:
    """
    Fetch the last N inbox threads using the Gmail MCP server.

    Uses the official MCP Python SDK.  Only one call (search_emails) is
    made, so no batching is needed.  For enhanced snippets, see
    fetch_threads_enhanced().

    Returns a list of thread dicts sorted by date descending.
    """
    async with _mcp_session() as session:
        threads = await _search_inbox(session, max_results)
        return threads[:max_results]


async def fetch_threads_enhanced(max_results: int = 20) -> list[dict]:
    """
    Fetch threads with full snippet by reading each email individually.

    Uses batching to avoid Gmail API quota limits:
      - Processes emails in groups of BATCH_SIZE (default 5)
      - Waits BATCH_DELAY_SECONDS (default 1.0 s) between batches
      - Catches quota errors and falls back to basic info for remaining threads

    Returns a list of thread dicts (deduplicated by thread_id).
    """
    async with _mcp_session() as session:
        # Step 1: Search for inbox messages
        basic_info = await _search_inbox(session, max_results)
        if not basic_info:
            return []

        # Step 2: Batch-read full email details
        enriched = []

        for i in range(0, len(basic_info), BATCH_SIZE):
            batch = basic_info[i:i + BATCH_SIZE]
            tasks = [
                _read_email_with_quota_handling(session, info["thread_id"])
                for info in batch
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results — None / Exception means fallback to basic info
            for info, full_text_or_none in zip(batch, results):
                if isinstance(full_text_or_none, Exception):
                    full_text = None
                else:
                    full_text = full_text_or_none

                if full_text:
                    snippet = _extract_snippet(full_text)
                    enriched.append({
                        "thread_id": info["thread_id"],
                        "sender": info["sender"],
                        "subject": info["subject"],
                        "snippet": snippet,
                        "date": info["date"],
                    })
                else:
                    # Fallback: use basic info (subject as snippet)
                    enriched.append({
                        "thread_id": info["thread_id"],
                        "sender": info["sender"],
                        "subject": info["subject"],
                        "snippet": info.get("snippet", info["subject"]),
                        "date": info["date"],
                    })

            # Delay between batches to stay within quota
            if i + BATCH_SIZE < len(basic_info):
                await asyncio.sleep(BATCH_DELAY_SECONDS)

        # De-duplicate by thread_id
        seen = set()
        unique = []
        for t in enriched:
            if t["thread_id"] not in seen:
                seen.add(t["thread_id"])
                unique.append(t)

        return unique[:max_results]


# ---------- synchronous wrappers for compatibility ----------

def fetch_threads_sync(max_results: int = 20) -> list[dict]:
    """Synchronous wrapper around fetch_threads."""
    return asyncio.run(fetch_threads(max_results))


def fetch_threads_enhanced_sync(max_results: int = 20) -> list[dict]:
    """Synchronous wrapper around fetch_threads_enhanced."""
    return asyncio.run(fetch_threads_enhanced(max_results))


# ---------- formatting ----------

# ─── Unicode box-drawing constants ──────────────────────────────
_H_RULE       = "━" * 72           # thick horizontal rule
_H_RULE_LIGHT = "─" * 72           # thin horizontal rule
_H_RULE_DASH  = "─" * 70           # separator inside a card

# Priority display configuration
_PRIORITY_CFG = {
    "urgent": {
        "emoji": "🔴",
        "label": "URGENT",
        "color_start": "\033[91m",  # red
        "color_end": "\033[0m",
    },
    "needs reply": {
        "emoji": "🟡",
        "label": "NEEDS REPLY",
        "color_start": "\033[93m",  # yellow
        "color_end": "\033[0m",
    },
    "fyi": {
        "emoji": "🔵",
        "label": "FYI",
        "color_start": "\033[94m",  # blue
        "color_end": "\033[0m",
    },
    "ignore": {
        "emoji": "⚫",
        "label": "IGNORE",
        "color_start": "\033[90m",  # grey
        "color_end": "\033[0m",
    },
    "unknown": {
        "emoji": "⚪",
        "label": "UNKNOWN",
        "color_start": "\033[97m",
        "color_end": "\033[0m",
    },
}

_PRIORITY_ORDER = ["urgent", "needs reply", "fyi", "ignore", "unknown"]


def _pcf(priority: str, text: str) -> str:
    """Wrap *text* in ANSI colour codes for the given priority level."""
    cfg = _PRIORITY_CFG.get(priority, _PRIORITY_CFG["unknown"])
    return f"{cfg['color_start']}{text}{cfg['color_end']}"


def _format_date_human(raw_date: str) -> str:
    """Try to parse an email Date header and return a friendlier string."""
    try:
        from email.utils import parsedate_to_datetime
        from datetime import timezone
        dt = parsedate_to_datetime(raw_date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone()
        return local.strftime("%a, %b %d · %I:%M %p")
    except Exception:
        return raw_date


def format_digest(results: list[dict]) -> None:
    """
    Print a beautiful, detailed terminal digest of triaged inbox threads.

    Threads are sorted by priority (URGENT → NEEDS REPLY → FYI → IGNORE)
    and each message is displayed as a richly formatted card with the
    sender, subject, snippet, AI category, and a bold reason.
    """
    today = date.today().strftime("%B %d, %Y")

    print()
    print(_H_RULE)
    print(f"  📬  INBOX DIGEST  —  {today}  |  {len(results)} messages")
    print(_H_RULE)
    print()

    # Group by priority
    groups: dict[str, list[dict]] = {}
    for r in results:
        p = r.get("priority", "unknown")
        groups.setdefault(p, []).append(r)

    group_count = 0
    for prio in _PRIORITY_ORDER:
        group = groups.get(prio)
        if not group:
            continue
        group_count += 1
        if group_count > 1:
            print()  # blank line between groups

        cfg = _PRIORITY_CFG.get(prio, _PRIORITY_CFG["unknown"])

        # ── Priority group header ──────────────────────────────────────
        print(f"  {cfg['emoji']}  {_pcf(prio, cfg['label'])}  ─  {len(group)} message{'s' if len(group) > 1 else ''}")
        print(_H_RULE_LIGHT)

        # ── Individual email cards ─────────────────────────────────────
        for idx, msg in enumerate(group, start=1):
            sender   = msg.get("sender", "(unknown)")
            subject  = msg.get("subject", "(no subject)")
            snippet  = msg.get("snippet", "")
            category = msg.get("category", "")
            reason   = msg.get("reason", "")
            raw_date = msg.get("date", "")
            nice_date = _format_date_human(raw_date)

            print()
            print(f"     #{idx}  ╭{'─' * 66}╮")

            # Sender + Date line
            print(f"        From      │  {sender}")
            if nice_date:
                print(f"        Date      │  {nice_date}")

            # Subject line
            print(f"        Subject   │  {subject}")

            # Category (if available)
            if category:
                print(f"        Category  │  {category}")

            # Snippet (truncated nicely)
            snippet_display = snippet if snippet else "(no preview)"
            if len(snippet_display) > 120:
                snippet_display = snippet_display[:120].rsplit(" ", 1)[0] + "…"
            if snippet_display:
                print(f"        Snippet   │  {snippet_display}")

            # Reason — displayed prominently
            print(f"     ╰{'─' * 66}╯")
            if reason:
                print(f"        💬  {_pcf(prio, 'Reason')}  │  {reason}")
            else:
                print(f"        💬  {_pcf(prio, 'Reason')}  │  (no analysis available)")
            print(f"     {' ' * 70}")

    # ── Footer ────────────────────────────────────────────────────────
    print(_H_RULE)
    summary_parts = []
    for prio in _PRIORITY_ORDER:
        group = groups.get(prio)
        if group:
            cfg = _PRIORITY_CFG.get(prio, _PRIORITY_CFG["unknown"])
            summary_parts.append(f"{cfg['emoji']} {cfg['label']}: {len(group)}")
    print(f"  Summary:  {'  |  '.join(summary_parts)}")
    print(_H_RULE)
    print()


# ---------- CLI entry point ----------

if __name__ == "__main__":
    print("📡  Fetching last 20 inbox threads using Gmail MCP server...", end=" ", flush=True)
    threads = fetch_threads_sync(20)
    print(f"\r✅  Found {len(threads)} threads in inbox.\n")

    print("📡  Fetching enhanced snippets (with batching)...", end=" ", flush=True)
    enhanced = fetch_threads_enhanced_sync(20)
    print(f"\r✅  Enriched {len(enhanced)} threads with full content.\n")

    print("🧠  Running AI triage (categorising + prioritising)...", end=" ", flush=True)
    results = triage_inbox(enhanced)
    print(f"\r✅  Triage complete — {len(results)} messages sorted by priority.\n")

    # ── Display the beautiful, detailed output ────────────────────────
    format_digest(results)