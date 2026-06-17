"""
triage.py - Email triage engine.

Uses a fast, deterministic rule-based classifier (*no* external API calls)
so it never hits quota limits.  A Gemini-powered path is available as an
optional fallback for users who prefer AI-based categorisation.

The rule-based classifier runs in <50 µs per email and handles:
  - urgency detection (deadlines, ASAP, security alerts, etc.)
  - category tagging (work, personal, promotions, spam)
  - reason generation
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Rule-based classifier  (primary — no API calls, no rate limits)
# ---------------------------------------------------------------------------

# ── Sender blacklists / whitelists ─────────────────────────────────────────
_SPAM_DOMAINS = {
    "mailchimp.com", "sendgrid.net", "mailgun.org", "sendinblue.com",
    "constantcontact.com", "benchmarkemail.com", "getresponse.com",
    "aweber.com", "icontact.com", "verticalresponse.com",
    "emlnk.com", "emailsrvr.com", "bronto.com", "list-manage.com",
    "amazonses.com", "sparkpostmail.com",
}

_PROMO_SENDERS = {
    "noreply@", "no-reply@", "donotreply@", "do-not-reply@",
}

_PROMO_DOMAINS = {
    "amazon.com", "amazon.in", "flipkart.com", "myntra.com",
    "nykaa.com", "ajio.com", "meesho.com", "tatacliq.com",
    "snapdeal.com", "shopify.com", "etsy.com", "ebay.com", "walmart.com",
    "target.com", "bestbuy.com", "homedepot.com",
}

_WORK_KEYWORDS = {
    "meeting", "deadline", "project", "sprint", "standup", "retro",
    "quarterly", "review", "performance", "timesheet", "invoice",
    "payroll", "salary", "hiring", "interview", "candidate", "resume",
    "budget", "forecast", "kpi", "okr", "pipeline", "deployment",
    "release", "build", "ci/cd", "staging", "production", "code review",
    "pull request", "merge", "commit", "bug", "hotfix", "incident",
}

_URGENT_KEYWORDS = {
    "urgent", "asap", "critical", "deadline today", "overdue",
    "action required", "security alert", "unauthorized", "breach",
    "password reset", "account locked", "suspended", "vulnerability",
    "expiring", "expired", "past due", "final notice", "immediate",
    "emergency", "outage", "downtime", "sev1", "sev2", "p0", "p1",
    "compliance", "audit", "legal", "lawsuit",
}

_SUBJECT_OVERRIDE_URGENT = {
    "security alert", "password reset", "account locked", "urgent",
    "critical", "final notice", "past due", "overdue", "action required",
    "immediate action", "compliance", "audit", "legal notice",
}


def _extract_domain(sender: str) -> str:
    """Extract the domain part from an email sender string."""
    m = re.search(r'@([\w.-]+)', sender)
    return m.group(1).lower() if m else ""


def _extract_email(sender: str) -> str:
    """Extract the email address from sender string."""
    m = re.search(r'<([^>]+)>', sender)
    if m:
        return m.group(1).lower()
    m = re.search(r'[\w.+-]+@[\w.-]+', sender)
    return m.group(0).lower() if m else sender.lower()


def _contains_any(text: str, keywords: set) -> bool:
    """Check if *text* contains any of the given keywords (case-insensitive)."""
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            return True
    return False


def rule_based_triage(sender: str, subject: str, snippet: str) -> dict:
    """
    Triage a single email using deterministic rules.
    
    No API calls — runs in <50 µs.
    
    Returns
    -------
    dict with keys: priority, category, reason
    """
    # Normalise inputs
    sender_lower = sender.lower()
    subject_lower = subject.lower()
    snippet_lower = snippet.lower()
    combined = f"{subject_lower} {snippet_lower}"
    domain = _extract_domain(sender)
    email_addr = _extract_email(sender)

    # ── 1. Spam detection ──────────────────────────────────────────────
    if any(domain.endswith(spam_domain) for spam_domain in _SPAM_DOMAINS):
        return {
            "priority": "ignore",
            "category": "spam",
            "reason": f"Sender domain {domain} is a known mass-mailing platform.",
        }

    # ── 2. Urgent override (subject-based) ─────────────────────────────
    for urgent_subj in _SUBJECT_OVERRIDE_URGENT:
        if urgent_subj in subject_lower:
            return {
                "priority": "urgent",
                "category": "security" if any(w in subject_lower for w in ["security", "unauthorized", "breach", "password", "locked", "suspended"]) else "alert",
                "reason": f"Subject contains '{urgent_subj}' — requires immediate attention.",
            }

    # ── 3. Urgent keyword check in full text ───────────────────────────
    if _contains_any(combined, _URGENT_KEYWORDS):
        matched_kw = next(kw for kw in _URGENT_KEYWORDS if kw in combined)
        return {
            "priority": "urgent",
            "category": "alert",
            "reason": f"Contains urgent keyword '{matched_kw}'.",
        }

    # ── 4. Promotions / commercial ─────────────────────────────────────
    is_promo_sender = any(email_addr.startswith(p) for p in _PROMO_SENDERS)
    is_promo_domain = domain in _PROMO_DOMAINS

    if is_promo_sender or is_promo_domain:
        promos_detected = []
        if is_promo_sender:
            promos_detected.append("no-reply sender pattern")
        if is_promo_domain:
            promos_detected.append(f"commercial domain ({domain})")
        return {
            "priority": "ignore",
            "category": "promotions",
            "reason": f"Promotional email detected via {'; '.join(promos_detected)}.",
        }

    # ── 5. Work / professional ─────────────────────────────────────────
    if _contains_any(combined, _WORK_KEYWORDS):
        matched_kw = next(kw for kw in _WORK_KEYWORDS if kw in combined)
        return {
            "priority": "needs reply",
            "category": "work",
            "reason": f"Work-related keyword '{matched_kw}' found.",
        }

    # ── 6. Newsletters / subscriptions ─────────────────────────────────
    newsletter_indicators = [
        "unsubscribe", "newsletter", "weekly digest", "daily digest",
        "you're receiving this", "to stop receiving", "manage preferences",
        "email preferences", "view in browser", "sent to",
    ]
    if _contains_any(combined, set(newsletter_indicators)):
        return {
            "priority": "fyi",
            "category": "newsletter",
            "reason": "Subscription-type content detected (unsubscribe link / newsletter pattern).",
        }

    # ── 7. Social / notifications ──────────────────────────────────────
    social_domains = {
        "linkedin.com", "linkedin", "facebook.com", "facebookmail.com",
        "twitter.com", "x.com", "instagram.com", "redditmail.com",
        "youtube.com", "tiktok.com", "snapchat.com", "pinterest.com",
        "github.com", "gitlab.com", "bitbucket.org", "slack.com",
        "discord.com", "teams.microsoft.com", "zoom.us",
        "notion.com", "notifications@", "atlassian.com", "jira.com",
        "confluence.com", "trello.com", "asana.com", "monday.com",
    }
    if domain in social_domains or any(sd in email_addr for sd in social_domains):
        return {
            "priority": "fyi",
            "category": "notification",
            "reason": f"Notification from {domain}.",
        }

    # ── 8. Personal email ──────────────────────────────────────────────
    personal_indicators = {
        "hey", "hi", "hello", "dear", "thanks", "thank you", 
        "family", "friend", "invitation", "party", "weekend plans",
        "how are you", "thinking of you", "miss you", "love",
        "photos", "pictures", "get together", "catch up",
    }
    if _contains_any(combined, personal_indicators):
        return {
            "priority": "needs reply",
            "category": "personal",
            "reason": "Personal correspondence pattern detected.",
        }

    # ── 9. Default ─────────────────────────────────────────────────────
    # Check if it's a calendar / automated
    if any(w in combined for w in ["calendar", "invite", "event", "reminder"]):
        return {
            "priority": "fyi",
            "category": "calendar",
            "reason": "Calendar event or reminder.",
        }

    return {
        "priority": "fyi",
        "category": "other",
        "reason": f"Email from {domain}; no urgent or work-related keywords detected.",
    }


# ---------------------------------------------------------------------------
#  Gemini AI classifier (optional — can be toggled with USE_GEMINI env var)
# ---------------------------------------------------------------------------

_USE_GEMINI = os.environ.get("USE_GEMINI", "").lower() in ("1", "true", "yes")

if _USE_GEMINI:
    from dotenv import load_dotenv
    import time
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted

    _triage_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_triage_dir, ".env"))

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    _model = genai.GenerativeModel("gemini-2.5-flash")

    # Gemini retry constants
    _MAX_RETRIES = 3
    _BASE_RETRY_DELAY = 10.0
    _INTER_THREAD_DELAY = 6.0  # seconds between calls

    def _call_gemini(prompt: str) -> str:
        last_exc = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = _model.generate_content(prompt)
                return response.text
            except ResourceExhausted as exc:
                last_exc = exc
                delay = _BASE_RETRY_DELAY * (1.5 ** (attempt - 1))
                logger.warning("Gemini ResourceExhausted (attempt %d/%d). Sleeping %.0f s.",
                               attempt, _MAX_RETRIES, delay)
                time.sleep(delay)
            except Exception as exc:
                raise exc
        logger.error("Gemini exhausted after %d retries. Falling back to rule-based.", _MAX_RETRIES)
        return ""

    def _gemini_triage(sender: str, subject: str, snippet: str) -> dict:
        prompt = f"""You are an email assistant that triages incoming emails based on the sender, subject, and a snippet of the email content.
The goal is to categorize the email into one of the following categories: "Work", "Personal", "Spam", "Promotions", or "Other".
Based on the provided information, determine the most appropriate category for the email.

Sender: {sender}
Subject: {subject}
Snippet: {snippet}

Respond with this exact format:
Priority: <urgent | needs reply | fyi | ignore>
Category: <One short tag like: meeting-request, follow-up, newsletter, billing, job-app, social etc.>
Reason: <One statement explaining why you categorized the email this way>
"""
        text = _call_gemini(prompt)
        if not text:
            return rule_based_triage(sender, subject, snippet)
        result = {"priority": None, "category": None, "reason": None}
        for line in text.strip().split("\n"):
            ls = line.strip()
            if ls.startswith("Priority:"):
                result["priority"] = ls.replace("Priority:", "").strip().lower()
            elif ls.startswith("Category:"):
                result["category"] = ls.replace("Category:", "").strip().lower()
            elif ls.startswith("Reason:"):
                result["reason"] = ls.replace("Reason:", "").strip()
        if not result["priority"]:
            return rule_based_triage(sender, subject, snippet)
        return result


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def triage_thread(sender: str, subject: str, snippet: str) -> dict:
    """
    Triage a single email thread.

    Uses rule-based classifier by default.
    Set ``USE_GEMINI=true`` in .env to enable Gemini AI fallback.
    
    Never raises quota/resource exhausted errors.
    """
    if _USE_GEMINI:
        return _gemini_triage(sender, subject, snippet)
    return rule_based_triage(sender, subject, snippet)


def triage_inbox(threads: list) -> list:
    """
    Triage a list of email thread dicts and return them sorted by priority.

    Each thread dict must contain ``sender``, ``subject``, and ``snippet``
    keys.

    Runs synchronously.  Uses fast rule-based classifier by default,
    so no rate-limit delays are needed.
    """
    triaged = []
    for idx, thread in enumerate(threads):
        logger.info(
            "Triaging thread %d/%d: %s – %s",
            idx + 1, len(threads), thread.get("sender", "?"), thread.get("subject", "?"),
        )
        label = triage_thread(
            sender=thread["sender"],
            subject=thread["subject"],
            snippet=thread["snippet"],
        )
        triaged.append({**thread, **label})

    # Sort by descending urgency
    priority_order = {"urgent": 0, "needs reply": 1, "fyi": 2, "ignore": 3}
    triaged.sort(key=lambda x: priority_order.get(x.get("priority"), 4))
    return triaged


# ---------------------------------------------------------------------------
#  CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    sample_threads = [
        {
            "sender": "boss@company.com",
            "subject": "URGENT: Project Update Needed Today",
            "snippet": "Please send me the latest update on the project by EOD.",
        },
        {
            "sender": "newsletter@medium.com",
            "subject": "Your Weekly Reading List",
            "snippet": "Here are some articles you might find interesting this week.",
        },
        {
            "sender": "recruiter@startup.io",
            "subject": "Job Opportunity at Top Tech Company",
            "snippet": "We have an exciting job opportunity that matches your profile.",
        },
        {
            "sender": "noreply@flipkart.com",
            "subject": "Your order has been shipped!",
            "snippet": "Your package is on its way. Track your order here.",
        },
        {
            "sender": "security@bank.com",
            "subject": "Security Alert: New device login",
            "snippet": "A new device was used to log into your account.",
        },
        {
            "sender": "friend@personal.com",
            "subject": "Weekend plans?",
            "snippet": "Hey! Are you free this weekend to catch up?",
        },
    ]

    results = triage_inbox(sample_threads)
    for r in results:
        print(
            f"  Priority: {r['priority']:12s}  Category: {r['category']:15s}  "
            f"Sender: {r['sender']:30s}  Subject: {r['subject']}"
        )
        print(f"    Reason: {r['reason']}")
        print()