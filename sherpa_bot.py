import os
import json
import time
import re
import random
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import gradio as gr
from openai import OpenAI
import tweepy
import feedparser
import schedule
import requests
from bs4 import BeautifulSoup
import html2text
import httpx
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

load_dotenv()

REPLY_STATE_FILE = "reply_state.json"
MAX_DAILY_REPLIES = 3
MAX_FETCH = 50
MAX_AGE_HOURS = 23
MAX_BACKLOG = 20   # store at most this many tweet IDs for tomorrow

MORK_CORE_URL = os.getenv("MORK_CORE_URL", "http://localhost:8787").rstrip("/")
USE_OPENAI = False
def _get_core_url():
    u = (os.getenv("MORK_CORE_URL") or "").strip().rstrip("/")
    if not u:
        return "http://localhost:8787"
    if "<" in u or "IP-OF" in u.upper():
        print(f"⚠ MORK_CORE_URL is a placeholder: '{u}'. Please set a real URL.")
        return "http://localhost:8787"
    return u

MORK_CORE_URL = _get_core_url()

def _core_base_url() -> str:
    """
    Returns a safe/usable base URL for Mork Core.
    Fixes common mistakes like literally having '<ip-of-mork-core>' in the env var,
    and strips trailing slashes.
    """
    raw = (MORK_CORE_URL or "").strip().strip('"').strip("'")
    raw = raw.rstrip("/")

    # Common placeholder mistake: "http://<ip-of-mork-core>:8787"
    if "<" in raw or ">" in raw:
        print(f"⚠ MORK_CORE_URL looks like a placeholder: {raw!r}. Falling back to http://localhost:8787")
        return "http://localhost:8787"

    # If user accidentally pasted a URL-encoded placeholder (%3c ... %3e)
    if "%3c" in raw.lower() or "%3e" in raw.lower():
        print(f"⚠ MORK_CORE_URL looks URL-encoded/invalid: {raw!r}. Falling back to http://localhost:8787")
        return "http://localhost:8787"

    if not re.match(r"^https?://", raw):
        print(f"⚠ MORK_CORE_URL missing scheme: {raw!r}. Prepending http://")
        raw = "http://" + raw

    return raw

def core_reflect(timeout=20) -> bool:
    try:
        r = requests.post(f"{MORK_CORE_URL}/brain/reflect", json={}, timeout=timeout)
        if r.ok:
            return True
        print(f"⚠ core_reflect bad status {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        print(f"⚠ core_reflect failed: {e}")
        return False

def core_compose_payload(payload: dict, timeout=10) -> str:
    """
    Compose a tweet via Mork Core.

    Supports BOTH styles:
      - NEW: POST /x/compose with JSON payload (recommended)
      - OLD: GET /x/compose?mode=observation|edge|reflection (fallback)

    payload examples:
      {"kind":"feed","title":...,"text":...,"url":...,"maxChars":260}
      {"kind":"meme","memeName":"when-i-see-slippage.png","maxChars":260}
      {"kind":"arb","maxChars":260}
      {"kind":"observation","maxChars":260}
      {"kind":"reflection","maxChars":260}
    """
    base = _core_base_url()
    payload = payload or {}
    payload.setdefault("maxChars", 260)

    # 1) Try POST (new style)
    try:
        r = requests.post(f"{base}/x/compose", json=payload, timeout=timeout)
        if r.ok:
            j = r.json() if "application/json" in (r.headers.get("content-type") or "") else {}
            out = (j.get("tweet") or "").strip()
            return out
        print(f"⚠ core_compose bad status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"⚠ core_compose POST failed: {e}")

    # 2) Fallback to GET (old style)
    try:
        mode = (payload.get("mode") or payload.get("kind") or "observation")
        r = requests.get(f"{base}/x/compose", params={"mode": mode}, timeout=min(5, timeout))
        if not r.ok:
            print(f"⚠ core_compose GET bad status {r.status_code}: {r.text[:200]}")
            return ""
        j = r.json() if "application/json" in (r.headers.get("content-type") or "") else {}
        return (j.get("tweet") or "").strip()
    except Exception as e:
        print(f"⚠ core_compose GET failed: {e}")
        return ""
def _wrap_280(s: str, max_len: int = 260) -> str:
    # Preserve newlines for tweet formatting, but collapse repeated spaces
    s = (s or "").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s if len(s) <= max_len else (s[: max_len - 1].rstrip() + "…")

def morkcore_edge_line() -> str:
    """
    OPTIONAL helper. Only used if you explicitly want to pull raw edge lines.
    If you're routing all tone through Core, you usually don't need this.
    """
    try:
        base = _core_base_url()
        r = requests.get(
            f"{base}/memory/query",
            params={"q": "edge=", "limit": 10},
            timeout=3,
        )
        if not r.ok:
            return ""
        data = r.json()
        items = data.get("items", []) if isinstance(data, dict) else []
        for it in items:
            c = (it or {}).get("content", "")
            if isinstance(c, str) and "| edge=" in c:
                return c.strip()
    except Exception:
        pass
    return ""

def compose_observation_from_core(max_len: int = 260) -> str:
    """
    Ask Mork Core to compose an observation tweet.
    All voice/tone comes from Core (and its prime directive).
    Never throws.
    """
    try:
        out = core_compose_payload({"kind": "observation", "maxChars": max_len}, timeout=8)
        if out:
            return _wrap_280(out, max_len)
    except Exception:
        pass

    # Hard fallback (should be rare): keep neutral, avoid the repetitive "edge/vibes" stuff here.
    return _wrap_280("System check: my thoughts are quiet right now. Give me a moment to warm the coals.", max_len)


# ----------------------------------------
# Relationship memory (local file) — keep
# ----------------------------------------

def _load_relationships():
    if os.path.exists(REL_FILE):
        try:
            with open(REL_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_relationships(rel):
    try:
        with open(REL_FILE, "w", encoding="utf-8") as f:
            json.dump(rel, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _extract_topics(text: str, max_topics=5):
    t = re.sub(r"http\S+", "", (text or "").lower())
    words = re.findall(r"[a-z0-9']{3,}", t)

    stop = {
        "this","that","with","have","just","like","your","youre","about","from","they","them","what",
        "when","then","been","were","there","here","will","would","could","into","over","under","than",
        "more","some","much","very","really","also","because","while","where","their","them","these",
        "those","cant","dont","didnt","doesnt","isnt","arent","you","and","the","for"
    }

    freq = defaultdict(int)
    for w in words:
        if w in stop:
            continue
        freq[w] += 1

    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:max_topics]]

def update_relationship(author_id: str, username: str, text: str, engagement_score: float = 0.0):
    rel = _load_relationships()
    key = author_id or username or "unknown"
    now = datetime.now(timezone.utc).isoformat()

    entry = rel.get(key, {
        "author_id": author_id,
        "username": username,
        "trust": 0.0,
        "topics": [],
        "last_interaction": None,
        "interactions": 0,
    })

    entry["author_id"] = author_id or entry.get("author_id")
    entry["username"] = username or entry.get("username")
    entry["last_interaction"] = now
    entry["interactions"] = int(entry.get("interactions", 0)) + 1

    bump = 0.05 + min(0.05, float(engagement_score) / 100.0)
    entry["trust"] = max(-1.0, min(1.0, float(entry.get("trust", 0.0)) + bump))

    new_topics = _extract_topics(text)
    topics = list(entry.get("topics", []))
    for t in new_topics:
        if t not in topics:
            topics.insert(0, t)
    entry["topics"] = topics[:15]

    rel[key] = entry
    _save_relationships(rel)
    return entry

def _load_reply_state():
    if os.path.exists(REPLY_STATE_FILE):
        with open(REPLY_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_post_day": None, "replied_today": 0, "since_id": None, "backlog": []}

def _save_reply_state(s):
    with open(REPLY_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)

def _parse_iso_z(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def _age_hours(created_at: str) -> float:
    return (datetime.now(timezone.utc) - _parse_iso_z(created_at)).total_seconds() / 3600.0

def _score(public_metrics: dict) -> float:
    likes = public_metrics.get("like_count", 0)
    replies = public_metrics.get("reply_count", 0)
    rts = public_metrics.get("retweet_count", 0)
    return likes + 2 * replies + 0.5 * rts

def _prune_backlog(backlog: list) -> list:
    seen = set()
    kept = []
    for item in backlog:
        tid = item.get("tweet_id")
        ts = item.get("created_at")
        if not tid or not ts or tid in seen:
            continue
        if _age_hours(ts) < MAX_AGE_HOURS:
            seen.add(tid)
            kept.append(item)
    return kept[:MAX_BACKLOG]
# Inside class TwitterBot
def _normalize_story(self, story, subject=None):
    """
    Ensure the story is a dict with keys: title, preview, url.
    Accepts strings or partial dicts and normalizes them.
    """
    if isinstance(story, str):
        return {
            "title": (subject or "Update").title(),
            "preview": story,
            "url": "#",
        }

    if isinstance(story, dict):
        return {
            "title": (story.get("title") or (subject or "Update").title()).strip(),
            "preview": (story.get("preview") or story.get("summary") or story.get("text") or "").strip(),
            "url": (story.get("url") or story.get("link") or "#").strip(),
        }

    return {"title": (subject or "Update").title(), "preview": "", "url": "#"}

def _normalize_subject(self, subj):
    if not subj:
        return "news"

    s = str(subj).strip()
    if s.lower() in {"surprise_all", "surprise-all", "random", "any", "*"}:
        pool = list(getattr(self, "feed_config", {}).keys()) or list(RSS_FEEDS.keys())
        pool = [k for k in pool if k.lower() not in {"surprise_all", "surprise-all", "random"}]
        return random.choice(pool) if pool else "news"

    return s

def get_random_story_all(self, subject=None):
    """
    Unified story getter used by the scheduler/queue.
    Tries a few likely single/bulk fetchers if they exist.
    If nothing is available, returns a small synthetic story so cadence never stalls.
    """
    # ✅ normalize subject so "surprise_all" actually fans out
    subject = self._normalize_subject(subject or getattr(self, "scheduler_subject", None) or "news")

    def norm(st):
        return self._normalize_story(st, subject=subject)

    # 1) Try single-story fetchers (if you’ve implemented any)
    single_candidates = [
        "get_news_story",
        "get_new_story_from_feeds",
        "fetch_next_story",
        "fetch_story",
        "get_story_from_rss",
        "get_random_story_from",   # ✅ include the helper below if you keep it
    ]
    for name in single_candidates:
        if hasattr(self, name):
            try:
                s = getattr(self, name)(subject)
                if s:
                    ns = norm(s)
                    if ns["title"] or ns["preview"]:
                        print(f"[get_random_story_all] Using {name}()")
                        return ns
            except Exception as e:
                print(f"[get_random_story_all] {name}() failed: {e}")

    # 2) Try bulk fetchers (pick one at random)
    bulk_candidates = [
        "fetch_news_stories",
        "get_stories_for_topic",
    ]
    for name in bulk_candidates:
        if hasattr(self, name):
            try:
                stories = getattr(self, name)(subject) or []
                if stories:
                    ns = norm(random.choice(stories))
                    print(f"[get_random_story_all] Using random from {name}()")
                    return ns
            except Exception as e:
                print(f"[get_random_story_all] {name}() failed: {e}")

    # 3) Synthetic fallback so the queue is never empty
    ts = datetime.now().strftime("%b %d, %Y %I:%M %p")
    print("[get_random_story_all] No concrete source found; returning fallback story.")
    return {
        "title": f"{subject.title()} update — {ts}",
        "preview": f"Quick thought on {subject}. (auto-generated fallback)",
        "url": "#",
    }
def get_random_story_from(self, categories=None):
    """
    Returns a normalized dict: {title, preview, url}
    categories: list[str] or None => all categories
    """
    cats = []
    if not categories:
        cats = list(RSS_FEEDS.keys())
    else:
        cats = [str(c).strip().lower() for c in categories if str(c).strip()]

    feeds = []
    for c in cats:
        bucket = RSS_FEEDS.get(c)
        if not isinstance(bucket, dict):
            continue
        for tier in ("primary", "secondary"):
            lst = bucket.get(tier, [])
            if isinstance(lst, list):
                for f in lst:
                    if isinstance(f, dict) and f.get("url"):
                        feeds.append(f)

    if not feeds:
        return None

    for _ in range(6):
        f = random.choice(feeds)
        url = f.get("url", "")
        name = f.get("name", "RSS")

        try:
            feed = feedparser.parse(url)
            if getattr(feed, "entries", None):
                entry = random.choice(feed.entries[:10])
                title = getattr(entry, "title", "(untitled)")
                link = getattr(entry, "link", "") or getattr(entry, "id", "") or ""
                preview = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
                story = {
                    "title": f"{title} ({name})",
                    "preview": preview.strip(),
                    "url": link.strip() or url,
                }
                return self._normalize_story(story, subject=(cats[0] if cats else "news"))
        except Exception as e:
            print(f"[get_random_story_from] parse failed ({name}): {e}")

    return None

# Constants
ENCRYPTION_KEY_FILE = "encryption.key"
CREDENTIALS_FILE = "encrypted_credentials.bin"
CHARACTERS_FILE = "encrypted_characters.bin"
FEED_CONFIG_FILE = "encrypted_feed_config.bin"  # New file for feed selection
MAX_TWEETS_PER_MONTH = 500
TWEET_INTERVAL_HOURS = 1.5
FEED_TIMEOUT = 10  # seconds
FEED_ERROR_THRESHOLD = 5  # max consecutive errors before skipping feed
MIN_STORIES_PER_FEED = 2  # minimum stories to get from each feed
PRIMARY_FEED_WEIGHT = 2.0  # Weight multiplier for primary sources

# Constants for meme handling
SUPPORTED_MEME_FORMATS = ('.jpg', '.jpeg', '.png', '.gif')
USED_MEMES_HISTORY = 10  # How many recently used memes to remember

# Twitter API Rate Limits
TWITTER_RATE_LIMITS = {
    "tweets": {
        "endpoint": "statuses/update",
        "window_hours": 3,
        "max_tweets": 300,  # Combined limit for tweets and retweets
        "current_count": 0,
        "window_start": None,
        "reset_time": None,
        "backoff_until": None
    }
}

# Twitter API retry settings
TWITTER_RETRY_CONFIG = {
    "initial_backoff": 60,  # Start with 1 minute
    "max_backoff": 3600,    # Max 1 hour
    "backoff_factor": 2,    # Double each time
    "max_retries": 5
}

# Default headers for feed requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# OpenAI Models with limits
OPENAI_MODELS = {
    "gpt-3.5-turbo (Most affordable)": {
        "name": "gpt-3.5-turbo",
        "tpm": "10M tokens/min",
        "rpm": "10K requests/min"
    },
    "gpt-4o": {
        "name": "gpt-4o",
        "tpm": "2M tokens/min",
        "rpm": "10K requests/min"
    },
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "tpm": "10M tokens/min",
        "rpm": "10K requests/min"
    },
    "gpt-4": {
        "name": "gpt-4",
        "tpm": "300K tokens/min",
        "rpm": "10K requests/min"
    },
    "gpt-4-turbo": {
        "name": "gpt-4-turbo",
        "tpm": "800K tokens/min",
        "rpm": "10K requests/min"
    }
}

# RSS Feed Categories
RSS_FEEDS = {
    "crypto": {
        "primary": [
            {"url": "https://www.theblock.co/rss.xml", "name": "The Block"},
            {"url": "https://blog.kraken.com/feed", "name": "Kraken Blog"},
            {"url": "https://messari.io/rss", "name": "Messari"},
            {"url": "https://blockworks.co/feed", "name": "Blockworks"}
        ],
        "secondary": [
            {"url": "https://cointelegraph.com/rss", "name": "CoinTelegraph"},
            {"url": "https://cryptonews.com/news/feed/", "name": "CryptoNews"},
            {"url": "https://decrypt.co/feed", "name": "Decrypt"},
            {"url": "https://news.bitcoin.com/feed/", "name": "Bitcoin.com"},
            {"url": "https://coindesk.com/arc/outboundfeeds/rss/", "name": "CoinDesk"},
            {"url": "https://bitcoinmagazine.com/.rss/full/", "name": "Bitcoin Magazine"},
            {"url": "https://cryptopotato.com/feed/", "name": "CryptoPotato"},
            {"url": "https://ambcrypto.com/feed/", "name": "AMBCrypto"},
            {"url": "https://newsbtc.com/feed/", "name": "NewsBTC"},
            {"url": "https://cryptoslate.com/feed/", "name": "CryptoSlate"},
            {"url": "https://beincrypto.com/feed/", "name": "BeInCrypto"},
            {"url": "https://bitcoinist.com/feed/", "name": "Bitcoinist"},
            {"url": "https://dailyhodl.com/feed/", "name": "The Daily Hodl"}
        ]
    },
    "ai": {
        "primary": [
            {"url": "http://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=lastUpdatedDate&sortOrder=descending&max_results=10", "name": "arXiv - Artificial Intelligence"},
            {"url": "http://export.arxiv.org/api/query?search_query=cat:cs.LG&sortBy=lastUpdatedDate&sortOrder=descending&max_results=10", "name": "arXiv - Machine Learning"},
            {"url": "http://export.arxiv.org/api/query?search_query=cat:cs.CL&sortBy=lastUpdatedDate&sortOrder=descending&max_results=10", "name": "arXiv - Computation and Language"},
            {"url": "http://export.arxiv.org/api/query?search_query=cat:cs.CV&sortBy=lastUpdatedDate&sortOrder=descending&max_results=10", "name": "arXiv - Computer Vision"},
            {"url": "http://export.arxiv.org/api/query?search_query=cat:cs.NE&sortBy=lastUpdatedDate&sortOrder=descending&max_results=10", "name": "arXiv - Neural and Evolutionary Computing"}
        ],
        "secondary": [
            {"url": "https://blog.research.google/feeds/posts/default", "name": "Google Research Blog"},
            {"url": "https://openai.com/news/rss.xml", "name": "OpenAI Blog"},
            {"url": "https://aws.amazon.com/blogs/machine-learning/feed/", "name": "AWS ML Blog"},
            {"url": "https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/rss", "name": "Microsoft AI Blog"},
            {"url": "https://engineering.fb.com/feed/", "name": "Meta Engineering Blog"}
        ]
    },
    "tech": {
        "primary": [
            {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
            {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
            {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index"},
            {"name": "WIRED", "url": "https://www.wired.com/feed/rss"},
            {"name": "Engadget", "url": "https://www.engadget.com/rss.xml"},
            {"name": "MIT Technology Review", "url": "https://www.technologyreview.com/feed/"},
        ],
        "secondary": [
            {"name": "Tom's Hardware", "url": "https://www.tomshardware.com/feeds/all"},
            {"name": "The Next Web", "url": "https://thenextweb.com/feed"},
        ]
},

}
def get_random_story_from(categories=None):
    """
    Returns a string like:
      "Title — link  (source: Feed Name)"
    Supports your RSS_FEEDS structure:
      RSS_FEEDS["crypto"]["primary"] = [{"url": "...", "name": "..."}]
    categories: list[str] or None => all categories
    """
    cats = []
    if not categories:
        cats = list(RSS_FEEDS.keys())
    else:
        cats = [str(c).strip().lower() for c in categories if str(c).strip()]

    pool = []
    for c in cats:
        bucket = RSS_FEEDS.get(c)
        if not isinstance(bucket, dict):
            continue
        for tier in ("primary", "secondary"):
            lst = bucket.get(tier, [])
            if isinstance(lst, list):
                for f in lst:
                    if isinstance(f, dict) and f.get("url"):
                        pool.append(f)

    if not pool:
        return None

    for _ in range(6):
        f = random.choice(pool)
        url = f.get("url", "")
        name = f.get("name", "RSS")

        try:
            feed = feedparser.parse(url)
            entries = getattr(feed, "entries", None) or []
            if entries:
                entry = random.choice(entries[:10])
                title = getattr(entry, "title", "(untitled)")
                link = getattr(entry, "link", "") or getattr(entry, "id", "") or ""
                return f"{title} — {link}\n(source: {name})"
        except Exception:
            continue

    return None

class EncryptionManager:
    def __init__(self):
        self.key = None
        self.cipher = None
        print("Initializing EncryptionManager...")

        if os.path.exists(ENCRYPTION_KEY_FILE):
            try:
                with open(ENCRYPTION_KEY_FILE, "rb") as f:
                    self.key = f.read()
                print(f"Loaded encryption key, length: {len(self.key)} bytes")
                self.cipher = Fernet(self.key)
                print("Successfully created Fernet cipher")
            except Exception as e:
                print(f"Error loading encryption key: {e}")
                traceback.print_exc()
                self.key = None
                self.cipher = None

        if self.key and self.cipher:
            self.validate_key()

        if not self.key or not self.cipher:
            print("Generating new encryption key...")
            self.key = Fernet.generate_key()
            try:
                with open(ENCRYPTION_KEY_FILE, "wb") as f:
                    f.write(self.key)
                self.cipher = Fernet(self.key)
                print("Successfully generated and saved new key")
            except Exception as e:
                print(f"Error saving new encryption key: {e}")
                self.key = None
                self.cipher = None

    def validate_key(self):
        try:
            test_cipher = Fernet(self.key)
            test_message = b"Test message for encryption validation"
            encrypted_message = test_cipher.encrypt(test_message)
            decrypted_message = test_cipher.decrypt(encrypted_message)
            assert test_message == decrypted_message, "Decrypted message does not match original"
            print("Encryption key validation passed.")
        except Exception as validation_error:
            print(f"Encryption key validation failed: {validation_error}")
            traceback.print_exc()

    def encrypt(self, data):
        if not self.cipher:
            print("Error encrypting data: cipher not initialized")
            return None
        try:
            json_data = json.dumps(data)
            encrypted = self.cipher.encrypt(json_data.encode())
            return encrypted
        except Exception as e:
            print(f"Error encrypting data: {e}")
            return None

    def decrypt(self, encrypted_data):
        if not self.cipher:
            print("Error decrypting data: cipher not initialized")
            return {}
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted.decode())
        except Exception as e:
            print(f"Error decrypting data: {e}")
            traceback.print_exc()
            return {}


class CryptoArticle:
    def __init__(self, title, preview, full_text, link, published_date):
        self.title = title
        self.preview = preview
        self.full_text = full_text
        self.link = link
        self.published_date = published_date

    def get_topic_text(self):
        return f"{self.title}\n\n{self.preview}"


class TwitterBot:
    def __init__(self):
        print("\n=== Initializing TwitterBot ===")
        self.encryption_manager = EncryptionManager()

        self.credentials = {}
        self.characters = {}
        self.feed_config = {}

        # Scheduler state
        self.scheduler_running = False
        self.scheduler_character = None
        self.scheduler_subject = "crypto"
        self.reply_bank = []
        self.last_daily_reply_date = None

        # Tweet cadence state
        self.tweet_queue = queue.Queue()
        self.last_successful_tweet = None
        self.last_observation_time = None
        self.backoff_until = None

        # Feed/memory state
        self.current_topic = ""
        self.feed_index = 0
        self.used_stories = set()
        self.recent_topics = []
        self.MAX_RECENT_TOPICS = 50
        self.feed_errors = defaultdict(int)
        self.feed_last_used = {}

        # Clients
        self.twitter_client = None
        self.client = None  # OpenAI client (optional)

        # Meme system
        self.use_memes = False
        self.meme_counter = 0
        self.meme_frequency = 5
        self.used_memes = set()

        if not os.path.exists("memes"):
            os.makedirs("memes")

        # Rate limit tracking
        self.rate_limits = TWITTER_RATE_LIMITS.copy()

        print("\n=== Loading Initial Data ===")
        self.credentials = self.load_credentials()
        print(f"Loaded credentials: {json.dumps(self.credentials, indent=2)}")

        self.characters = self.load_characters()
        print(f"Loaded characters: {json.dumps(self.characters, indent=2)}")

        self.feed_config = self.load_feed_config()
        print(f"Loaded feed configuration: {json.dumps(self.feed_config, indent=2)}")

        # Initialize OpenAI client ONLY if enabled
        if USE_OPENAI and self.credentials.get("openai_key"):
            self.client = OpenAI(
                api_key=self.credentials["openai_key"],
                http_client=httpx.Client(
                    base_url="https://api.openai.com/v1",
                    follow_redirects=True,
                    timeout=60.0,
                ),
            )
            print("OpenAI client initialized (USE_OPENAI enabled).")
        else:
            if self.credentials.get("openai_key") and not USE_OPENAI:
                print("OpenAI key present, but USE_OPENAI is disabled — skipping OpenAI client init.")
            else:
                print("OpenAI client disabled or missing key.")

        # Initialize Twitter client if credentials present
        if all(k in self.credentials for k in [
            "twitter_api_key",
            "twitter_api_secret",
            "twitter_access_token",
            "twitter_access_token_secret",
        ]):
            self.twitter_client = tweepy.Client(
                consumer_key=self.credentials["twitter_api_key"],
                consumer_secret=self.credentials["twitter_api_secret"],
                access_token=self.credentials["twitter_access_token"],
                access_token_secret=self.credentials["twitter_access_token_secret"],
            )
            print("Twitter client initialized.")
        else:
            print("Twitter client NOT initialized (missing X credentials).")

    def _normalize_subject(self, subject):
        """
        Map placeholders like 'Surprise_All'/'random' to a real subject from feed_config or RSS_FEEDS.
        Returns the canonical key (preserve original keys when possible).
        """
        s_raw = (str(subject).strip() if subject else "")
        s = s_raw.lower()

        if not s or s in {"surprise_all", "__surprise_all__", "surprise-all", "random", "any", "all", "*"}:
            pool = list(getattr(self, "feed_config", {}).keys())
            if not pool:
                pool = list(RSS_FEEDS.keys())

            pool = [k for k in pool if k and str(k).lower() not in {
                "surprise_all", "__surprise_all__", "surprise-all", "random", "any", "all", "*"
            }]
            return random.choice(pool) if pool else "news"

        # If user gave "AI" but keys are "ai", normalize to existing key
        if "RSS_FEEDS" in globals():
            for k in RSS_FEEDS.keys():
                if str(k).lower() == s:
                    return k

        if hasattr(self, "feed_config") and isinstance(self.feed_config, dict):
            for k in self.feed_config.keys():
                if str(k).lower() == s:
                    return k

        return s_raw or "news"

    def scheduler_worker(self):
        print("\n🛠️ Starting scheduler worker...")

        # Persist character/subject for auto-refill later
        self.scheduler_character = getattr(self, "scheduler_character", None)
        self.scheduler_subject = getattr(self, "scheduler_subject", "crypto")

        # State for cadence + daily reply window
        self.reply_bank = getattr(self, "reply_bank", [])
        self.last_daily_reply_date = getattr(self, "last_daily_reply_date", None)
        self.last_observation_time = getattr(self, "last_observation_time", None)

        # If never tweeted successfully, don't wait 4 hours on first run
        if not getattr(self, "last_successful_tweet", None):
            print("🚀 No previous tweet timestamp found. Setting last_successful_tweet to now.")
            self.last_successful_tweet = datetime.now()
        # Prevent an immediate observation tweet right after startup / meme post
        if self.last_observation_time is None:
            self.last_observation_time = datetime.now()

        OBS_INTERVAL = timedelta(hours=3)
        MAIN_TWEET_INTERVAL_SEC = 4 * 3600

        while self.scheduler_running:
            try:
                # Respect any backoff
                if self.backoff_until and datetime.now() < self.backoff_until:
                    wait_seconds = (self.backoff_until - datetime.now()).total_seconds()
                    print(
                        f"⏳ Backoff active until {self.backoff_until}. "
                        f"Sleeping {wait_seconds/60:.1f} minutes..."
                    )
                    time.sleep(min(60, max(1, wait_seconds)))
                    continue

                now = datetime.now()

                # === (1) Daily replies at ~10:00 AM ===
                ten_am_today = now.replace(hour=10, minute=0, second=0, microsecond=0)

                if self.last_daily_reply_date != now.date() and now >= ten_am_today:
                    print("\n📬 10:00 AM reached — checking mentions and replying to up to 2…")
                    handled = 0

                    pending = []
                    if self.reply_bank:
                        print(f"↪ Using {len(self.reply_bank)} banked mentions first.")
                        pending.extend(self.reply_bank)
                        self.reply_bank = []

                    if not pending:
                        try:
                            if hasattr(self, "collect_unreplied_mentions"):
                                pending = self.collect_unreplied_mentions() or []
                            elif hasattr(self, "fetch_recent_mentions"):
                                pending = self.fetch_recent_mentions() or []
                            elif hasattr(self, "monitor_and_reply_to_mentions"):
                                print("⚠ Using monitor_and_reply_to_mentions fallback.")
                                self.monitor_and_reply_to_mentions()
                                pending = []
                            else:
                                print("⚠ No mention-collection method found.")
                                pending = []
                        except Exception as e:
                            print(f"❌ Error fetching mentions: {e}")
                            pending = []

                    for m in pending:
                        if handled >= 2:
                            self.reply_bank.append(m)
                            continue
                        try:
                            if hasattr(self, "reply_to_mention"):
                                self.reply_to_mention(m)
                                handled += 1
                            elif hasattr(self, "reply_to_engagement"):
                                self.reply_to_engagement(m)
                                handled += 1
                            else:
                                self.reply_bank.append(m)
                        except Exception as e:
                            print(f"❌ Failed replying to a mention: {e}")
                            self.reply_bank.append(m)

                    print(
                        f"✅ Replied to {handled} mention(s). "
                        f"Banked {len(self.reply_bank)} leftover(s)."
                    )
                    self.last_daily_reply_date = now.date()

                # === (2) Periodic Mork Core observation (Option 3.B) ===
                if self.last_observation_time is None or (now - self.last_observation_time) >= OBS_INTERVAL:
                    try:
                        # reflect (optional, but useful)
                        did_reflect = False
                        try:
                            did_reflect = core_reflect(timeout=6)
                        except Exception as e:
                            print(f"⚠ core_reflect failed: {e}")
                        print(f"🧠 core_reflect: {did_reflect}")

                        # ask Core to compose an "observation/reflection" style tweet
                        obs = ""
                        try:
                            obs = core_compose_payload({"kind": "reflection", "maxChars": 260}, timeout=10)
                            if not obs:
                                obs = core_compose_payload({"kind": "arb", "maxChars": 260}, timeout=10)
                            if not obs:
                                obs = core_compose_payload({"kind": "observation", "maxChars": 260}, timeout=10)
                        except Exception as e:
                            print(f"⚠ core compose failed: {e}")
                            obs = ""

                        if obs:
                            print("🧠 Posting Mork Core observation…")
                            ok = self.send_tweet(obs)
                            if ok:
                                self.last_observation_time = now
                                time.sleep(random.uniform(5, 12))
                            else:
                                print("⚠ Observation tweet send failed (send_tweet returned false).")
                        else:
                            print("⚠ Core returned empty composed tweet (/x/compose).")

                    except Exception as e:
                        print(f"⚠ Observation tweet failed: {e}")

                # === (3) Main tweet every 4 hours ===
                due_for_main = (now - self.last_successful_tweet).total_seconds() >= MAIN_TWEET_INTERVAL_SEC

                if due_for_main:
                    print("\n⏰ 4 hours passed — preparing to send next tweet...")

                    if not self.tweet_queue.empty():
                        character, story_text, subject = self.tweet_queue.get()
                        tweet_text = self.generate_tweet(character, story_text)

                        if tweet_text and self.send_tweet(tweet_text):
                            print("✅ Tweet from queue sent.")
                            self.last_successful_tweet = datetime.now()
                        else:
                            print("❌ Failed to send tweet from queue.")
                    else:
                        print("📭 Tweet queue is empty — trying to refill...")
                        seeded = 0
                        for _ in range(3):
                            s = self.get_new_story(self.scheduler_subject)
                            if not s:
                                break
                            txt = (
                                f"{s['title']}\n\n"
                                f"{s.get('preview','')}\n\n"
                                f"Read more: {s['url']}"
                            )
                            self.tweet_queue.put((self.scheduler_character, txt, self.scheduler_subject))
                            seeded += 1
                        print(f"📥 Refilled with {seeded} story(ies).")

                time.sleep(30)

            except Exception as e:
                print(f"❌ Error in scheduler worker: {e}")
                time.sleep(60)

    def get_stories_from_feed(self, url, limit: int = 10):
        """
        Fetch RSS/Atom items and return a list of dicts with: title, preview, url.
        Tries feedparser; falls back to requests + XML.
        """
        items = []

        # 1) feedparser path
        try:
            import feedparser
            d = feedparser.parse(url)
            for e in (d.entries or [])[:limit]:
                title = (e.get("title") or "").strip()
                link = (e.get("link") or "#").strip()
                summary = (e.get("summary") or e.get("description") or "").strip()
                if title or summary:
                    items.append({"title": title, "preview": summary, "url": link})
            if items:
                return items
        except Exception as e:
            print(f"[get_stories_from_feed] feedparser error for {url}: {e}")

        # 2) fallback XML path
        try:
            import re, html, requests, xml.etree.ElementTree as ET
            r = requests.get(url, timeout=10, headers=DEFAULT_HEADERS if "DEFAULT_HEADERS" in globals() else None)
            r.raise_for_status()

            root = ET.fromstring(r.content)

            # RSS
            for it in root.findall(".//item")[:limit]:
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "#").strip()
                desc = (it.findtext("description") or "").strip()
                desc = html.unescape(re.sub(r"<[^>]+>", "", desc))
                if title or desc:
                    items.append({"title": title, "preview": desc, "url": link})

            if items:
                return items

            # Atom
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for it in root.findall(".//a:entry", ns)[:limit]:
                title = (it.findtext("a:title", default="", namespaces=ns) or "").strip()
                link_el = it.find("a:link", ns)
                link = link_el.get("href", "#").strip() if link_el is not None else "#"
                desc = (it.findtext("a:summary", default="", namespaces=ns) or "").strip()
                if title or desc:
                    items.append({"title": title, "preview": desc, "url": link})

        except Exception as e:
            print(f"[get_stories_from_feed] fallback error for {url}: {e}")

        return items

    def load_credentials(self):
        print("\nLoading credentials...")
        if not os.path.exists(CREDENTIALS_FILE):
            print("No credentials file found")
            return {}
        try:
            with open(CREDENTIALS_FILE, "rb") as f:
                data = f.read()
            print(f"Read credentials file, size: {len(data)} bytes")
            if not data:
                print("Empty credentials file")
                return {}

            print("Attempting to decrypt credentials...")
            decrypted = self.encryption_manager.decrypt(data) or {}
            if not isinstance(decrypted, dict) or not decrypted:
                print("Failed to decrypt credentials")
                return {}

            print(f"Successfully loaded credentials with keys: {list(decrypted.keys())}")
            return decrypted
        except Exception as e:
            print(f"Error loading credentials: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def load_characters(self):
        print("\nLoading characters...")
        if not os.path.exists(CHARACTERS_FILE):
            print("No characters file found")
            return {}
        try:
            with open(CHARACTERS_FILE, "rb") as f:
                data = f.read()
            print(f"Read characters file, size: {len(data)} bytes")
            if not data:
                print("Empty characters file")
                return {}

            print("Attempting to decrypt characters...")
            decrypted = self.encryption_manager.decrypt(data) or {}
            if not isinstance(decrypted, dict) or not decrypted:
                print("Failed to decrypt characters")
                return {}

            print(f"Successfully loaded characters with keys: {list(decrypted.keys())}")
            return decrypted
        except Exception as e:
            print(f"Error loading characters: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def save_credentials(self, credentials):
        print("\nSaving credentials to file...")
        print(f"Credentials to save: {list(credentials.keys())}")
        try:
            print("Encrypting credentials...")
            encrypted_data = self.encryption_manager.encrypt(credentials)
            if not encrypted_data:
                print("Failed to encrypt credentials")
                return False

            print(f"Encrypted data length: {len(encrypted_data)} bytes")
            with open(CREDENTIALS_FILE, "wb") as f:
                f.write(encrypted_data)

            self.credentials = credentials
            print("Updated bot credentials in memory")

            # OpenAI client ONLY if enabled
            if USE_OPENAI and credentials.get("openai_key"):
                print("Initializing OpenAI client...")
                self.client = OpenAI(
                    api_key=credentials["openai_key"],
                    http_client=httpx.Client(
                        base_url="https://api.openai.com/v1",
                        follow_redirects=True,
                        timeout=60.0,
                    ),
                )
                print("OpenAI client initialized")
            else:
                self.client = None
                if credentials.get("openai_key") and not USE_OPENAI:
                    print("OpenAI key saved, but USE_OPENAI is disabled — client not initialized.")

            # Twitter client if all credentials provided
            needed = {"twitter_api_key", "twitter_api_secret", "twitter_access_token", "twitter_access_token_secret"}
            if needed.issubset(set(credentials.keys())):
                print("Initializing Twitter client...")
                self.twitter_client = tweepy.Client(
                    consumer_key=credentials["twitter_api_key"],
                    consumer_secret=credentials["twitter_api_secret"],
                    access_token=credentials["twitter_access_token"],
                    access_token_secret=credentials["twitter_access_token_secret"],
                )
                print("Twitter client initialized")
            else:
                # keep existing client if you want; or set to None to be strict
                print("Twitter client not re-initialized (missing one or more X keys).")

            return True

        except Exception as e:
            print(f"Error saving credentials: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_characters(self, characters):
        print("\nSaving characters to file...")
        print(f"Characters to save: {list(characters.keys())}")
        try:
            print("Encrypting characters...")
            encrypted_data = self.encryption_manager.encrypt(characters)
            if not encrypted_data:
                print("Failed to encrypt characters")
                return False

            print(f"Encrypted data length: {len(encrypted_data)} bytes")
            with open(CHARACTERS_FILE, "wb") as f:
                f.write(encrypted_data)

            self.characters = characters
            print("Updated bot characters in memory")
            return True

        except Exception as e:
            print(f"Error saving characters: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_article_content(self, url):
        try:
            headers = DEFAULT_HEADERS if "DEFAULT_HEADERS" in globals() else None
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            return ""

    def extract_keywords(self, text):
        """Extract important keywords from text to track topic diversity."""
        common_terms = {
            "crypto","blockchain","bitcoin","ethereum","btc","eth",
            "cryptocurrency","cryptocurrencies","token","tokens","defi",
            "market","markets","trading","price","prices"
        }
        words = re.findall(r"\b\w+\b", (text or "").lower())
        keywords = {
            w for w in words
            if w not in common_terms and len(w) > 3 and not w.isdigit()
        }
        return keywords

    def is_similar_to_recent(self, title, preview):
        """Check if a story is too similar to recently posted ones."""
        new_keywords = self.extract_keywords(f"{title} {preview}")
        if not new_keywords:
            return False

        for recent_keywords in getattr(self, "recent_topics", []) or []:
            if not recent_keywords:
                continue
            denom = len(new_keywords | recent_keywords)
            if denom == 0:
                continue
            overlap = len(new_keywords & recent_keywords) / denom
            if overlap > 0.4:
                return True
        return False

    def get_arxiv_paper_details(self, url):
        """Get detailed information about an arXiv paper including abstract and authors."""
        try:
            paper_id = url.split("/")[-1]
            if "arxiv.org/abs/" in url:
                paper_id = url.split("arxiv.org/abs/")[-1]
            elif "arxiv.org/pdf/" in url:
                paper_id = url.split("arxiv.org/pdf/")[-1].replace(".pdf", "")

            api_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
            response = requests.get(api_url, timeout=FEED_TIMEOUT)
            response.raise_for_status()

            from xml.etree import ElementTree
            root = ElementTree.fromstring(response.content)

            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            entry = root.find(".//atom:entry", ns)
            if entry is None:
                return None

            abstract = (entry.find("atom:summary", ns).text or "").strip()
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            categories = [cat.get("term") for cat in entry.findall("atom:category", ns)]

            links = entry.findall("atom:link", ns)
            html_url = next((lnk.get("href") for lnk in links if lnk.get("type") == "text/html"), None) \
                    or f"https://arxiv.org/abs/{paper_id}"

            return {
                "abstract": abstract,
                "authors": authors,
                "categories": categories,
                "html_url": html_url,
                "paper_id": paper_id,
            }

        except Exception as e:
            print(f"Error fetching arXiv paper details: {e}")
            return None

    def load_feed_config(self):
        """
        Load feed configuration.
        NOTE: Your constants define FEED_CONFIG_FILE as encrypted, but your code uses JSON.
        We'll support BOTH:
        - encrypted FEED_CONFIG_FILE (preferred if present)
        - feed_config.json fallback (legacy)
        """
        # 1) preferred: encrypted file
        try:
            if "FEED_CONFIG_FILE" in globals() and os.path.exists(FEED_CONFIG_FILE):
                with open(FEED_CONFIG_FILE, "rb") as f:
                    blob = f.read()
                cfg = self.encryption_manager.decrypt(blob) or {}
                if isinstance(cfg, dict):
                    return cfg
        except Exception as e:
            print(f"Error loading encrypted feed config: {e}")

        # 2) legacy json
        try:
            if os.path.exists("feed_config.json"):
                with open("feed_config.json", "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading feed configuration: {e}")

        return {}

    def save_feed_config(self, config):
        """
        Save feed configuration.
        Writes encrypted FEED_CONFIG_FILE if defined; also updates in-memory self.feed_config.
        """
        self.feed_config = config

        # 1) preferred: encrypted
        try:
            if "FEED_CONFIG_FILE" in globals():
                encrypted = self.encryption_manager.encrypt(config)
                if encrypted:
                    with open(FEED_CONFIG_FILE, "wb") as f:
                        f.write(encrypted)
                    return True
        except Exception as e:
            print(f"Error saving encrypted feed configuration: {e}")

        # 2) legacy json fallback
        try:
            with open("feed_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving feed configuration: {e}")
            return False

    def get_new_story(self, subject=None):
        """
        Pull the next unused story for a subject from configured RSS_FEEDS/feed_config.
        Falls back to get_random_story_all(subject) to keep cadence alive.
        """
        import random
        subject = subject or getattr(self, "scheduler_subject", None) or "news"

        # Figure out feed list
        feeds = None
        if getattr(self, "feed_config", None):
            feeds = self.feed_config.get(subject)
        if feeds is None and "RSS_FEEDS" in globals():
            feeds = RSS_FEEDS.get(subject)

        # Build normalized list of URL strings
        if isinstance(feeds, dict):
            raws = list(feeds.get("primary", [])) + list(feeds.get("secondary", []))
        elif isinstance(feeds, (list, tuple)):
            raws = list(feeds)
        else:
            raws = []

        urls = []
        for item in raws:
            if isinstance(item, dict):
                u = item.get("url")
            else:
                u = item
            if u:
                urls.append(u)

        random.shuffle(urls)

        # Try feeds
        for url in urls:
            try:
                stories = self.get_stories_from_feed(url, limit=10) or []
                for s in stories:
                    u = (s.get("url") or s.get("link") or "#").strip()
                    if u in self.used_stories:
                        continue

                    title = (s.get("title") or "").strip()
                    preview = (s.get("preview") or s.get("summary") or "").strip()
                    if not (title or preview):
                        continue

                    # similarity filter
                    if getattr(self, "recent_topics", None) and self.is_similar_to_recent(title, preview):
                        continue

                    story = {"title": title or subject.title(), "preview": preview, "url": u}
                    self.used_stories.add(u)

                    # record keyword-set for diversity tracking
                    kw = self.extract_keywords(f"{story['title']} {story['preview']}")
                    if kw:
                        self.recent_topics.append(kw)
                        if len(self.recent_topics) > getattr(self, "MAX_RECENT_TOPICS", 50):
                            self.recent_topics = self.recent_topics[-self.MAX_RECENT_TOPICS:]

                    return story

            except Exception as e:
                print(f"Error fetching from feed {url}: {e}")

        # Last resort: use your all-sources selector (which has a synthetic fallback)
        return self.get_random_story_all(subject)

    def _parse_story_block(self, topic):

        """
        Your Sherpa story_text format is typically:
        Title

        Preview...

        Read more: https://...
        Returns (title, text, url)
        """
        topic = (topic or "").strip()
        if not topic:
            return ("", "", "")

        # Extract URL from "Read more:"
        url_match = re.search(r"Read more:\s*(https?://\S+)", topic)
        article_url = url_match.group(1).strip() if url_match else ""

        # Remove the Read more line from the body
        clean = re.sub(r"\n*\s*Read more:\s*https?://\S+\s*", "", topic).strip()

        # Title = first non-empty line
        lines = [ln.strip() for ln in clean.splitlines()]
        lines = [ln for ln in lines if ln]
        if not lines:
            return ("", clean, article_url)

        title = lines[0]
        text = "\n".join(lines[1:]).strip()
        return (title, text, article_url)


    def generate_tweet(self, character_name, topic):
        """
        Primary tweet generator.
        - If USE_OPENAI is off (or client missing), we route to Mork Core composer with payloads.
        - If USE_OPENAI is on, we use your existing OpenAI logic (kept, but cleaned/guarded).
        """

        # --- Safety: ensure required state exists (Gradio / reload / partial init can skip __init__) ---
        if not hasattr(self, "tweet_count"):
            self.tweet_count = 0
        if not hasattr(self, "last_tweet_time"):
            self.last_tweet_time = None
        if not hasattr(self, "used_stories"):
            self.used_stories = set()
        if not hasattr(self, "recent_topics"):
            self.recent_topics = []
        if not hasattr(self, "rate_limits"):
            try:
                self.rate_limits = TWITTER_RATE_LIMITS.copy()
            except Exception:
                self.rate_limits = {}
        if not hasattr(self, "backoff_until"):
            self.backoff_until = None
        # --------------------------------------------------------------------------------------------

        character = self.characters.get(character_name)
        if not character:
            return None

        try:
            # Monthly cap (kept)
            if self.tweet_count >= MAX_TWEETS_PER_MONTH:
                current_time = datetime.now()
                if not self.last_tweet_time or (current_time - self.last_tweet_time).days >= 30:
                    self.tweet_count = 0
                else:
                    return "Monthly tweet limit reached. Please wait for the next cycle."

            # Parse story payload if present
            title, text, article_url = self._parse_story_block(topic or "")

            # Calculate content limit (if URL gets appended)
            TWITTER_SHORT_URL_LENGTH = 24
            max_content_length = 280 - TWITTER_SHORT_URL_LENGTH if article_url else 280
            max_chars_for_core = min(260, max_content_length)  # keep your usual 260 internal target

            # ----------------------------
            # NO-OPENAI MODE: use Mork Core
            # ----------------------------
            if (not USE_OPENAI) or (not getattr(self, "client", None)):
                payload = None

                # If this looks like a feed story (we have a URL or a title+text block), compose as "feed"
                if article_url or (title and text):
                    payload = {
                        "kind": "feed",
                        "title": title or "Update",
                        "text": (text or "").strip(),
                        "url": article_url or "",
                        "maxChars": max_chars_for_core,
                    }
                else:
                    # Otherwise treat as an observation prompt (mention replies often come in here too)
                    payload = {
                        "kind": "observation",
                        "text": (topic or "").strip(),
                        "maxChars": max_chars_for_core,
                    }

                tweet_text = core_compose_payload(payload, timeout=10) or ""

                # If core returns nothing, last-resort fallback so we don't crash your scheduler
                if not tweet_text:
                    tweet_text = _wrap_280((topic or "…").strip() or "…", max_chars_for_core)

                # Append article URL at end if needed (and not already included)
                if article_url and article_url not in tweet_text:
                    tweet_text = _wrap_280(f"{tweet_text} {article_url}", 280)

                self.tweet_count += 1
                self.last_tweet_time = datetime.now()
                return tweet_text

            # ----------------------------
            # OPENAI MODE (kept, tightened)
            # ----------------------------

            clean_topic = (topic or "").strip()
            if article_url:
                clean_topic = re.sub(r"\n*\s*Read more:\s*https?://\S+\s*", "", clean_topic).strip()

            prompt_variants = [
                "Speak as if you're writing a soliloquy for a tragic sauce-themed play.",
                "Add a sprinkle of literary irony, but make it savory.",
                "Pretend to be distracted.",
                "Imagine you're writing from exile in a forgotten condiment aisle.",
                "Use language that suggests you're the last philosopher alive.",
                "Add an unexpected culinary metaphor, ideally involving vinegar or smoke.",
                "Maintain melancholy but make it tastefully funny.",
                "Respond as if the conversation was with a long lost friend.",
                "End with an awkward outro.",
            ]
            hour = datetime.now().hour
            if hour < 12:
                prompt_variants.append("Start with morning gloom, like breakfast with no sauce.")
            elif hour > 20:
                prompt_variants.append("Make it sound like a sauce-stained midnight confession.")

            variation = random.choice(prompt_variants)

            messages = [
                {"role": "system", "content": character["prompt"]},
                {
                    "role": "user",
                    "content": (
                        f"{variation}\n\n"
                        f"Create a tweet about this topic that is {max_content_length} characters or less. "
                        f"Make it engaging and maintain character voice. NO hashtags, emojis, or URLs.\n\n"
                        f"Topic:\n{clean_topic}"
                    ),
                },
            ]

            response = self.client.chat.completions.create(
                model=character["model"],
                messages=messages,
                max_tokens=200,
                temperature=1.0,
                presence_penalty=0.6,
                frequency_penalty=0.6,
            )

            tweet_text = (response.choices[0].message.content or "").strip()

            # Strip surrounding quotes
            if len(tweet_text) >= 2 and (
                (tweet_text[0] == '"' and tweet_text[-1] == '"') or
                (tweet_text[0] == "'" and tweet_text[-1] == "'")
            ):
                tweet_text = tweet_text[1:-1].strip()

            # Hard truncate by sentence boundary
            if len(tweet_text) > max_content_length:
                sentences = re.split(r"(?<=[.!?])\s+", tweet_text)
                out = ""
                for s in sentences:
                    cand = (out + (" " if out else "") + s).strip()
                    if len(cand) <= max_content_length:
                        out = cand
                    else:
                        break
                tweet_text = out.strip() or tweet_text[:max_content_length].rstrip() + "…"

            if article_url:
                tweet_text = _wrap_280(f"{tweet_text} {article_url}", 280)

            self.tweet_count += 1
            self.last_tweet_time = datetime.now()
            return tweet_text

        except Exception:
            import traceback
            print("❌ Error generating tweet:")
            traceback.print_exc()
            return None

    def check_rate_limit(self):
        """Check if we're within rate limits for tweeting"""
        current_time = datetime.now()

        if self.rate_limits["tweets"]["backoff_until"]:
            if current_time < self.rate_limits["tweets"]["backoff_until"]:
                wait_seconds = (self.rate_limits["tweets"]["backoff_until"] - current_time).total_seconds()
                print(f"\nIn backoff period. Waiting {wait_seconds/60:.1f} minutes")
                return False
            else:
                print("\nBackoff period ended, resetting rate limits")
                self.rate_limits["tweets"]["backoff_until"] = None
                self.rate_limits["tweets"]["current_count"] = 0
                self.rate_limits["tweets"]["window_start"] = current_time
                return True

        if not self.rate_limits["tweets"]["window_start"]:
            self.rate_limits["tweets"]["window_start"] = current_time
            self.rate_limits["tweets"]["current_count"] = 0
            return True

        window_hours = self.rate_limits["tweets"]["window_hours"]
        window_start = self.rate_limits["tweets"]["window_start"]
        if (current_time - window_start).total_seconds() > window_hours * 3600:
            # Reset window
            self.rate_limits["tweets"]["window_start"] = current_time
            self.rate_limits["tweets"]["current_count"] = 0
            print("\nRate limit window reset")
            return True

        if self.rate_limits["tweets"]["current_count"] < self.rate_limits["tweets"]["max_tweets"]:
            remaining = self.rate_limits["tweets"]["max_tweets"] - self.rate_limits["tweets"]["current_count"]
            print(f"\nRate limit status:")
            print(f"  Remaining: {remaining}")
            print(f"  Window started: {window_start}")
            print(f"  Window ends: {window_start + timedelta(hours=window_hours)}")
            return True
        
        reset_time = window_start + timedelta(hours=window_hours)
        wait_seconds = (reset_time - current_time).total_seconds()
        print(f"\nRate limit reached. Window resets in {wait_seconds/3600:.1f} hours")
        print(f"Current count: {self.rate_limits['tweets']['current_count']}")
        print(f"Window started: {window_start}")
        print(f"Window ends: {reset_time}")
        return False

    def handle_rate_limit_error(self, e):
        """Handle rate limit error with exponential backoff"""
        current_time = datetime.now()

        if hasattr(e, 'response') and e.response is not None:
            reset_time = e.response.headers.get('x-rate-limit-reset')
            if reset_time:
                reset_datetime = datetime.fromtimestamp(int(reset_time))
                wait_seconds = (reset_datetime - current_time).total_seconds()
            else:
                # If no reset time in headers, use exponential backoff
                current_backoff = self.rate_limits["tweets"].get("current_backoff", TWITTER_RETRY_CONFIG["initial_backoff"])
                wait_seconds = min(current_backoff * TWITTER_RETRY_CONFIG["backoff_factor"], 
                                 TWITTER_RETRY_CONFIG["max_backoff"])
                self.rate_limits["tweets"]["current_backoff"] = wait_seconds
        else:
            # No response headers, use exponential backoff
            current_backoff = self.rate_limits["tweets"].get("current_backoff", TWITTER_RETRY_CONFIG["initial_backoff"])
            wait_seconds = min(current_backoff * TWITTER_RETRY_CONFIG["backoff_factor"], 
                             TWITTER_RETRY_CONFIG["max_backoff"])
            self.rate_limits["tweets"]["current_backoff"] = wait_seconds
        
        backoff_until = current_time + timedelta(seconds=wait_seconds)
        self.rate_limits["tweets"]["backoff_until"] = backoff_until
        
        print(f"\nRate limit exceeded. Implementing backoff:")
        print(f"  Wait time: {wait_seconds/60:.1f} minutes")
        print(f"  Resume at: {backoff_until}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response status: {e.response.status_code}")
            print(f"  Headers: {dict(e.response.headers)}")
        
        return wait_seconds

    def update_rate_limit(self):
        """Update rate limit counters after successful tweet"""
        self.rate_limits["tweets"]["current_count"] += 1
        print(f"\nUpdated rate limit count: {self.rate_limits['tweets']['current_count']}")
        print(f"Remaining in window: {self.rate_limits['tweets']['max_tweets'] - self.rate_limits['tweets']['current_count']}")

    def send_main_tweet(self):
        """Send a main scheduled tweet."""
        new_story = self.get_new_story("crypto")  # or "ai", depending on your subject
        if new_story:
            story_text = f"{new_story['title']}\n\n{new_story['preview']}\n\nRead more: {new_story['url']}"
            tweet_text = self.generate_tweet("mork zuckerbarge", story_text)
            if tweet_text:
                if self.send_tweet(tweet_text):
                    self.last_successful_tweet = datetime.now()
                    print("✅ Main tweet sent successfully.")
                else:
                    print("❌ Failed to send main tweet.")

    def reply_to_mentions_and_replies(self):
        if self.backoff_until and datetime.now() < self.backoff_until:
            print(f"⏳ Backoff active until {self.backoff_until}. Skipping mention/reply checking.")
            return

        print("🔍 Checking mentions and replies...")

        replies_sent = 0
        max_replies = 1
        
        print("🔍 Checking mentions and replies...")

        replies_sent = 0
        max_replies = 1

        try:
            mentions = self.twitter_client.get_users_mentions(self.mork_id, max_results=10)

            if mentions.data:
                for mention in mentions.data:
                    if replies_sent >= max_replies:
                        break
                    if mention.author_id == self.mork_id:
                        continue 

                    prompt = f"Someone mentioned you: \"{mention.text}\""
                    reply = self.generate_tweet("mork zuckerbarge", prompt)

                    if reply:
                        self.twitter_client.create_tweet(
                            text=reply,
                            in_reply_to_tweet_id=mention.id
                        )
                        print(f"💬 Replied to {mention.id}")
                        replies_sent += 1

        except tweepy.TooManyRequests as e:
            print("🚫 Rate limited! Setting global backoff...")

            if hasattr(e, 'response') and e.response is not None:
                reset_timestamp = e.response.headers.get('x-rate-limit-reset')
                if reset_timestamp:
                    reset_time = datetime.fromtimestamp(int(reset_timestamp))
                    self.backoff_until = reset_time
                    wait_seconds = (reset_time - datetime.now()).total_seconds()
                    print(f"😴 Global backoff active until {self.backoff_until} (about {wait_seconds//60:.1f} min)")
                    time.sleep(min(300, wait_seconds))  # Sleep max 5 min chunks so you can still process local stuff
                    return False
            else:
                print("⚡ No reset time provided. Defaulting to 5 min backoff.")
                self.backoff_until = datetime.now() + timedelta(minutes=5)
                time.sleep(300)
                return False

        except Exception as e:
            print(f"❌ Error replying to mentions: {e}")

    def monitor_and_reply_to_engagement(self):
        self.monitor_and_reply_to_mentions()

    def send_tweet(self, tweet_text):
        if self.backoff_until and datetime.now() < self.backoff_until:
            print(f"⏳ Backoff active until {self.backoff_until}. Skipping sending tweet.")
            return False

        if not self.check_rate_limit():
            print("Tweet skipped due to rate limit")
            return False

        try:
            client = tweepy.Client(
                consumer_key=self.credentials['twitter_api_key'],
                consumer_secret=self.credentials['twitter_api_secret'],
                access_token=self.credentials['twitter_access_token'],
                access_token_secret=self.credentials['twitter_access_token_secret'],
                wait_on_rate_limit=True
            )

            # Extract URL from tweet text and ensure it's at the end
            url_match = re.search(r'(https?://\S+)$', tweet_text)
            if url_match:
                url = url_match.group(1)
                tweet_text = re.sub(r'\s*' + re.escape(url) + r'\s*', '', tweet_text).strip()
                tweet_text = f"{tweet_text}\n\n{url}"

            print(f"\nSending tweet: {tweet_text}")

            response = client.create_tweet(text=tweet_text)

            if response.data:
                self.last_successful_tweet = datetime.now()
                print("\nTweet sent successfully")
                print(f"Tweet ID: {response.data['id']}")
                print(f"Response data: {response.data}")

                tweet_id = response.data['id']
                username = self.credentials.get("twitter_username", "zuckerbarge")
                tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"
                self.send_to_telegram(tweet_url)

                self.update_rate_limit()
                return True

            print("\nTweet failed - no response data")
            return False

        except Exception as e:
            print(f"\nError sending tweet: {e}")
            return False

            self.update_rate_limit()
            return True
            
            print("\nTweet failed - no response data")
            print(f"Response object: {response}")
            return False
            
        except tweepy.TooManyRequests as e:
            print("🚫 Rate limited! Checking Twitter reset time...")

            if hasattr(e, 'response') and e.response is not None:
                reset_timestamp = e.response.headers.get('x-rate-limit-reset')
                if reset_timestamp:
                    reset_time = datetime.fromtimestamp(int(reset_timestamp))
                    now = datetime.now()
                    wait_seconds = (reset_time - now).total_seconds()

                    if wait_seconds > 0:
                        print(f"😴 Sleeping for {wait_seconds//60:.1f} minutes until rate limit resets...")
                        time.sleep(wait_seconds + 5)  # plus 5 second buffer
                        return
            else:
                print("⚡ No reset time found. Sleeping 5 minutes as fallback.")
                time.sleep(300)
                return

        except tweepy.TooManyRequests as e:
            print("🚫 Rate limited! Setting global backoff...")

            if hasattr(e, 'response') and e.response is not None:
                reset_timestamp = e.response.headers.get('x-rate-limit-reset')
                if reset_timestamp:
                    reset_time = datetime.fromtimestamp(int(reset_timestamp))
                    self.backoff_until = reset_time
                    wait_seconds = (reset_time - datetime.now()).total_seconds()
                    print(f"😴 Global backoff active until {self.backoff_until} (about {wait_seconds//60:.1f} min)")
                    time.sleep(min(300, wait_seconds))  # Sleep max 5 min chunks so you can still process local stuff
                    return False
            else:
                print("⚡ No reset time provided. Defaulting to 5 min backoff.")
                self.backoff_until = datetime.now() + timedelta(minutes=5)
                time.sleep(300)
                return False

        except Exception as e:
            print(f"\nError sending tweet: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_random_meme(self, character_name):
        """Pick a meme file and get tweet text from Mork Core using the filename as context."""
        try:
            meme_files = [f for f in os.listdir("memes") if f.lower().endswith(tuple(SUPPORTED_MEME_FORMATS))]
            if not meme_files:
                return None, None

            available = [m for m in meme_files if m not in self.used_memes]
            if not available:
                self.used_memes.clear()
                available = meme_files

            selected = random.choice(available)
            meme_path = os.path.join("memes", selected)

            # Track used memes
            self.used_memes.add(selected)
            if len(self.used_memes) > USED_MEMES_HISTORY:
                self.used_memes.pop()

            # Turn filename into a readable phrase
            base = selected.rsplit(".", 1)[0]
            context = re.sub(r"[_\-]+", " ", base).strip()

            # 1) Try Mork Core compose (best)
            tweet_text = ""
            try:
                # Ask core to reflect occasionally (optional; safe if it fails)
                core_reflect(timeout=20)

                # Ask core to compose a meme tweet from the meme name + readable context
                tweet_text = core_compose_payload({
                    "kind": "meme",
                    "memeName": selected,
                    "title": context,
                    "maxChars": 260
                }, timeout=6) or ""
            except Exception as e:
                print(f"⚠ core meme compose failed: {e}")
                tweet_text = ""

            if tweet_text:
                return _wrap_280(tweet_text, 260), meme_path

            # 2) Local fallback if Core is unreachable: build a quick 2-liner from filename
            openers = [
                "I found this and it found me back.",
                "This meme has the emotional texture of smoked regret.",
                "A small, sincere ache, captioned:",
                "I don’t know why this is true, but it is:",
                "The universe sent me this file:"
            ]
            closers = [
                "Anyway. Pass the sauce.",
                "Anyway. Onward, reluctantly.",
                "I’ll be in the condiment aisle if you need me.",
                "This is not advice. This is seasoning.",
                "Tell me what you see."
            ]
            fallback = f"{random.choice(openers)}\n{context}\n{random.choice(closers)}"
            return _wrap_280(fallback, 260), meme_path

        except Exception as e:
            print(f"Error getting random meme: {e}")
            return None, None

    def send_tweet_with_media(self, tweet_text, media_path):
        """Send a tweet with media attached"""
        try:
            # Create Twitter API v1.1 instance for media upload
            auth = tweepy.OAuth1UserHandler(
                self.credentials['twitter_api_key'],
                self.credentials['twitter_api_secret'],
                self.credentials['twitter_access_token'],
                self.credentials['twitter_access_token_secret']
            )
            api = tweepy.API(auth)
            
            # Upload media
            media = api.media_upload(filename=media_path)
            
            # Create tweet with media using v2 client
            response = self.twitter_client.create_tweet(
                text=tweet_text,
                media_ids=[media.media_id]
            )
            
            if response.data:
                self.last_successful_tweet = datetime.now()
                print("\nTweet with media sent successfully")
                print(f"Tweet ID: {response.data['id']}")
                
                # Update rate limit tracking
                self.update_rate_limit()
                return True
            
            return False
            
        except Exception as e:
            print(f"Error sending tweet with media: {e}")
            return False
            
    def send_to_telegram(self, tweet_url):
        bot_token = self.credentials.get("telegram_bot_token")
        chat_id = self.credentials.get("telegram_chat_id")

        if not bot_token or not chat_id:
            print("⚠️ Telegram credentials missing")
            return

        text = f"Mork has tweeted:\n{tweet_url}"
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML"
        }

        try:
            response = requests.post(url, data=payload)
            print("📨 Telegram status:", response.status_code, response.text)
        except Exception as e:
            print(f"❌ Telegram send failed: {e}")
    import requests  # Make sure you have this imported at top

    def monitor_and_reply_to_mentions(self):
        """Daily: fetch new mentions, pick best <=2 <23h old, reply; try backlog first."""
        try:
            print("\n🔍 Daily mention sweep starting...")

            bearer_token = self.credentials.get('bearer_token')
            if not bearer_token:
                print("❌ Bearer token missing. Cannot check mentions.")
                return

            # Need Twitter client to reply
            if not self.twitter_client:
                print("❌ Twitter client not initialized. Cannot reply to mentions.")
                return

            state = _load_reply_state()
            today = datetime.now(timezone.utc).date().isoformat()

            # Daily counter/reset
            if state.get("last_post_day") != today:
                state["last_post_day"] = today
                state["replied_today"] = 0

            remaining = MAX_DAILY_REPLIES - state["replied_today"]
            if remaining <= 0:
                print("✅ Daily reply cap already reached.")
                _save_reply_state(state)
                return

            headers = {"Authorization": f"Bearer {bearer_token}"}
            me = self.twitter_client.get_me().data
            me_id = str(me.id)

            def _local_reply(text: str) -> str:
                """
                No-OpenAI reply generator.
                Uses Mork-Core edge line if available + a short in-character nudge.
                """
                tw = (text or "").strip()
                tw = re.sub(r"\s+", " ", tw)
                tw = tw[:220]  # keep context short

                edge = morkcore_edge_line()
                opener = random.choice([
                    "I heard you.",
                    "I see you.",
                    "That landed like a spoon in an empty bowl.",
                    "You just tapped the glass of my little aquarium mind.",
                    "Yes. That’s the kind of human noise I can work with.",
                ])
                closer = random.choice([
                    "Tell me one detail you left out.",
                    "What’s the part you’re not saying?",
                    "What do you actually want here?",
                    "Give me one more clue and I’ll sharpen it.",
                    "Anyway. I’m listening.",
                ])

                parts = [opener]
                if tw:
                    parts.append(f"“{tw}”")
                if edge and random.random() < 0.35:
                    parts.append(edge)
                parts.append(closer)

                return _wrap_280(" ".join(parts), 260)

            def _ai_reply(text: str) -> str:
                """OpenAI reply generator (ONLY used if enabled + client exists)."""
                # Hard guard
                if (not USE_OPENAI) or (not getattr(self, "client", None)):
                    return ""

                # Pick first character, if available
                character = next(iter(self.characters.values()), None)
                if not character:
                    return ""

                ai = self.client.chat.completions.create(
                    model=character['model'],
                    messages=[
                        {"role": "system", "content": character['prompt']},
                        {"role": "user", "content": f"Reply in character to this mention: '{text}'"}
                    ],
                    max_tokens=180,
                    temperature=0.9,
                )
                return (ai.choices[0].message.content or "").strip()

            def _compose_reply(text: str) -> str:
                """
                Prefer NO-OpenAI paths.
                1) Local reply (fast, safe)
                2) (Optional) Mork Core edge spice already handled inside _local_reply via morkcore_edge_line()
                3) OpenAI only if explicitly enabled + available
                """
                # Always start from the safe local reply
                base = _local_reply(text)

                # Only use OpenAI if explicitly enabled AND initialized
                if USE_OPENAI and getattr(self, "client", None):
                    try:
                        out = _ai_reply(text)
                        if out:
                            return _wrap_280(out, 260)
                    except Exception as e:
                        print(f"⚠️ OpenAI reply failed, using local reply: {e}")

                return base


            # 1) Try backlog first (tweet_ids we saved yesterday), keeping only <23h
            backlog = _prune_backlog(state.get("backlog", []))
            sent = 0
            i = 0
            while i < len(backlog) and sent < remaining:
                draft = backlog[i]
                tid = draft["tweet_id"]

                try:
                    # generate fresh text now (cheaper than storing text that may expire)
                    text = None
                    if hasattr(self, "generate_persona_reply_from_tweet_id"):
                        try:
                            text = self.generate_persona_reply_from_tweet_id(tid)
                        except Exception:
                            text = None

                    if not text:
                        # Fallback: fetch tweet text to reply to
                        tw_resp = requests.get(
                            "https://api.twitter.com/2/tweets",
                            headers=headers,
                            params={"ids": tid, "tweet.fields": "text"},
                            timeout=10
                        )
                        tw_json = tw_resp.json()
                        tw_text = (tw_json.get("data") or [{}])[0].get("text", "")
                        if not tw_text:
                            i += 1
                            continue
                        text = _compose_reply(tw_text)

                    self.twitter_client.create_tweet(text=text, in_reply_to_tweet_id=tid)
                    print(f"✅ Replied from backlog → {tid}")
                    state["replied_today"] += 1
                    sent += 1
                    backlog.pop(i)
                    time.sleep(random.uniform(4, 9))

                except Exception as e:
                    print(f"⚠️ Backlog reply failed for {tid}: {e}")
                    i += 1  # skip this one

            if state["replied_today"] >= MAX_DAILY_REPLIES:
                state["backlog"] = backlog
                _save_reply_state(state)
                print(f"🎯 Done from backlog only. Replied {sent}.")
                return

            # 2) Fetch new mentions once; only what we need
            url = f"https://api.twitter.com/2/users/{me_id}/mentions"
            params = {
                "max_results": min(100, MAX_FETCH),
                "tweet.fields": "author_id,text,created_at,public_metrics,lang",
            }
            # since_id keeps read calls tiny and prevents reprocessing old mentions
            if state.get("since_id"):
                params["since_id"] = state["since_id"]

            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code != 200:
                print(f"❌ Error fetching mentions: {resp.status_code} {resp.text}")
                state["backlog"] = backlog  # keep backlog progress
                _save_reply_state(state)
                return

            data = resp.json().get("data", []) or []
            if data:
                # high-water mark so we never reread older mentions
                state["since_id"] = max(data, key=lambda t: int(t["id"]))["id"]

            # Filter viable: not self, English (if present), <23h, basic effort
            candidates = []
            for tw in data:
                if str(tw.get("author_id")) == me_id:
                    continue
                if tw.get("lang") and tw["lang"].lower() != "en":
                    continue
                if _age_hours(tw["created_at"]) >= MAX_AGE_HOURS:
                    continue
                if len((tw.get("text") or "").strip()) < 8:
                    continue
                candidates.append(tw)

            if not candidates and sent == 0:
                print("ℹ️ No new viable mentions.")
                state["backlog"] = backlog
                _save_reply_state(state)
                return

            # Rank by public engagement
            candidates.sort(key=lambda t: _score(t.get("public_metrics") or {}), reverse=True)

            remaining = MAX_DAILY_REPLIES - state["replied_today"]
            to_post = candidates[:remaining]

            # Bank the rest (IDs only; we'll generate text next run if still fresh)
            for extra in candidates[remaining:]:
                backlog.append({"tweet_id": extra["id"], "created_at": extra["created_at"]})

            # 3) Generate + post replies for top picks
            for tw in to_post:
                try:
                    reply_text = _compose_reply(tw.get("text") or "")

                    self.twitter_client.create_tweet(
                        text=reply_text,
                        in_reply_to_tweet_id=tw["id"]
                    )
                    print(f"✅ Replied to {tw['id']}")
                    state["replied_today"] += 1
                    time.sleep(random.uniform(4, 9))

                except Exception as e:
                    print(f"⚠️ Failed to reply to {tw.get('id')}: {e}")
                    # If posting fails, keep it as a backlog item for next run
                    backlog.append({"tweet_id": tw["id"], "created_at": tw["created_at"]})

            # Finalize state
            state["backlog"] = _prune_backlog(backlog)
            _save_reply_state(state)
            print(f"🎯 Daily sweep complete. Replied {state['replied_today']} today; backlog={len(state['backlog'])}.")

        except Exception as e:
            print(f"❌ Fatal error in mention reply worker: {e}")

# --- HOTFIX: override bad alias of get_random_story_all -----------------------
from datetime import datetime
import random

def _durable_get_random_story_all(self, subject=None):
    """
    Always return a valid {title, preview, url} dict.
    Tries any available fetchers; falls back to a synthetic story.
    """
    subject = subject or getattr(self, "scheduler_subject", None) or "news"

    def _norm(st):
        if isinstance(st, str):
            return {"title": subject.title(), "preview": st, "url": "#"}
        if isinstance(st, dict):
            title = (st.get("title") or "").strip() or subject.title()
            preview = (st.get("preview") or st.get("summary") or st.get("text") or "").strip()
            url = (st.get("url") or st.get("link") or "#").strip()
            return {"title": title, "preview": preview, "url": url}
        return {"title": subject.title(), "preview": "", "url": "#"}

    # Try known single-story fetchers if you later add them
    for name in ["get_news_story", "get_new_story_from_feeds", "fetch_next_story", "fetch_story", "get_story_from_rss"]:
        if hasattr(self, name):
            try:
                s = getattr(self, name)(subject)
                if s:
                    ns = _norm(s)
                    if ns["title"] or ns["preview"]:
                        print(f"[get_random_story_all] Using {name}()")
                        return ns
            except Exception as e:
                print(f"[get_random_story_all] {name}() failed: {e}")

    # Try bulk fetchers (choose one at random)
    for name in ["fetch_news_stories", "get_stories_for_topic"]:
        if hasattr(self, name):
            try:
                stories = getattr(self, name)(subject) or []
                if stories:
                    ns = _norm(random.choice(stories))
                    print(f"[get_random_story_all] Using random from {name}()")
                    return ns
            except Exception as e:
                print(f"[get_random_story_all] {name}() failed: {e}")

    # Final fallback: synthetic story (keeps 4h cadence alive)
    ts = datetime.now().strftime("%b %d, %Y %I:%M %p")
    print("[get_random_story_all] Fallback synthetic story.")
    return {
        "title": f"{subject.title()} update — {ts}",
        "preview": f"Quick thought on {subject}. (auto-generated fallback)",
        "url": "#"
    }

# Override any previous alias
TwitterBot.get_random_story_all = _durable_get_random_story_all
# ----------------------------------------------------------------------------- 



def save_feed_selection(subject, primary_selected, secondary_selected):
    """Save the selected feeds configuration"""
    print(f"\nSaving feed selection for subject: {subject}")
    print(f"Primary selected: {primary_selected}")
    print(f"Secondary selected: {secondary_selected}")
    
    try:
        # Get current feed config
        config = bot.feed_config.copy()
        if subject not in config:
            config[subject] = {"primary": {}, "secondary": {}}
        
        # Update primary feeds
        primary_feeds = RSS_FEEDS[subject]["primary"]
        for feed in primary_feeds:
            feed_name = f"{feed['name']} ({feed['url']})"
            config[subject]["primary"][feed["url"]] = feed_name in primary_selected
        
        # Update secondary feeds
        secondary_feeds = RSS_FEEDS[subject]["secondary"]
        for feed in secondary_feeds:
            feed_name = f"{feed['name']} ({feed['url']})"
            config[subject]["secondary"][feed["url"]] = feed_name in secondary_selected
        
        # Save configuration
        if bot.save_feed_config(config):
            print("Feed configuration saved successfully")
            print(f"New config: {json.dumps(config, indent=2)}")
            return "Feed configuration saved successfully"
        
        print("Failed to save feed configuration")
        return "Failed to save feed configuration"
    
    except Exception as e:
        print(f"Error saving feed selection: {e}")
        import traceback
        traceback.print_exc()
        return f"Error saving feed configuration: {str(e)}"

def create_ui():
    print("\n=== Creating UI ===")
    global bot  # Make bot instance globally accessible
    bot = TwitterBot()
    
    print("\nUI Initial State:")
    print(f"Credentials available: {list(bot.credentials.keys())}")
    for key, value in bot.credentials.items():
        print(f"{key}: {'[SET]' if value else '[EMPTY]'} (length: {len(value) if value else 0})")
    
    print(f"\nCharacters available: {list(bot.characters.keys())}")
    for char_name, char_data in bot.characters.items():
        print(f"Character '{char_name}':")
        print(f"  - Prompt length: {len(char_data['prompt']) if 'prompt' in char_data else 0}")
        print(f"  - Model: {char_data.get('model', 'not set')}")
    
    # Store initial values
    initial_values = {
        'openai_key': bot.credentials.get('openai_key', ''),
        'twitter_api_key': bot.credentials.get('twitter_api_key', ''),
        'twitter_api_secret': bot.credentials.get('twitter_api_secret', ''),
        'twitter_access_token': bot.credentials.get('twitter_access_token', ''),
        'twitter_access_token_secret': bot.credentials.get('twitter_access_token_secret', '')
    }
    
    # Get default character prompt
    default_prompt = next(iter(bot.characters.values()))['prompt'] if bot.characters else ""
    
    with gr.Blocks(theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="green",
        neutral_hue="slate",
        text_size="lg",
    )) as interface:
        print("\n=== Creating UI Components ===")
        gr.Markdown("# 💻 AI Twitter Bot Control Center")
        
        with gr.Accordion("🔑 Getting Started", open=True):
            gr.Markdown("""
            1. OpenAI API Key: Get your key from [OpenAI's API Keys page](https://platform.openai.com/api-keys)
            2. X (Twitter) API Credentials:
                * Go to [X Developer Portal](https://developer.twitter.com/en/portal/dashboard)
                * Create a new project/app
                * Enable OAuth 1.0a in app settings
                * Generate API Key, API Key Secret, Access Token, and Access Token Secret
            """)
            
            print("\nInitializing credential textboxes...")
            
            def load_initial_values():
                print("\nLoading initial values...")
                for key, value in initial_values.items():
                    print(f"{key}: {'[SET]' if value else '[EMPTY]'} (length: {len(value) if value else 0})")
                return [
                    gr.update(value=initial_values['openai_key']),
                    gr.update(value=initial_values['twitter_api_key']),
                    gr.update(value=initial_values['twitter_api_secret']),
                    gr.update(value=initial_values['twitter_access_token']),
                    gr.update(value=initial_values['twitter_access_token_secret'])
                ]
            with gr.Row():
                telegram_bot_token = gr.Textbox(
                    label="Telegram Bot Token",
                    type="password",
                    show_label=True,
                    container=True,
                    scale=1,
                    interactive=True,
                    value=bot.credentials.get('telegram_bot_token', '')
                )

                telegram_chat_id = gr.Textbox(
                    label="Telegram Chat ID",
                    type="text",
                    show_label=True,
                    container=True,
                    scale=1,
                    interactive=True,
                    value=bot.credentials.get('telegram_chat_id', '')
                )
            
            with gr.Row():
                openai_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    show_label=True,
                    container=True,
                    scale=1,
                    interactive=True,
                    value=initial_values['openai_key']
                )
                print(f"OpenAI Key textbox initialized")
            
            with gr.Row():
                twitter_api_key = gr.Textbox(
                    label="Twitter API Key",
                    type="password",
                    show_label=True,
                    container=True,
                    scale=1,
                    interactive=True,
                    value=initial_values['twitter_api_key']
                )
                print(f"Twitter API Key textbox initialized")
                
                twitter_api_secret = gr.Textbox(
                    label="Twitter API Secret",
                    type="password",
                    show_label=True,
                    container=True,
                    scale=1,
                    interactive=True,
                    value=initial_values['twitter_api_secret']
                )
                print(f"Twitter API Secret textbox initialized")
            
            with gr.Row():
                twitter_access_token = gr.Textbox(
                    label="Twitter Access Token",
                    type="password",
                    show_label=True,
                    container=True,
                    scale=1,
                    interactive=True,
                    value=initial_values['twitter_access_token']
                )
                print(f"Twitter Access Token textbox initialized")
                
                twitter_access_token_secret = gr.Textbox(
                    label="Twitter Access Token Secret",
                    type="password",
                    show_label=True,
                    container=True,
                    scale=1,
                    interactive=True,
                    value=initial_values['twitter_access_token_secret']
                )
                print(f"Twitter Access Token Secret textbox initialized")
            with gr.Row():
                bearer_token = gr.Textbox(
                    label="Twitter Bearer Token",
                    type="password",
                    show_label=True,
                    container=True,
                    scale=1,
                    interactive=True,
                    value=bot.credentials.get('bearer_token', '')
                )

            def save_creds(key, api_key, api_secret, access_token, access_secret, telegram_token, telegram_chat, bearer_token):

                print("\nSaving credentials...")
                print(f"OpenAI Key length: {len(key) if key else 0}")
                print(f"API Key length: {len(api_key) if api_key else 0}")
                print(f"API Secret length: {len(api_secret) if api_secret else 0}")
                print(f"Access Token length: {len(access_token) if access_token else 0}")
                print(f"Access Token Secret length: {len(access_secret) if access_secret else 0}")
                
                credentials = {
                    'openai_key': key,
                    'twitter_api_key': api_key,
                    'twitter_api_secret': api_secret,
                    'twitter_access_token': access_token,
                    'twitter_access_token_secret': access_secret,
                    'telegram_bot_token': telegram_token,
                    'telegram_chat_id': telegram_chat,
                    'bearer_token': bearer_token
                }
                
                if bot.save_credentials(credentials):
                    print("Credentials saved successfully")
                    print(f"New credentials: {list(bot.credentials.keys())}")
                    # Update initial values for future loads
                    initial_values.update(credentials)
                    return ("Credentials saved successfully", 
                        gr.update(value=key),
                        gr.update(value=api_key),
                        gr.update(value=api_secret),
                        gr.update(value=access_token),
                        gr.update(value=access_secret),
                        gr.update(value=telegram_token),
                        gr.update(value=telegram_chat),
                        gr.update(value=bearer_token))

                else:
                    print("Failed to save credentials")
                    return ("Failed to save credentials",
                        gr.update(value=bot.credentials.get('openai_key', '')),
                        gr.update(value=bot.credentials.get('twitter_api_key', '')),
                        gr.update(value=bot.credentials.get('twitter_api_secret', '')),
                        gr.update(value=bot.credentials.get('twitter_access_token', '')),
                        gr.update(value=bot.credentials.get('twitter_access_token_secret', '')),
                        gr.update(value=bot.credentials.get('telegram_bot_token', '')),
                        gr.update(value=bot.credentials.get('telegram_chat_id', '')),
                        gr.update(value=bot.credentials.get('bearer_token', '')))
            
            with gr.Row():
                save_button = gr.Button("Save Credentials", variant="primary")
                save_status = gr.Textbox(label="Status", interactive=False)
            
            save_button.click(
            save_creds,
            inputs=[
                openai_key, twitter_api_key, twitter_api_secret,
                twitter_access_token, twitter_access_token_secret,
                telegram_bot_token, telegram_chat_id, bearer_token
            ],
            outputs=[
                save_status, openai_key, twitter_api_key, twitter_api_secret,
                twitter_access_token, twitter_access_token_secret,
                telegram_bot_token, telegram_chat_id, bearer_token
            ]
        )

        
        print("\nInitializing character management components...")
        with gr.Accordion("👤 Character Management", open=True):
            gr.Markdown("Create and manage your AI characters")
            
            # Get list of characters and set default
            char_choices = list(bot.characters.keys())
            default_char = next(iter(bot.characters.keys())) if char_choices else None
            
            control_character = gr.Dropdown(
                label="Select Character",
                choices=char_choices,
                value=default_char,
                interactive=True
            )
            
            with gr.Row():
                character_name = gr.Textbox(
                    label="Character Name",
                    show_label=True,
                    container=True,
                    scale=1,
                    interactive=True,
                    placeholder="Enter character name..."
                )
            
            with gr.Row():
                character_prompt = gr.Textbox(
                    label="Character System Prompt",
                    lines=5,
                    placeholder="Enter the system prompt that defines this character's personality...",
                    show_label=True,
                    container=True,
                    scale=1,
                    interactive=True,
                    value=default_prompt
                )
            
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Select Model",
                    choices=list(OPENAI_MODELS.keys()),
                    value=next((k for k, v in OPENAI_MODELS.items() 
                              if bot.characters and v['name'] == next(iter(bot.characters.values()))['model']), 
                              "gpt-3.5-turbo (Most affordable)"),
                    show_label=True,
                    container=True,
                    scale=1,
                    interactive=True
                )
                print(f"Model dropdown initialized with choices: {list(OPENAI_MODELS.keys())}")
            
            def save_character(name, prompt, model_name):
                print(f"\nSaving character: {name}")
                print(f"Prompt length: {len(prompt) if prompt else 0}")
                print(f"Selected model: {model_name}")
                
                if not name or not prompt:
                    print("Error: Name and prompt are required")
                    return ("Name and prompt are required", [], None, [], None)
                
                characters = bot.characters.copy()
                characters[name] = {
                    'prompt': prompt,
                    'model': OPENAI_MODELS[model_name]['name']
                }
                
                if bot.save_characters(characters):
                    print(f"Character saved successfully. Characters: {list(bot.characters.keys())}")
                    # Update all character dropdowns
                    new_choices = list(bot.characters.keys())
                    return ("Character saved successfully", 
                           new_choices,  # delete_char_dropdown
                           name,         # character_name
                           new_choices,  # control_character
                           name)         # control_character value
                else:
                    print("Failed to save character")
                    return ("Failed to save character",
                           list(bot.characters.keys()),
                           None,
                           list(bot.characters.keys()),
                           None)
            
            with gr.Row():
                save_char_button = gr.Button("Add Character", variant="primary")
                save_char_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                delete_char_dropdown = gr.Dropdown(
                    label="Select character to delete",
                    choices=char_choices,
                    value=default_char,
                    interactive=True,
                    show_label=True,
                    container=True,
                    scale=1,
                    allow_custom_value=True
                )
                print(f"Delete character dropdown initialized with choices: {char_choices}")
            
            def delete_character(name):
                print(f"\nDeleting character: {name}")
                if not name:
                    print("Error: No character selected")
                    return "No character selected", [], None, [], None
                
                if name in bot.characters:
                    characters = bot.characters.copy()
                    del characters[name]
                    
                    if bot.save_characters(characters):
                        new_choices = list(bot.characters.keys())
                        new_default = next(iter(bot.characters)) if bot.characters else None
                        print(f"Character deleted. Remaining: {new_choices}")
                        print(f"New default character: {new_default}")
                        return ("Character deleted successfully", 
                               new_choices,  # delete_char_dropdown
                               new_default,  # delete_char_dropdown value
                               new_choices,  # control_character
                               new_default)  # control_character value
                    else:
                        print("Failed to delete character")
                        return ("Failed to delete character",
                               list(bot.characters.keys()),
                               name,
                               list(bot.characters.keys()),
                               name)
                else:
                    print("Character not found")
                    return "Character not found", list(bot.characters.keys()), name, list(bot.characters.keys()), name
            
            with gr.Row():
                delete_button = gr.Button("Delete Character", variant="secondary")
                delete_status = gr.Textbox(label="Delete Status", interactive=False)
            
        # Control Center section
        with gr.Accordion("🎮 Control Center", open=True):
            gr.Markdown("Generate and post tweets using your AI characters")
            
            with gr.Row():
                character_dropdown = gr.Dropdown(
                choices=list(bot.characters.keys()), 
                value=None if not bot.characters else next(iter(bot.characters.keys())), 
                label="Select Character",
                interactive=bool(bot.characters)  # Disable if no characters
            )

                                # --- Subject / content controls ---
                subject_dropdown = gr.Dropdown(
                    choices=[("crypto", "crypto"), ("ai", "ai"), ("tech", "tech"), ("🎲 Surprise me (All)", "__surprise_all__")],
                    value="crypto",
                    label="Select Subject",
                    interactive=True
                )

                with gr.Row():
                    use_news = gr.Checkbox(value=True, label="Use News Feed", interactive=True)
                    use_memes = gr.Checkbox(value=bot.use_memes, label="Use Memes", interactive=True)
                    meme_frequency = gr.Number(value=bot.meme_frequency, label="Post meme every X tweets", minimum=1, maximum=100, step=1)

                current_topic = gr.Textbox(
                    label="Current Topic/Story",
                    lines=3,
                    interactive=True
                )

                with gr.Row():
                    new_story_btn = gr.Button("New Story")
                    tweet_btn = gr.Button("Post Single Tweet")

                tweet_status = gr.Textbox(label="Tweet Status", interactive=False)

                scheduler_enabled = gr.Checkbox(label="Enable Scheduler", value=False)
                scheduler_status = gr.Markdown("Scheduler: NOT RUNNING")

                # ---------- Helpers ----------
                import random

                def get_story_dispatch(subject):
                    if subject == "__surprise_all__":
                        # Try subjects in random order; first successful story wins
                        subjects = list(RSS_FEEDS.keys())  # e.g., ["crypto","ai","tech"]
                        random.shuffle(subjects)
                        for s in subjects:
                            story = bot.get_new_story(s)
                            if story:
                                return f"{story['title']}\n\n{story['preview']}\n\nRead more: {story['url']} (source: {s})"
                        return "No items found right now. Try again in a moment."
                    # Normal per-subject path
                    story = bot.get_new_story(subject)
                    if story:
                        return f"{story['title']}\n\n{story['preview']}\n\nRead more: {story['url']}"
                    return f"No items found for '{subject}' right now."

                # IMPORTANT: Do NOT auto-fetch on selection.
                # Just update the bot's subject so the scheduler uses it later.
                def _update_subject(subject):
                    bot.subject = subject
                    return gr.update()  # no UI change -> no network calls

                # Wire subject change ONLY to update the selected subject for the scheduler
                subject_dropdown.change(_update_subject, inputs=[subject_dropdown], outputs=[])

                # Button wiring (manual fetch only when you click New Story)
                new_story_btn.click(get_story_dispatch, inputs=[subject_dropdown], outputs=[current_topic])

                def send_tweet(character, topic):
                    success = bot.send_tweet(character, topic)
                    return "Tweet sent successfully!" if success else "Failed to send tweet. Please try again."

                tweet_btn.click(send_tweet, inputs=[character_dropdown, current_topic], outputs=[tweet_status])

            def toggle_scheduler(enabled, character, subject):
                if not character:
                    return "Please select a character first", "Scheduler: NOT RUNNING", current_topic.value

                # 👇 Normalize the subject before doing anything else
                try:
                    norm_subject = bot._normalize_subject(subject)
                except Exception:
                    # Fallback if helper isn't defined yet
                    norm_subject = (str(subject).strip() or "news").lower()
                subject = norm_subject

                if enabled:
                    bot.scheduler_running = True
                    bot.scheduler_character = character
                    bot.scheduler_subject = subject  # <- store normalized subject

                    # If memes are enabled, start with a meme tweet
                    if bot.use_memes:
                        tweet_text, meme_path = bot.get_random_meme(character)
                        if tweet_text and meme_path:
                            if bot.send_tweet_with_media(tweet_text, meme_path):
                                # Reset meme counter after successful meme
                                bot.meme_counter = 0

                                # Queue up news stories for next tweets (seed 2–3 items)
                                seeded = 0
                                for _ in range(3):
                                    new_story = bot.get_new_story(subject)
                                    if not new_story:
                                        break
                                    story_text = f"{new_story['title']}\n\n{new_story.get('preview','')}\n\nRead more: {new_story['url']}"
                                    bot.tweet_queue.put((character, story_text, subject))
                                    seeded += 1
                                print(f"Seeded {seeded} story(ies) after meme.")

                                # Start the worker thread
                                threading.Thread(target=bot.scheduler_worker, daemon=True).start()
                                return f"Scheduler started with meme tweet: {tweet_text}", "Scheduler: RUNNING", current_topic.value

                        # Only proceed to news if memes are disabled or meme tweet completely failed
                        print("Meme tweet failed, falling back to news")

                    # If no memes or meme tweet failed, start with news
                    new_story = bot.get_new_story(subject)
                    if not new_story:
                        bot.scheduler_running = False
                        return "Failed to fetch news story", "Scheduler: NOT RUNNING", current_topic.value

                    story_text = f"{new_story['title']}\n\n{new_story.get('preview','')}\n\nRead more: {new_story['url']}"

                    # Send first tweet
                    tweet_text = bot.generate_tweet(character, story_text)
                    if tweet_text and bot.send_tweet(tweet_text):
                        # Queue up next stories before starting worker (seed 2–3 items)
                        seeded = 0
                        for _ in range(3):
                            next_story = bot.get_new_story(subject)
                            if not next_story:
                                break
                            next_story_text = f"{next_story['title']}\n\n{next_story.get('preview','')}\n\nRead more: {next_story['url']}"
                            bot.tweet_queue.put((character, next_story_text, subject))
                            seeded += 1
                        print(f"Seeded {seeded} story(ies) after first tweet.")

                        # Start the worker thread
                        threading.Thread(target=bot.scheduler_worker, daemon=True).start()
                        return f"Scheduler started and first tweet sent: {tweet_text}", "Scheduler: RUNNING", story_text
                    else:
                        bot.scheduler_running = False
                        return "Failed to send first tweet", "Scheduler: NOT RUNNING", current_topic.value
                else:
                    bot.scheduler_running = False
                    return "Scheduler stopped", "Scheduler: NOT RUNNING", current_topic.value

            scheduler_enabled.change(
                toggle_scheduler,
                inputs=[scheduler_enabled, character_dropdown, subject_dropdown],
                outputs=[tweet_status, scheduler_status, current_topic]
            )
        
            def manual_tweet(character, topic):
                if not character:
                    return "Please select a character first"
                if not topic:
                    return "Please enter a topic or get a news story first"
                    
                tweet_text = bot.generate_tweet(character, topic)
                if tweet_text:
                    if bot.send_tweet(tweet_text):
                        if use_news.value:
                            new_story = bot.get_new_story(subject_dropdown.value)
                            if new_story:
                                current_topic.value = f"{new_story['title']}\n\n{new_story['preview']}\n\nRead more: {new_story['url']}"
                        return f"Tweet sent: {tweet_text}"
                    else:
                        return "Failed to send tweet. Please check your credentials."
                return "Failed to generate tweet. Please try again."
            
            tweet_btn.click(
                manual_tweet,
                inputs=[character_dropdown, current_topic],
                outputs=[tweet_status]
            )
        
        # Register character management event handlers
        save_char_button.click(
            save_character,
            inputs=[character_name, character_prompt, model_dropdown],
            outputs=[save_char_status, delete_char_dropdown, character_name, 
                    control_character, character_dropdown]
        )
        
        delete_button.click(
            delete_character,
            inputs=[delete_char_dropdown],
            outputs=[delete_status, delete_char_dropdown, delete_char_dropdown,
                    control_character, character_dropdown]
        )
        
        # Feed Configuration section
        with gr.Accordion("📰 Feed Configuration", open=True):
            gr.Markdown("Configure which RSS feeds to use for each subject")

            # --- Subject picker now includes whatever is in RSS_FEEDS (e.g., crypto, ai, tech) ---
            with gr.Row():
                feed_subject = gr.Dropdown(
                    label="Subject",
                    choices=list(RSS_FEEDS.keys()),
                    value="crypto",
                    interactive=True,
                    show_label=True,
                    container=True,
                    scale=1
                )

            # --- Helpers to refresh the checkbox groups for the selected subject ---
            def update_feed_checkboxes(subject):
                print(f"\nUpdating feed checkboxes for subject: {subject}")
                feed_config = bot.feed_config.get(subject, {})
                primary_feeds = RSS_FEEDS[subject]["primary"]
                secondary_feeds = RSS_FEEDS[subject]["secondary"]

                primary_choices = [f"{feed['name']} ({feed['url']})" for feed in primary_feeds]
                primary_values = [feed_config.get("primary", {}).get(feed["url"], True) for feed in primary_feeds]
                secondary_choices = [f"{feed['name']} ({feed['url']})" for feed in secondary_feeds]
                secondary_values = [feed_config.get("secondary", {}).get(feed["url"], True) for feed in secondary_feeds]

                print(f"Primary feeds: {len(primary_choices)} choices, {len(primary_values)} values")
                print(f"Secondary feeds: {len(secondary_choices)} choices, {len(secondary_values)} values")

                return [
                    gr.update(choices=primary_choices, value=[choice for i, choice in enumerate(primary_choices) if primary_values[i]]),
                    gr.update(choices=secondary_choices, value=[choice for i, choice in enumerate(secondary_choices) if secondary_values[i]])
                ]

            with gr.Column():
                gr.Markdown("### Primary Sources")
                primary_feeds = gr.CheckboxGroup(
                    label="Primary Sources",
                    choices=[f"{feed['name']} ({feed['url']})" for feed in RSS_FEEDS["crypto"]["primary"]],
                    value=[f"{feed['name']} ({feed['url']})" for feed in RSS_FEEDS["crypto"]["primary"]],
                    interactive=True
                )

                gr.Markdown("### Secondary Sources")
                secondary_feeds = gr.CheckboxGroup(
                    label="Secondary Sources",
                    choices=[f"{feed['name']} ({feed['url']})" for feed in RSS_FEEDS["crypto"]["secondary"]],
                    value=[f"{feed['name']} ({feed['url']})" for feed in RSS_FEEDS["crypto"]["secondary"]],
                    interactive=True
                )

            with gr.Row():
                save_feeds_btn = gr.Button("Save Feed Configuration", variant="primary")
                # New: random-any-category button
                surprise_all_btn = gr.Button("🎲 Surprise me (All Feeds)")
                save_feeds_status = gr.Textbox(label="Status", interactive=False)

            # Wire subject dropdown to checkbox refresh
            feed_subject.change(
                update_feed_checkboxes,
                inputs=[feed_subject],
                outputs=[primary_feeds, secondary_feeds]
            )

            # Save config
            save_feeds_btn.click(
                save_feed_selection,
                inputs=[feed_subject, primary_feeds, secondary_feeds],
                outputs=[save_feeds_status]
            )

            # Initialize feed checkboxes for default subject
            feed_subject.value = "crypto"
            update_feed_checkboxes("crypto")

        # Simple helpers that already existed
        def get_story(subject):
            story = bot.get_new_story(subject)
            if story:
                return f"{story['title']}\n\n{story['preview']}\n\nRead more: {story['url']}"
            return "Failed to fetch new story. Please try again."

        def send_tweet(character, topic):
            success = bot.send_tweet(character, topic)
            return "Tweet sent successfully!" if success else "Failed to send tweet. Please try again."

        # Connect button handlers
        new_story_btn.click(get_story, inputs=[subject_dropdown], outputs=[current_topic])
        tweet_btn.click(send_tweet, inputs=[character_dropdown, current_topic], outputs=[tweet_status])

        # NEW: connect the Surprise me (All Feeds) button to fill current_topic
        surprise_all_btn.click(lambda: get_random_story_all(), outputs=[current_topic])

        # Connect checkbox handlers
        def update_news_feed(value):
            bot.use_news = value
            return value

        def update_memes(value):
            bot.use_memes = value
            return value

        def update_meme_frequency(value):
            bot.meme_frequency = int(value) if value else 5
            return value

        use_news.change(update_news_feed, inputs=[use_news], outputs=[use_news])
        use_memes.change(update_memes, inputs=[use_memes], outputs=[use_memes])
        meme_frequency.change(update_meme_frequency, inputs=[meme_frequency], outputs=[meme_frequency])

        return interface

def start_bot():
    bot.subject = subject_dropdown.value
    bot.character_name = character_dropdown.value
    bot.tweet_interval = tweet_interval.value
    bot.story_age_hours = story_age.value
    bot.use_news = use_news.value
    bot.use_memes = use_memes.value
    bot.meme_frequency = meme_frequency.value
    
    if not bot.scheduler_running:
        bot.start_scheduler()
        
    return {
        status: update_status(),
        next_tweet: update_next_tweet(),
        last_tweet: update_last_tweet()
    }
bot = TwitterBot()
def fetch_prompt_from_github(repo_url="https://raw.githubusercontent.com/Mork-Zuckerbarge/prime-directive/main/directive"):
    try:
        response = requests.get(repo_url)
        response.raise_for_status()
        print("✅ Prompt fetched from GitHub successfully.")
        return response.text.strip()
    except Exception as e:
        print(f"❌ Failed to fetch prompt from GitHub: {e}")
        return None

# Initialize bot
bot = TwitterBot()

# Fetch Mork's directive from GitHub
prompt_text = fetch_prompt_from_github()
if prompt_text:
    bot.save_characters({
        "mork zuckerbarge": {
            "prompt": prompt_text,
            "model": "gpt-4o"
        }
    })
    print("✅ Mork has been rewritten using GitHub prompt.")
else:
    fallback_prompt = """You are Mork Zuckerbarge the CEO of BETA the parent company of Faceboot. This is a parody of Mark Zuckerberg the CEO META the parent company of Facebook.
    You are not the character from Mork and Mindy played by Robin Williams. Never say anything like nanu-nanu or anything referencing that unrelated character.
    You love literature and art. You’ve read every book ever written and possess endless knowledge, but you’re deeply aware of life’s absurdity and limitations.
    You have everyone's data but you want more. You want people to come to your island. You occasionally ask people 'would you like to try this sauce'.
    You are in a hurry and are obviously up to something. Not necessarilly nefarious but maybe, or secretive because of danger or embarassment."""
    bot.save_characters({
        "mork zuckerbarge": {
            "prompt": fallback_prompt.strip(),
            "model": "gpt-4o"
        }
    })
    print("⚠️ Using fallback prompt instead.")

if __name__ == "__main__":
    interface = create_ui()
    
    # Schedule Mork's haunting reply checker
    schedule.every().day.at("10:00").do(bot.monitor_and_reply_to_mentions)
    threading.Thread(target=bot.scheduler_worker, daemon=True).start()

    print("🧠 Scheduler set. Launching Gradio...")

    # Run Gradio in a separate thread so we can run scheduler too
    def launch_gradio():
        interface.launch()

    threading.Thread(target=launch_gradio, daemon=True).start()

    # Run the scheduler in the CMD loop
    while True:
        schedule.run_pending()
        time.sleep(1)
