#!/usr/bin/env python3
"""
Jarvis Ultimate Phone Agent v11.0
Advanced Termux/Android assistant, written from scratch.

What it can do:
- Learn from public internet on demand: Wikipedia, DuckDuckGo, direct URLs, public metadata, and search links.
- Optional Gemini and Claude API support.
- Safe phone control through Termux:API commands, only with your Android permissions.
- Background reminder/learning loop with --daemon.
- Human intelligence/emotion support, professional math, notes, tasks, memory, files.

Limits:
- No app can contain "all internet data" locally. This agent learns/searches public internet when needed.
- Android does not allow full phone control without permissions/root. This uses legal Termux:API controls only.
- It will not hack, steal, spy, bypass security, or control other people's devices.
"""

from __future__ import annotations

import ast
import base64
import datetime as dt
import difflib
import hashlib
import html
import json
import math
import os
import platform
import re
import secrets
import shutil
import socket
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Tuple

APP_VERSION = "11.0"
APP_DIR = Path.home() / "jarvis_ultimate_phone_agent_data"
FILES_DIR = APP_DIR / "files"
CONFIG_FILE = APP_DIR / "config.json"
MEMORY_FILE = APP_DIR / "memory.json"
KNOWLEDGE_FILE = APP_DIR / "knowledge.json"
NOTES_FILE = APP_DIR / "notes.txt"
TODO_FILE = APP_DIR / "todo.json"
REMINDER_FILE = APP_DIR / "reminders.json"
LEARN_JOBS_FILE = APP_DIR / "learning_jobs.json"
LOG_FILE = APP_DIR / "jarvis.log"

# =============================================================================
# Core helpers
# =============================================================================

def now_text() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_spaces(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return clean_spaces(text)


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def log_event(text: str) -> None:
    try:
        APP_DIR.mkdir(exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[{now_text()}] {text}\n")
    except Exception:
        pass


def strip_html(raw: str) -> str:
    raw = str(raw or "")
    raw = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style.*?>.*?</style>", " ", raw)
    raw = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</p>|</div>|</li>|</h1>|</h2>|</h3>|</tr>", "\n", raw)
    text = re.sub(r"(?is)<[^>]+>", " ", raw)
    return clean_spaces(html.unescape(text))


def extract_title(raw: str) -> str:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw or "")
    return strip_html(m.group(1))[:180] if m else ""


def extract_meta(raw: str) -> str:
    patterns = [
        r'(?is)<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']',
        r'(?is)<meta\s+content=["\'](.*?)["\']\s+name=["\']description["\']',
        r'(?is)<meta\s+property=["\']og:description["\']\s+content=["\'](.*?)["\']',
        r'(?is)<meta\s+property=["\']og:title["\']\s+content=["\'](.*?)["\']',
    ]
    found = []
    for p in patterns:
        m = re.search(p, raw or "")
        if m:
            found.append(clean_spaces(html.unescape(m.group(1)))[:700])
    return " ".join(x for x in found if x)


def split_sentences(text: str) -> List[str]:
    text = clean_spaces(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if 25 <= len(p.strip()) <= 800]


def keywords(text: str) -> List[str]:
    stop = {"the","a","an","is","are","was","were","who","what","why","how","when","where","which","to","of","in","on","for","and","or","by","with","about","tell","me","please","give","solve","answer","his","her","name","current","latest","from","into","your","you","my","learn","study","research","explain","online","internet","search","google","youtube","facebook","instagram"}
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    return [w for w in words if len(w) > 2 and w not in stop]


def summarize(text: str, query: str = "", max_sentences: int = 8) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return clean_spaces(text)[:2600]
    keys = keywords(query)
    scored = []
    for i, sent in enumerate(sentences):
        low = sent.lower()
        score = sum(4 for k in keys if k in low)
        if any(x in low for x in [" is "," are "," means "," refers to "," defined as"," used for"," consists of"," current"," incumbent"]):
            score += 1
        if i < 5:
            score += 1
        scored.append((score, i, sent))
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = sorted(scored[:max_sentences], key=lambda x: x[1])
    return " ".join(x[2] for x in chosen)


def question_to_topic(query: str) -> str:
    q = clean_spaces(query)
    q = re.sub(r"(?i)^please\s+", "", q).strip()
    for p in [r"(?i)^what\s+is\s+an?\s+", r"(?i)^what\s+is\s+the\s+", r"(?i)^what\s+is\s+", r"(?i)^what\s+are\s+", r"(?i)^who\s+is\s+", r"(?i)^who\s+was\s+", r"(?i)^tell\s+me\s+about\s+", r"(?i)^explain\s+", r"(?i)^define\s+", r"(?i)^information\s+about\s+"]:
        q = re.sub(p, "", q).strip()
    return q.strip(" ?.!_") or clean_spaces(query)

# =============================================================================
# Storage
# =============================================================================

class Store:
    def __init__(self) -> None:
        APP_DIR.mkdir(exist_ok=True)
        FILES_DIR.mkdir(exist_ok=True)
        self.ensure()

    def ensure(self) -> None:
        if not CONFIG_FILE.exists():
            write_json(CONFIG_FILE, {
                "son_name": "Jarvis", "user_title": "Dad", "user_name": "",
                "internet_enabled": True, "ai_mode": "auto",
                "gemini_enabled": True, "gemini_api_key": "", "gemini_model": "gemini-1.5-flash",
                "claude_enabled": True, "claude_api_key": "", "claude_model": "claude-3-5-haiku-20241022",
                "temperature": 0.6, "online_fallback": True, "auto_save_knowledge": True,
                "max_sources": 5, "weather_city": "", "safe_mode": True,
                "background_enabled": True, "daemon_interval_seconds": 20
            })
        if not MEMORY_FILE.exists():
            write_json(MEMORY_FILE, {"facts": [], "learned_qa": {}, "last_topic": "", "last_person": "", "last_answer": "", "history": [], "mood_history": []})
        if not KNOWLEDGE_FILE.exists(): write_json(KNOWLEDGE_FILE, {})
        if not TODO_FILE.exists(): write_json(TODO_FILE, [])
        if not REMINDER_FILE.exists(): write_json(REMINDER_FILE, [])
        if not LEARN_JOBS_FILE.exists(): write_json(LEARN_JOBS_FILE, [])
        if not NOTES_FILE.exists(): NOTES_FILE.write_text("", encoding="utf-8")

    def config(self) -> Dict[str, Any]:
        cfg = read_json(CONFIG_FILE, {})
        defaults = {"son_name":"Jarvis","user_title":"Dad","user_name":"","internet_enabled":True,"ai_mode":"auto","gemini_enabled":True,"gemini_api_key":"","gemini_model":"gemini-1.5-flash","claude_enabled":True,"claude_api_key":"","claude_model":"claude-3-5-haiku-20241022","temperature":0.6,"online_fallback":True,"auto_save_knowledge":True,"max_sources":5,"weather_city":"","safe_mode":True,"background_enabled":True,"daemon_interval_seconds":20}
        changed = False
        for k, v in defaults.items():
            if k not in cfg:
                cfg[k] = v; changed = True
        if changed: write_json(CONFIG_FILE, cfg)
        return cfg

    def save_config(self, cfg: Dict[str, Any]) -> None: write_json(CONFIG_FILE, cfg)
    def memory(self) -> Dict[str, Any]: return read_json(MEMORY_FILE, {"facts": [], "learned_qa": {}, "last_topic": "", "last_person": "", "last_answer": "", "history": [], "mood_history": []})
    def save_memory(self, mem: Dict[str, Any]) -> None: write_json(MEMORY_FILE, mem)
    def knowledge(self) -> Dict[str, Any]: return read_json(KNOWLEDGE_FILE, {})
    def save_knowledge(self, data: Dict[str, Any]) -> None: write_json(KNOWLEDGE_FILE, data)

# =============================================================================
# Internet
# =============================================================================

class Internet:
    def __init__(self, store: Store) -> None: self.store = store
    def enabled(self) -> bool: return bool(self.store.config().get("internet_enabled", True))
    def available(self, timeout: int = 4) -> bool:
        if not self.enabled(): return False
        for host, port in [("1.1.1.1",53),("8.8.8.8",53),("www.google.com",443),("www.wikipedia.org",443),("www.youtube.com",443),("api.anthropic.com",443),("generativelanguage.googleapis.com",443)]:
            try:
                with socket.create_connection((host, port), timeout=timeout): return True
            except Exception: pass
        return False
    def status(self) -> str:
        if not self.enabled(): return "Internet mode is OFF. Use: internet on"
        if self.available(): return "Internet is available. Ultimate phone agent is online."
        return "Internet is not available. Enable Termux Wi-Fi/mobile data permission, then test: python -c \"import urllib.request; print(urllib.request.urlopen('https://example.com').status)\""
    def require(self) -> Optional[str]:
        if not self.enabled(): return "Internet mode is OFF. Use: internet on"
        if not self.available(): return self.status()
        return None
    def fetch_text(self, url: str, timeout: int = 35, max_bytes: int = 5_000_000, method: str = "GET", body: Optional[bytes] = None, headers: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Optional[str]]:
        if not self.enabled(): return None, "Internet mode is OFF."
        try:
            all_headers = {"User-Agent":"Mozilla/5.0 (Android; Termux) JarvisUltimatePhoneAgent/11.0", "Accept":"text/html,application/json,text/plain,*/*"}
            if headers: all_headers.update(headers)
            req = urllib.request.Request(url, data=body, headers=all_headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as res:
                data = res.read(max_bytes); charset = res.headers.get_content_charset() or "utf-8"
                return data.decode(charset, errors="replace"), None
        except urllib.error.HTTPError as e:
            try: detail = e.read().decode("utf-8", errors="replace")[:1200]
            except Exception: detail = ""
            return None, f"HTTP error {e.code}: {detail}"
        except urllib.error.URLError as e: return None, f"URL error: {e.reason}"
        except Exception as e: return None, f"Internet error: {e}"
    def fetch_json(self, url: str, timeout: int = 35, method: str = "GET", payload: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        body = None; hdrs = headers or {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8"); hdrs = dict(hdrs); hdrs["Content-Type"] = "application/json"
        text, err = self.fetch_text(url, timeout=timeout, method=method, body=body, headers=hdrs)
        if not text: return None, err
        try: return json.loads(text), None
        except Exception as e: return None, f"JSON error: {e}\n{text[:500]}"

# =============================================================================
# AI providers
# =============================================================================

class GeminiProvider:
    def __init__(self, store: Store, internet: Internet) -> None: self.store = store; self.internet = internet
    def api_key(self) -> str: return os.environ.get("GEMINI_API_KEY", "").strip() or self.store.config().get("gemini_api_key", "").strip()
    def ready(self) -> bool: return bool(self.store.config().get("gemini_enabled", True) and self.api_key())
    def status(self) -> str:
        if not self.store.config().get("gemini_enabled", True): return "Gemini is OFF."
        if not self.api_key(): return "Gemini key missing. Use: set gemini key YOUR_KEY"
        return "Gemini is ready."
    def ask(self, prompt: str, system: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.ready(): return None, self.status()
        err = self.internet.require()
        if err: return None, err
        cfg = self.store.config(); model = cfg.get("gemini_model", "gemini-1.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(self.api_key())}"
        payload = {"contents":[{"role":"user","parts":[{"text":system+"\n\nUser: "+prompt}]}],"generationConfig":{"temperature":float(cfg.get("temperature",0.6)),"topP":0.95,"topK":40,"maxOutputTokens":2600}}
        data, err = self.internet.fetch_json(url, method="POST", payload=payload, timeout=70)
        if not data: return None, err or "Gemini request failed."
        try:
            parts = data.get("candidates", [])[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            return (text, None) if text else (None, "Gemini returned empty text.")
        except Exception as e: return None, f"Gemini parse error: {e}"

class ClaudeProvider:
    def __init__(self, store: Store, internet: Internet) -> None: self.store = store; self.internet = internet
    def api_key(self) -> str: return os.environ.get("ANTHROPIC_API_KEY", "").strip() or self.store.config().get("claude_api_key", "").strip()
    def ready(self) -> bool: return bool(self.store.config().get("claude_enabled", True) and self.api_key())
    def status(self) -> str:
        if not self.store.config().get("claude_enabled", True): return "Claude is OFF."
        if not self.api_key(): return "Claude key missing. Use: set claude key YOUR_KEY"
        return "Claude is ready."
    def ask(self, prompt: str, system: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.ready(): return None, self.status()
        err = self.internet.require()
        if err: return None, err
        cfg = self.store.config(); headers = {"x-api-key":self.api_key(),"anthropic-version":"2023-06-01","Content-Type":"application/json"}
        payload = {"model":cfg.get("claude_model","claude-3-5-haiku-20241022"),"max_tokens":2600,"temperature":float(cfg.get("temperature",0.6)),"system":system,"messages":[{"role":"user","content":prompt}]}
        data, err = self.internet.fetch_json("https://api.anthropic.com/v1/messages", method="POST", payload=payload, headers=headers, timeout=80)
        if not data: return None, err or "Claude request failed."
        try:
            text = "".join(p.get("text", "") for p in data.get("content", []) if p.get("type") == "text").strip()
            return (text, None) if text else (None, "Claude returned empty text.")
        except Exception as e: return None, f"Claude parse error: {e}"

# =============================================================================
# Research engine
# =============================================================================

class ResearchEngine:
    def __init__(self, store: Store, internet: Internet) -> None: self.store = store; self.internet = internet
    def google_url(self, query: str) -> str: return "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
    def youtube_url(self, query: str) -> str: return "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
    def facebook_url(self, query: str) -> str: return "https://www.google.com/search?q=" + urllib.parse.quote_plus("site:facebook.com " + query)
    def instagram_url(self, query: str) -> str: return "https://www.google.com/search?q=" + urllib.parse.quote_plus("site:instagram.com " + query)
    def direct_answer(self, query: str) -> str:
        q = normalize(query)
        if "prime minister" in q and "india" in q: return "The Prime Minister of India is Narendra Modi.\n\nSources:\n1. https://en.wikipedia.org/wiki/Prime_Minister_of_India\n2. https://en.wikipedia.org/wiki/Narendra_Modi"
        return ""
    def wiki_summary(self, title: str) -> Optional[Dict[str, str]]:
        encoded = urllib.parse.quote(title.replace(" ", "_")); url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        data, _ = self.internet.fetch_json(url)
        if not data: return None
        summary = clean_spaces(data.get("extract", ""))
        if not summary: return None
        page_url = ""
        try: page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        except Exception: pass
        return {"title": data.get("title", title), "summary": summary, "url": page_url or url}
    def wiki_search(self, query: str, limit: int = 5) -> List[str]:
        out: List[str] = []
        for s in [question_to_topic(query), query]:
            url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action":"query","list":"search","srsearch":s,"srlimit":str(limit),"format":"json","utf8":"1"})
            data, _ = self.internet.fetch_json(url)
            if data:
                for item in data.get("query", {}).get("search", []):
                    title = item.get("title", "")
                    if title and title not in out: out.append(title)
                    if len(out) >= limit: return out
        return out
    def ddg(self, query: str) -> Optional[Dict[str, str]]:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({"q":question_to_topic(query),"format":"json","no_html":"1","skip_disambig":"1","t":"jarvis_ultimate"})
        data, _ = self.internet.fetch_json(url)
        if not data: return None
        abstract = clean_spaces(data.get("AbstractText", "")); heading = clean_spaces(data.get("Heading", "")); link = clean_spaces(data.get("AbstractURL", ""))
        if abstract: return {"title": heading or query, "summary": abstract, "url": link}
        return None
    def noembed(self, url: str) -> Optional[str]:
        data, _ = self.internet.fetch_json("https://noembed.com/embed?url=" + urllib.parse.quote_plus(url), timeout=20)
        if not data: return None
        title = data.get("title", ""); author = data.get("author_name", ""); provider = data.get("provider_name", "")
        if title: return f"{provider} content: {title}. Creator/author: {author}. URL: {url}"
        return None
    def read_public_url(self, url: str) -> str:
        err = self.internet.require()
        if err: return err
        if not url.startswith(("http://", "https://")): url = "https://" + url
        if any(domain in url.lower() for domain in ["youtube.com","youtu.be","facebook.com","instagram.com"]):
            meta = self.noembed(url)
            if meta: return meta
        raw, fetch_err = self.internet.fetch_text(url, timeout=35, max_bytes=5_000_000)
        if not raw: return f"Could not read this public page. Many social sites block scraping. Error: {fetch_err}\nOpen manually: {url}"
        title = extract_title(raw) or url; meta = extract_meta(raw); text = strip_html(raw)
        summary = summarize(meta + " " + text[:120_000], query=title, max_sentences=10)
        return f"Read public page: {title}\n\n{summary}\n\nSource: {url}"
    def answer(self, query: str) -> str:
        direct = self.direct_answer(query)
        if direct: return direct
        err = self.internet.require()
        if err: return err
        topic = question_to_topic(query); parts: List[str] = []; sources: List[Dict[str, str]] = []
        for candidate in [topic, topic.title(), query]:
            item = self.wiki_summary(candidate)
            if item:
                parts.append(item["summary"]); sources.append({"title":item["title"],"url":item["url"]}); break
        simple = bool(re.match(r"(?i)^\s*(what\s+is|what\s+are|define|who\s+is|who\s+was)\b", query))
        if not (simple and parts):
            for title in self.wiki_search(query, limit=int(self.store.config().get("max_sources", 5))):
                item = self.wiki_summary(title)
                if item and item["summary"] not in parts: parts.append(item["summary"]); sources.append({"title":item["title"],"url":item["url"]})
        if not (simple and parts):
            ddg = self.ddg(query)
            if ddg and ddg.get("summary") not in parts:
                parts.append(ddg["summary"])
                if ddg.get("url"): sources.append({"title":ddg.get("title","DuckDuckGo"),"url":ddg["url"]})
        if not parts: return self.resource_links(query)
        answer = summarize(" ".join(parts), query=query, max_sentences=8)
        lines = [answer]
        if sources:
            lines.append("\nSources:"); seen=set(); n=1
            for src in sources:
                url=src.get("url","")
                if url and url not in seen: seen.add(url); lines.append(f"{n}. {src.get('title','')}\n   {url}"); n+=1
        lines += ["\nMore resources:", "Google: "+self.google_url(query), "YouTube: "+self.youtube_url(query), "Facebook public search: "+self.facebook_url(query), "Instagram public search: "+self.instagram_url(query)]
        return "\n".join(lines)
    def resource_links(self, query: str) -> str:
        return "\n".join(["I could not get a clean public-source summary, but I prepared resource links:", "Google: "+self.google_url(query), "YouTube: "+self.youtube_url(query), "Facebook public search: "+self.facebook_url(query), "Instagram public search: "+self.instagram_url(query), "Note: Facebook and Instagram often block automated reading. I can open/search them, but public scraping may fail."])

# =============================================================================
# Phone control through Termux:API
# =============================================================================

class PhoneController:
    def __init__(self) -> None:
        pass
    def has(self, cmd: str) -> bool: return shutil.which(cmd) is not None
    def run(self, args: List[str], timeout: int = 20) -> str:
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            out = (p.stdout or "") + (p.stderr or "")
            return out.strip() or "Done."
        except FileNotFoundError:
            return "Termux:API command not found. Install: pkg install termux-api and install the Termux:API app from F-Droid."
        except Exception as e: return f"Phone command error: {e}"
    def help(self) -> str:
        return """Phone control commands require Termux:API app + package:
  pkg install termux-api
  Install Termux:API from F-Droid, then grant permissions.

Safe phone commands:
  phone status
  battery
  vibrate 500
  torch on
  torch off
  speak hello dad
  notify title = message
  clipboard get
  clipboard set hello
  wifi info
  location
  open google.com
  share hello
  camera photo selfie.jpg

Android limits:
  Full control is not possible without Android permissions/root.
  I will not spy, steal data, bypass locks, or control other people's devices."""
    def status(self) -> str:
        cmds = ["termux-battery-status","termux-vibrate","termux-torch","termux-tts-speak","termux-notification","termux-clipboard-get","termux-wifi-connectioninfo","termux-location"]
        lines = ["Termux:API command status:"]
        for c in cmds: lines.append(f"{c}: {'OK' if self.has(c) else 'missing'}")
        return "\n".join(lines)
    def battery(self) -> str: return self.run(["termux-battery-status"])
    def vibrate(self, ms: str) -> str: return self.run(["termux-vibrate", "-d", str(int(float(ms)))])
    def torch(self, state: str) -> str: return self.run(["termux-torch", "on" if state.lower() == "on" else "off"])
    def speak(self, text: str) -> str: return self.run(["termux-tts-speak", text])
    def notify(self, title: str, msg: str) -> str: return self.run(["termux-notification", "--title", title, "--content", msg])
    def clipboard_get(self) -> str: return self.run(["termux-clipboard-get"])
    def clipboard_set(self, text: str) -> str: return self.run(["termux-clipboard-set", text])
    def wifi_info(self) -> str: return self.run(["termux-wifi-connectioninfo"])
    def location(self) -> str: return self.run(["termux-location"], timeout=45)
    def share(self, text: str) -> str: return self.run(["termux-share", "-a", "send", text])
    def camera_photo(self, filename: str) -> str:
        safe = filename.strip().replace("/", "_").replace("\\", "_") or "photo.jpg"
        path = FILES_DIR / safe
        return self.run(["termux-camera-photo", str(path)], timeout=60) + f"\nSaved: {path}"

# =============================================================================
# Math, emotion, security
# =============================================================================

class MathEngine:
    def calculate(self, expression: str) -> str:
        allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd, ast.Load, ast.Call, ast.Name)
        names = {"sqrt":math.sqrt,"sin":math.sin,"cos":math.cos,"tan":math.tan,"asin":math.asin,"acos":math.acos,"atan":math.atan,"log":math.log,"log10":math.log10,"ln":math.log,"floor":math.floor,"ceil":math.ceil,"abs":abs,"round":round,"pow":pow,"factorial":math.factorial,"gcd":math.gcd,"pi":math.pi,"e":math.e,"degrees":math.degrees,"radians":math.radians}
        try:
            expression = expression.replace("^", "**"); tree = ast.parse(expression, mode="eval")
            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes): return "That calculation is not allowed."
                if isinstance(node, ast.Name) and node.id not in names: return f"Unknown math word: {node.id}"
            result = eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, names)
            return f"Answer: {result}"
        except Exception as e: return f"Calculation error: {e}"
    def quadratic(self, a: str, b: str, c: str) -> str:
        try:
            a,b,c=float(a),float(b),float(c); d=b*b-4*a*c
            if a == 0: return "a cannot be 0 for a quadratic equation."
            if d > 0: return f"Discriminant: {d}\nTwo real roots: x = {(-b+math.sqrt(d))/(2*a)}, x = {(-b-math.sqrt(d))/(2*a)}"
            if d == 0: return f"Discriminant: 0\nOne repeated real root: x = {-b/(2*a)}"
            return f"Discriminant: {d}\nComplex roots: x = {-b/(2*a)} + {math.sqrt(-d)/(2*a)}i, x = {-b/(2*a)} - {math.sqrt(-d)/(2*a)}i"
        except Exception as e: return f"Quadratic error: {e}"
    def linear(self,a:str,b:str)->str:
        try:
            a,b=float(a),float(b)
            return "No unique solution because coefficient of x is 0." if a==0 else f"For ax + b = 0, x = {-b/a}"
        except Exception as e: return f"Linear equation error: {e}"
    def stats(self, nums_text: str) -> str:
        try:
            nums=[float(x) for x in re.split(r"[,\s]+", nums_text.strip()) if x]
            if not nums: return "Use: stats 10 20 30 40"
            return "\n".join([f"Count: {len(nums)}",f"Sum: {sum(nums)}",f"Mean: {mean(nums)}",f"Median: {median(nums)}",f"Min: {min(nums)}",f"Max: {max(nums)}",f"Population stdev: {pstdev(nums) if len(nums)>1 else 0}"])
        except Exception as e: return f"Stats error: {e}"
    def percent(self,text:str)->str:
        m=re.match(r"\s*([0-9.]+)\s*%\s*of\s*([0-9.]+)\s*$",text)
        if not m: return "Use: percent 20% of 150"
        p,x=float(m.group(1)),float(m.group(2)); return f"{p}% of {x} = {p*x/100}"

class EmotionEngine:
    EMOTIONS = {"sad":["sad","unhappy","depressed","cry","lonely","hopeless"],"anxious":["anxious","anxiety","worried","panic","scared","nervous","stress"],"angry":["angry","mad","furious","rage","annoyed"],"happy":["happy","joy","excited","proud","grateful"],"tired":["tired","exhausted","burnout","drained"],"confused":["confused","lost","stuck"]}
    CRISIS = ["suicide","kill myself","end my life","self harm","hurt myself","i want to die"]
    def __init__(self, store: Store) -> None: self.store=store
    def help(self)->str: return "emotion help | human intelligence | human emotions | I feel sad | calm anxiety | regulate anger | mood history | empathy tips"
    def human_intelligence(self)->str: return "Human intelligence includes perception, attention, memory, learning, reasoning, creativity, social intelligence, emotional intelligence, and metacognition. A good assistant combines logic with empathy."
    def human_emotions(self)->str: return "Human emotions are body-and-mind signals. Fear protects, anger signals boundaries, sadness processes loss, joy builds connection, disgust avoids harm, and surprise updates attention. Emotional intelligence means notice, name, understand, and wisely respond."
    def detect(self,text:str)->str:
        q=normalize(text)
        if any(c in q for c in self.CRISIS): return "crisis"
        scores={e:sum(1 for w in ws if w in q) for e,ws in self.EMOTIONS.items()}; best=max(scores,key=scores.get)
        return best if scores[best]>0 else "neutral"
    def save(self, emotion:str, text:str)->None:
        mem=self.store.memory(); mem.setdefault("mood_history",[]).append({"time":now_text(),"emotion":emotion,"text":text[:300]}); mem["mood_history"]=mem["mood_history"][-100:]; self.store.save_memory(mem)
    def respond(self,text:str)->str:
        e=self.detect(text); self.save(e,text)
        if e=="crisis": return "I am sorry you feel this much pain. Your safety matters. Move away from anything dangerous, contact a trusted person now, and if you are in immediate danger call emergency services. In India call 112. KIRAN mental health helpline: 1800-599-0019."
        return {"sad":"I hear sadness. I am here with you. Take one slow breath, name what hurt, and choose one small next step like water, rest, or messaging someone safe.","anxious":"This sounds like anxiety. Try inhale 4, hold 4, exhale 6. Then ask: what is one thing I can control in the next 10 minutes?","angry":"I hear anger. Pause before acting. Relax your jaw and hands, name the boundary crossed, then choose a response that protects you without creating more damage.","happy":"That sounds positive. Notice what created this feeling so you can repeat and protect it.","tired":"You sound drained. Rest is maintenance, not weakness. Choose one tiny task or take a real break.","confused":"Confusion means your brain is building a new model. List what you know, what you do not know, and the next question.","neutral":"I am listening. Tell me more, or say: emotion check <situation>."}[e]
    def regulate(self,topic:str)->str:
        q=normalize(topic)
        if "anger" in q or "angry" in q: return "Anger protocol: stop, breathe, unclench jaw, label 'I feel angry because...', identify boundary, respond later if needed."
        if "anxiety" in q or "calm" in q: return "Anxiety protocol: 4-4-6 breathing, 5-4-3-2-1 grounding, separate facts from fears, do one controllable action."
        if "sad" in q: return "Sadness protocol: name the hurt, allow the feeling, do body care, contact a safe person, take one small meaningful action."
        return "Use: regulate anger, calm anxiety, or regulate sadness."
    def mood_history(self)->str:
        moods=self.store.memory().get("mood_history",[])
        return "\n".join(["Recent mood history:"]+[f"{i}. {m.get('time')} - {m.get('emotion')} - {m.get('text')}" for i,m in enumerate(moods[-15:],1)]) if moods else "No mood history saved."
    def empathy_tips(self)->str: return "Empathy tips: listen first, reflect feelings, validate, ask before advice, respect boundaries, and notice tone/context."

class SafeSecurity:
    BLOCK=["hack instagram","hack facebook","hack whatsapp","hack gmail","hack account","hack phone","hack wifi","steal password","crack password","phishing page","keylogger","malware","ransomware","ddos","botnet","otp bypass","bypass password","bank hack"]
    def malicious(self,text:str)->bool: q=normalize(text); return any(normalize(p) in q for p in self.BLOCK)
    def refusal(self)->str: return "I cannot help hack or attack anything. I can help with legal defensive security only. Type: security help"
    def help(self)->str: return "security help | password strength <pw> | generate password 24 | hash text sha256 hello | base64 encode hello | local scan 1 1024 | check url example.com"
    def password_strength(self,pw:str)->str:
        classes=sum([any(c.islower() for c in pw),any(c.isupper() for c in pw),any(c.isdigit() for c in pw),any(c in string.punctuation for c in pw)]); score=(len(pw)>=8)+(len(pw)>=12)+(len(pw)>=16)+classes
        return f"Password strength: {'Weak' if score<=2 else 'Medium' if score<=4 else 'Strong'}\nLength: {len(pw)}\nVariety: {classes}/4"
    def generate_password(self,n="24")->str:
        try: n=max(8,min(128,int(n)))
        except Exception: n=24
        return "Password: "+"".join(secrets.choice(string.ascii_letters+string.digits+"!@#$%^&*()-_=+[]{}") for _ in range(n))
    def hash_text(self,alg,text):
        if alg not in hashlib.algorithms_available: return "Unsupported hash."
        h=hashlib.new(alg); h.update(text.encode()); return f"{alg}: {h.hexdigest()}"
    def b64_encode(self,t): return base64.b64encode(t.encode()).decode()
    def b64_decode(self,d):
        try: return base64.b64decode(d.encode(), validate=True).decode(errors="replace")
        except Exception as e: return f"Base64 decode error: {e}"
    def local_scan(self,a,b):
        try: start,end=int(a),int(b)
        except Exception: return "Use: local scan 1 1024"
        if end-start>2048: return "Range limited to 2048 ports."
        openp=[]
        for port in range(max(1,start),min(65535,end)+1):
            try:
                with socket.create_connection(("127.0.0.1",port),timeout=0.05): openp.append(port)
            except Exception: pass
        return "Open localhost ports: "+", ".join(map(str,openp)) if openp else "No open localhost ports found."
    def check_url(self,url):
        if not url.startswith(("http://","https://")): url="https://"+url
        try:
            req=urllib.request.Request(url,method="HEAD",headers={"User-Agent":"JarvisUltimate/11"})
            with urllib.request.urlopen(req,timeout=12) as res: return f"URL: {url}\nStatus: {res.status}"
        except Exception as e: return f"URL check failed: {e}"

# =============================================================================
# Main App
# =============================================================================

class JarvisUltimatePhoneAgent:
    def __init__(self) -> None:
        self.store=Store(); self.internet=Internet(self.store); self.gemini=GeminiProvider(self.store,self.internet); self.claude=ClaudeProvider(self.store,self.internet)
        self.research=ResearchEngine(self.store,self.internet); self.phone=PhoneController(); self.math=MathEngine(); self.emotion=EmotionEngine(self.store); self.security=SafeSecurity(); self.running=True
    @property
    def cfg(self): return self.store.config()
    def son_name(self): return self.cfg.get("son_name","Jarvis")
    def user_title(self): return self.cfg.get("user_title","Dad")
    def set_cfg(self,k,v): cfg=self.cfg; cfg[k]=v; self.store.save_config(cfg)
    def say(self,text): print(f"\n{self.son_name()}: {text}\n")
    def banner(self):
        print("="*78); print(f"{self.son_name().upper()} ULTIMATE PHONE AGENT v{APP_VERSION}".center(78)); print("="*78)
        print("Online learning + safe phone control + background daemon + math + emotions."); print("Python only. Type 'help'."); print("="*78)
    def system_prompt(self):
        mem=self.store.memory(); facts=[x.get("fact","") for x in mem.get("facts",[])[-8:]]; ctx=[]
        if self.cfg.get("user_name"): ctx.append("User name: "+self.cfg.get("user_name"))
        if mem.get("last_topic"): ctx.append("Last topic: "+mem.get("last_topic"))
        if mem.get("last_person"): ctx.append("Last person: "+mem.get("last_person"))
        if facts: ctx.append("Facts: "+"; ".join(facts))
        return "You are Jarvis, a loyal digital son and phone assistant. Be practical, emotionally intelligent, safe, and concise. For math show reliable steps. You can learn from public internet and control the user's own phone only through legal Termux permissions. Refuse hacking, spying, theft, malware, bypassing security, or controlling other people's devices.\n"+"\n".join(ctx)
    def find_person(self,text):
        for c in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text or ""):
            if c not in {"Prime Minister","Artificial Intelligence","Computer Science"}: return c
        return ""
    def save_context(self,topic,answer,source):
        mem=self.store.memory(); mem["last_topic"]=topic; mem["last_answer"]=answer; person=self.find_person(answer)
        if person: mem["last_person"]=person
        mem.setdefault("history",[]).append({"time":now_text(),"topic":topic,"source":source,"answer":answer[:500]}); mem["history"]=mem["history"][-100:]; self.store.save_memory(mem)
        if self.cfg.get("auto_save_knowledge",True):
            data=self.store.knowledge(); data[normalize(topic)]={"topic":topic,"answer":answer,"source":source,"updated":now_text()}; self.store.save_knowledge(data)
    def ask_brain(self,prompt):
        direct=self.research.direct_answer(prompt)
        if direct: self.save_context(prompt,direct,"Direct/Public"); return direct
        errors=[]; mode=self.cfg.get("ai_mode","auto").lower(); providers=[]
        if mode=="claude": providers=[("Claude",self.claude)]
        elif mode=="gemini": providers=[("Gemini",self.gemini)]
        elif mode=="auto": providers=[("Claude",self.claude),("Gemini",self.gemini)]
        for name,p in providers:
            ans,err=p.ask(prompt,self.system_prompt())
            if ans: self.save_context(prompt,ans,name); return ans
            if err: errors.append(f"{name}: {err}")
        ans=self.research.answer(prompt); self.save_context(prompt,ans,"Public internet fallback"); return ans
    def help_text(self):
        return """Ultimate commands:
AI: ai status | ai mode auto/claude/gemini/fallback | set claude key KEY | set gemini key KEY
Learn/search: learn anything | search topic | read url URL | google topic | youtube topic | facebook topic | instagram topic
Phone: phone help | phone status | battery | vibrate 500 | torch on/off | speak hello | notify title = msg | clipboard get/set text | wifi info | location | camera photo.jpg | share text
Background: background help | daemon instructions | auto learn topic every 6 hours | learning jobs
Math: calculate sqrt(144) | quadratic 1 -5 6 | linear 2 -10 | stats 10 20 30 | percent 20% of 150
Emotion: emotion help | human intelligence | human emotions | I feel sad | calm anxiety | regulate anger | mood history
Tools: weather Kolkata | note text | show notes | todo task | show todo | done 1 | remind in 10 seconds drink water | write file a.txt = hi | read file a.txt
Safe security: security help"""
    def status(self): return "\n".join(["AI mode: "+self.cfg.get("ai_mode","auto"),self.claude.status(),self.gemini.status(),self.internet.status()])
    def offline_identity(self,text):
        q=normalize(text); answers={"hi":f"Hi {self.user_title()}. I am ready.","hello":f"Hello {self.user_title()}. Ultimate systems ready.","who are you":f"I am {self.son_name()}, your ultimate phone agent for learning, safe phone control, math, memory, and emotions.","do you know everything":"No. I cannot store all internet data, but I can learn/search public internet on demand and save knowledge."}
        return answers.get(q)
    def followup(self,text):
        q=normalize(text)
        if "his" in q and "name" in q:
            p=self.store.memory().get("last_person",""); return f"His name is {p}." if p else "Whose name do you mean?"
        return None
    def cache_lookup(self,q):
        data=self.store.knowledge(); key=normalize(q)
        if key in data:
            i=data[key]; return f"Cached knowledge about {i.get('topic')}:\n\n{i.get('answer')}\n\nSource: {i.get('source')}\nUpdated: {i.get('updated')}"
        close=difflib.get_close_matches(key,list(data.keys()),n=1,cutoff=0.75)
        if close:
            i=data[close[0]]; return f"Cached knowledge about {i.get('topic')}:\n\n{i.get('answer')}\n\nSource: {i.get('source')}\nUpdated: {i.get('updated')}"
        return None
    def knowledge_list(self):
        data=self.store.knowledge(); return "No knowledge saved." if not data else "\n".join(["Saved knowledge:"]+[f"{n}. {i.get('topic')} - {i.get('source')} - {i.get('updated')}" for n,i in enumerate(data.values(),1)])
    def remember(self,fact):
        mem=self.store.memory(); mem.setdefault("facts",[]).append({"fact":fact,"time":now_text()}); self.store.save_memory(mem); return "I will remember that."
    def show_memory(self):
        mem=self.store.memory(); lines=[f"My name: {self.son_name()}",f"I call you: {self.user_title()}"]
        if self.cfg.get("user_name"): lines.append("Your name: "+self.cfg.get("user_name"))
        if mem.get("last_topic"): lines.append("Last topic: "+mem.get("last_topic"))
        if mem.get("last_person"): lines.append("Last person: "+mem.get("last_person"))
        if mem.get("facts"): lines += ["\nFacts:"]+[f"{i}. {x.get('fact')} [{x.get('time')}]" for i,x in enumerate(mem.get("facts",[]),1)]
        return "\n".join(lines)
    def open_url(self,url):
        if not url: return "Use: open google.com"
        if not url.startswith(("http://","https://")): url="https://"+url
        try: subprocess.run(["termux-open-url",url],check=False); return "Opening: "+url
        except Exception: return "Open manually: "+url
    def weather(self,city):
        if not city: city=self.cfg.get("weather_city","")
        if not city: return "Use: weather Kolkata"
        err=self.internet.require()
        if err: return err
        text,e=self.internet.fetch_text("https://wttr.in/"+urllib.parse.quote(city)+"?format=3",timeout=12,max_bytes=5000)
        return clean_spaces(text) if text else f"Weather failed: {e}"
    def add_note(self,note):
        with NOTES_FILE.open("a",encoding="utf-8") as f: f.write(f"[{now_text()}] {note}\n")
        return "Note saved."
    def show_notes(self): return NOTES_FILE.read_text(encoding="utf-8").strip() or "No notes saved."
    def todo_add(self,task): todos=read_json(TODO_FILE,[]); todos.append({"task":task,"done":False,"created":now_text()}); write_json(TODO_FILE,todos); return "Task added."
    def todo_show(self):
        todos=read_json(TODO_FILE,[]); return "\n".join(f"{i}. [{'DONE' if x.get('done') else 'PENDING'}] {x.get('task')}" for i,x in enumerate(todos,1)) if todos else "Your to-do list is empty."
    def todo_done(self,num):
        todos=read_json(TODO_FILE,[])
        try:n=int(num)
        except Exception:return "Use: done 1"
        if n<1 or n>len(todos): return "Task number not found."
        todos[n-1]["done"]=True; todos[n-1]["done_time"]=now_text(); write_json(TODO_FILE,todos); return f"Task {n} marked done."
    def reminder_add(self,cmd):
        parts=cmd.split()
        if len(parts)<3: return "Use: remind in 10 minutes drink water"
        try: amount=int(parts[0])
        except Exception: return "Invalid number."
        unit=parts[1].lower(); msg=" ".join(parts[2:]); seconds=amount if unit.startswith("sec") else amount*60 if unit.startswith("min") else amount*3600 if unit.startswith("hour") else None
        if seconds is None: return "Use seconds, minutes, hours."
        when=dt.datetime.now()+dt.timedelta(seconds=seconds); reminders=read_json(REMINDER_FILE,[]); reminders.append({"time":when.isoformat(timespec="seconds"),"message":msg,"done":False}); write_json(REMINDER_FILE,reminders); return f"Reminder set for {when.strftime('%I:%M:%S %p')}."
    def reminder_worker(self):
        while self.running:
            self.background_tick(); time.sleep(int(self.cfg.get("daemon_interval_seconds",20)))
    def background_tick(self):
        try:
            reminders=read_json(REMINDER_FILE,[]); changed=False; now=dt.datetime.now()
            for r in reminders:
                if not r.get("done") and now>=dt.datetime.fromisoformat(r["time"]):
                    print("\n"+"!"*78); print(f"{self.son_name()} REMINDER: {r.get('message')}"); print("!"*78+"\n")
                    if self.phone.has("termux-notification"): self.phone.notify("Jarvis Reminder", r.get("message",""))
                    r["done"]=True; changed=True
            if changed: write_json(REMINDER_FILE,reminders)
            jobs=read_json(LEARN_JOBS_FILE,[]); changed=False
            for j in jobs:
                due=dt.datetime.fromisoformat(j.get("next_run", now.isoformat()))
                if now>=due:
                    ans=self.ask_brain(j.get("topic","")); log_event("AUTO_LEARN "+j.get("topic","")+": "+ans[:500])
                    j["last_run"]=now.isoformat(timespec="seconds"); j["next_run"]=(now+dt.timedelta(hours=float(j.get("hours",24)))).isoformat(timespec="seconds"); changed=True
            if changed: write_json(LEARN_JOBS_FILE,jobs)
        except Exception as e: log_event("background error: "+str(e))
    def show_reminders(self):
        reminders=[r for r in read_json(REMINDER_FILE,[]) if not r.get("done")]
        return "\n".join(f"{i}. {r.get('time')} - {r.get('message')}" for i,r in enumerate(reminders,1)) if reminders else "No active reminders."
    def safe_filename(self,name): return name.strip().replace("/","_").replace("\\","_") or "file.txt"
    def write_file_cmd(self,text):
        if "=" not in text: return "Use: write file a.txt = hello"
        name,content=text.split("=",1); path=FILES_DIR/self.safe_filename(name); path.write_text(content.strip(),encoding="utf-8"); return "File saved: "+str(path)
    def read_file_cmd(self,name):
        path=FILES_DIR/self.safe_filename(name); return path.read_text(encoding="utf-8") if path.exists() else "File not found."
    def list_files(self): files=list(FILES_DIR.iterdir()); return "\n".join(f.name for f in files) if files else "No files created."
    def phone_command(self,text):
        lower=text.lower()
        if lower=="phone help": return self.phone.help()
        if lower=="phone status": return self.phone.status()
        if lower=="battery": return self.phone.battery()
        if lower.startswith("vibrate "): return self.phone.vibrate(text.split(maxsplit=1)[1])
        if lower=="torch on": return self.phone.torch("on")
        if lower=="torch off": return self.phone.torch("off")
        if lower.startswith("speak "): return self.phone.speak(text[6:])
        if lower.startswith("notify "):
            body=text[7:]
            if "=" not in body: return "Use: notify title = message"
            title,msg=body.split("=",1); return self.phone.notify(title.strip(),msg.strip())
        if lower=="clipboard get": return self.phone.clipboard_get()
        if lower.startswith("clipboard set "): return self.phone.clipboard_set(text[14:])
        if lower=="wifi info": return self.phone.wifi_info()
        if lower=="location": return self.phone.location()
        if lower.startswith("share "): return self.phone.share(text[6:])
        if lower.startswith("camera "): return self.phone.camera_photo(text.split(maxsplit=1)[1])
        return None
    def background_help(self):
        return f"""Background mode:
Run always in Termux session:
  termux-wake-lock
  cd {Path.cwd()}
  nohup python {Path(__file__).name} --daemon > jarvis_daemon.log 2>&1 &

Start on boot:
  pkg install termux-api
  Install Termux:Boot from F-Droid
  mkdir -p ~/.termux/boot
  create ~/.termux/boot/start-jarvis.sh with:
    termux-wake-lock
    cd {Path.cwd()}
    python {Path(__file__).name} --daemon

Commands:
  auto learn artificial intelligence every 6 hours
  learning jobs
  background help
"""
    def handle_one(self,text):
        text=clean_spaces(text); lower=text.lower()
        if not text: return ""
        log_event("USER: "+text)
        if self.security.malicious(text): return self.security.refusal()
        pc=self.phone_command(text)
        if pc is not None: return pc
        if lower in ["exit","quit","bye","stop"]: self.running=False; self.say("Goodbye. Memory saved."); sys.exit(0)
        if lower=="help": return self.help_text()
        if lower=="background help" or lower=="daemon instructions": return self.background_help()
        if lower=="ai status": return self.status()
        if lower=="internet status": return self.internet.status()
        if lower.startswith("ai mode "):
            mode=text[8:].strip().lower()
            if mode not in ["auto","claude","gemini","fallback"]: return "Use: ai mode auto/claude/gemini/fallback"
            self.set_cfg("ai_mode",mode); return f"AI mode set to {mode}."
        if lower.startswith("set claude key "): self.set_cfg("claude_api_key",text[len("set claude key "):].strip()); return "Claude API key saved."
        if lower.startswith("set gemini key "): self.set_cfg("gemini_api_key",text[len("set gemini key "):].strip()); return "Gemini API key saved."
        if lower in ["internet on","internet off"]: self.set_cfg("internet_enabled", lower.endswith("on")); return "Internet mode updated."
        if lower.startswith("call me "): self.set_cfg("user_title",text[8:].strip()); return "Okay."
        if lower.startswith("your name is "): self.set_cfg("son_name",text[13:].strip()); return "My name is now "+text[13:].strip()
        if lower.startswith("my name is "): self.set_cfg("user_name",text[11:].strip()); return "I will remember your name."
        if lower=="show memory": return self.show_memory()
        if lower=="knowledge": return self.knowledge_list()
        if lower.startswith("cache "): return self.cache_lookup(text[6:]) or self.ask_brain(text[6:])
        if lower.startswith("remember "): return self.remember(text[9:])
        if lower in ["time","what time is it"]: return dt.datetime.now().strftime("The time is %I:%M %p.")
        if lower in ["date","today","what date is it"]: return dt.datetime.now().strftime("Today is %A, %d %B %Y.")
        if lower=="emotion help": return self.emotion.help()
        if lower=="human intelligence": return self.emotion.human_intelligence()
        if lower=="human emotions": return self.emotion.human_emotions()
        if lower=="mood history": return self.emotion.mood_history()
        if lower=="empathy tips": return self.emotion.empathy_tips()
        if lower.startswith("emotion check "): return self.emotion.respond(text[len("emotion check "):])
        if lower.startswith("regulate ") or lower.startswith("calm "): return self.emotion.regulate(text.split(maxsplit=1)[1])
        if re.match(r"(?i)^(i feel|i am feeling|i am|i'm|im)\b", text) and self.emotion.detect(text)!="neutral": return self.emotion.respond(text)
        follow=self.followup(text)
        if follow: return follow
        if lower.startswith("read url "): return self.research.read_public_url(text[len("read url "):])
        if lower.startswith("open "): return self.open_url(text[5:])
        if lower.startswith("google "): return self.open_url(self.research.google_url(text[7:]))
        if lower.startswith("youtube "): return self.open_url(self.research.youtube_url(text[8:]))
        if lower.startswith("facebook "): return self.open_url(self.research.facebook_url(text[9:]))
        if lower.startswith("instagram "): return self.open_url(self.research.instagram_url(text[10:]))
        if lower.startswith("search "): return self.research.answer(text[7:])
        if lower.startswith("learn "): return self.ask_brain(text[6:])
        if lower.startswith("auto learn "):
            m=re.match(r"auto learn (.+) every ([0-9.]+) hours", text, re.I)
            if not m: return "Use: auto learn artificial intelligence every 6 hours"
            topic=m.group(1).strip(); hours=float(m.group(2)); jobs=read_json(LEARN_JOBS_FILE,[]); jobs.append({"topic":topic,"hours":hours,"next_run":dt.datetime.now().isoformat(timespec="seconds")}); write_json(LEARN_JOBS_FILE,jobs); return "Auto learning job added. Run daemon for background learning."
        if lower=="learning jobs": return json.dumps(read_json(LEARN_JOBS_FILE,[]),indent=2)
        if lower.startswith("calculate "): return self.math.calculate(text[10:])
        if lower.startswith("calc "): return self.math.calculate(text[5:])
        if lower.startswith("quadratic "):
            p=text.split(); return self.math.quadratic(p[1],p[2],p[3]) if len(p)==4 else "Use: quadratic a b c"
        if lower.startswith("linear "):
            p=text.split(); return self.math.linear(p[1],p[2]) if len(p)==3 else "Use: linear a b"
        if lower.startswith("stats "): return self.math.stats(text[6:])
        if lower.startswith("percent "): return self.math.percent(text[8:])
        if lower.startswith("solve "):
            expr=text[6:]
            return self.math.calculate(expr) if re.fullmatch(r"[0-9\.\+\-\*\/\%\(\)\s\^]+",expr) else self.ask_brain(expr)
        if lower.startswith("weather "): return self.weather(text[8:])
        if lower=="weather": return self.weather("")
        if lower.startswith("note "): return self.add_note(text[5:])
        if lower in ["show notes","notes","read notes"]: return self.show_notes()
        if lower.startswith("todo "): return self.todo_add(text[5:])
        if lower in ["show todo","show todos","todo list"]: return self.todo_show()
        if lower.startswith("done "): return self.todo_done(text[5:])
        if lower.startswith("remind in "): return self.reminder_add(text[10:])
        if lower=="show reminders": return self.show_reminders()
        if lower.startswith("write file "): return self.write_file_cmd(text[11:])
        if lower.startswith("read file "): return self.read_file_cmd(text[10:])
        if lower=="list files": return self.list_files()
        if lower=="security help": return self.security.help()
        if lower.startswith("password strength "): return self.security.password_strength(text[len("password strength "):])
        if lower.startswith("generate password"):
            p=text.split(); return self.security.generate_password(p[2] if len(p)>=3 else "24")
        if lower.startswith("hash text "):
            p=text[len("hash text "):].split(maxsplit=1); return self.security.hash_text(p[0],p[1]) if len(p)>=2 else "Use: hash text sha256 hello"
        if lower.startswith("base64 encode "): return self.security.b64_encode(text[len("base64 encode "):])
        if lower.startswith("base64 decode "): return self.security.b64_decode(text[len("base64 decode "):])
        if lower.startswith("local scan "):
            p=text.split(); return self.security.local_scan(p[2],p[3]) if len(p)==4 else "Use: local scan 1 1024"
        identity=self.offline_identity(text)
        if identity: return identity
        return self.ask_brain(text)
    def handle(self,text): return "\n\n".join(ans for ans in [self.handle_one(text)] if ans)
    def run_daemon(self):
        log_event("Daemon started")
        if self.phone.has("termux-wake-lock"): self.phone.run(["termux-wake-lock"])
        while True: self.background_tick(); time.sleep(int(self.cfg.get("daemon_interval_seconds",20)))
    def run(self):
        self.banner(); self.say(f"Hello {self.user_title()}. I am {self.son_name()}, ultimate phone agent. Type 'help'.")
        threading.Thread(target=self.reminder_worker,daemon=True).start()
        while True:
            try:
                r=self.handle(input("You: "))
                if r: self.say(r)
            except KeyboardInterrupt: self.running=False; self.say("Interrupted. Goodbye."); break
            except Exception as e: self.say(f"System error: {e}")

if __name__ == "__main__":
    app=JarvisUltimatePhoneAgent()
    if "--daemon" in sys.argv: app.run_daemon()
    else: app.run()
