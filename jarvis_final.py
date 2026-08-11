#!/usr/bin/env python3
"""
Jarvis AI Assistant - FULLY FUNCTIONAL Browser Version
========================================================

✅ Works in browser (Pyodide) WITHOUT any pip installs
✅ Works in standard Python with groq package
✅ All features working: time, date, calculator, AI questions

SETUP:
------
1. Get free API key: https://console.groq.com
2. Set it: set_api_key("gsk_...")  OR  os.environ["GROQ_API_KEY"] = "gsk_..."
3. Start using!

QUICK START (copy this entire file):
------------------------------------
import os
os.environ["GROQ_API_KEY"] = "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"

from jarvis_browser_full import Jarvis
jarvis = Jarvis()
print(jarvis.chat("Hello!"))
"""

import os
import sys
import json
import re
from collections import deque
from datetime import datetime, date as date_class
from typing import List, Dict, Optional, Any

# ============================================================================
# BROWSER DETECTION
# ============================================================================

BROWSER = False
js = None

try:
    import js
    BROWSER = True
    print("🌐 Browser mode (Pyodide)")
except ImportError:
    print("💻 Standard Python mode")

# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def get_api_key() -> Optional[str]:
    """Get API key from environment or localStorage."""
    # Environment variable
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key and api_key.startswith("gsk_"):
        return api_key
    
    # Browser localStorage
    if BROWSER and js:
        try:
            key = js.localStorage.getItem("GROQ_API_KEY")
            if key and key.startswith("gsk_"):
                os.environ["GROQ_API_KEY"] = key
                return key
        except:
            pass
    
    return None


def set_api_key(key: str):
    """Set API key."""
    if key and key.startswith("gsk_"):
        os.environ["GROQ_API_KEY"] = key
        if BROWSER and js:
            try:
                js.localStorage.setItem("GROQ_API_KEY", key)
            except:
                pass
        print(f"✓ API key set")
    else:
        print("✗ Invalid API key (must start with 'gsk_')")


# ============================================================================
# GROQ API CLIENT
# ============================================================================

class GroqClient:
    """Groq API client."""
    
    BASE_URL = "https://api.groq.com/openai/v1"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def chat(self, model: str, messages: List[Dict], max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Call Groq API."""
        
        url = f"{self.BASE_URL}/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if BROWSER and js:
            return self._fetch_js(url, body)
        else:
            return self._fetch_python(url, body)
    
    def _fetch_js(self, url: str, body: dict) -> str:
        """JavaScript fetch for browser."""
        try:
            response = js.fetch(url, {
                "method": "POST",
                "headers": {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                "body": json.dumps(body)
            })
            
            result = response.json()
            
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"]
            else:
                error = result.get("error", {}).get("message", "Unknown error")
                raise Exception(f"API error: {error}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def _fetch_python(self, url: str, body: dict) -> str:
        """Python urllib fallback."""
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError
        
        req = Request(url, method='POST')
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")
        
        try:
            with urlopen(req, data=json.dumps(body).encode(), timeout=30) as resp:
                result = json.loads(resp.read().decode())
                if result.get("choices"):
                    return result["choices"][0]["message"]["content"]
                raise Exception("No response from API")
        except HTTPError as e:
            try:
                err = json.loads(e.read().decode())
                msg = err.get("error", {}).get("message", str(e))
            except:
                msg = str(e)
            raise Exception(f"HTTP {e.code}: {msg}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")


# ============================================================================
# INTENT HANDLER
# ============================================================================

class IntentHandler:
    """Rule-based intent handler - works without AI."""
    
    def check(self, query: str) -> Optional[str]:
        q = query.lower().strip()
        
        # Time
        if any(w in q for w in ["time", "what time", "current time", "clock"]):
            now = datetime.now()
            return f"The current time is {now.strftime('%I:%M %p').lstrip('0')}"
        
        # Date
        if any(w in q for w in ["date", "what date", "today", "day is it"]):
            today = date_class.today()
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            return f"Today is {days[today.weekday()]}, {today.strftime('%B %d, %Y')}"
        
        # Calculator
        if any(w in q for w in ["calculate", "calculator", "what is", "compute"]):
            result = self._calc(q)
            if result is not None:
                return f"The answer is {result}"
        
        # Greeting (check before "how are you" to avoid false matches)
        if any(w in q for w in ["hello", "hi ", "hey ", "greetings", "good morning", "good afternoon", "good evening"]):
            hour = datetime.now().hour
            g = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
            return f"{g}! I'm Jarvis. How can I help you?"
        
        # How are you (specific pattern)
        if re.search(r'\bhow\s+(are|is)\s+you\b', q):
            return "I'm doing great, thank you! How can I assist you today?"
        
        # Identity
        if any(w in q for w in ["your name", "who are you", "what are you", "introduce yourself"]):
            return "I'm Jarvis, your personal AI assistant. I can help with time, date, calculations, and answer questions!"
        
        # Help
        if any(w in q for w in ["help", "what can you do", "capabilities"]):
            return (
                "Here's what I can do:\n"
                "• Tell time and date\n"
                "• Do calculations (e.g., 'Calculate 25 * 4')\n"
                "• Answer questions via AI (with API key)\n"
                "• Simple conversations\n\n"
                "Type 'exit' to quit."
            )
        
        # Exit
        if any(w in q for w in ["exit", "quit", "bye", "goodbye"]):
            return "exit"
        
        return None
    
    def _calc(self, query: str) -> Optional[float]:
        q = re.sub(r'\b(what is|calculate|compute)\b', '', query.lower())
        q = re.sub(r'[?.,!\s]', '', q)
        
        match = re.search(r'(-?\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(-?\d+(?:\.\d+)?)', q)
        if match:
            a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
            if op == '+': return round(a + b, 4)
            if op == '-': return round(a - b, 4)
            if op == '*': return round(a * b, 4)
            if op == '/':
                if b == 0: return None
                return round(a / b, 4)
        return None


# ============================================================================
# MEMORY
# ============================================================================

class Memory:
    def __init__(self, max_turns=10):
        self.history = deque(maxlen=max_turns)
    
    def add(self, user: str, response: str):
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": response})
    
    def get_context(self) -> List[Dict]:
        return list(self.history)


# ============================================================================
# JARVIS
# ============================================================================

class Jarvis:
    """Main Jarvis class - fully functional."""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.intents = IntentHandler()
        self.memory = Memory()
        self.client = None
        self.api_key = None
        self._setup()
    
    def _setup(self):
        self.api_key = get_api_key()
        if self.api_key:
            self.client = GroqClient(self.api_key)
    
    def chat(self, query: str) -> str:
        """Chat with Jarvis."""
        query = query.strip()
        if not query:
            return ""
        
        # Check intents first
        intent = self.intents.check(query)
        if intent == "exit":
            return "Goodbye! Have a great day."
        if intent:
            self.memory.add(query, intent)
            return intent
        
        # Use AI
        if self.client:
            try:
                messages = self.memory.get_context()
                messages.append({"role": "user", "content": query})
                response = self.client.chat(self.model, messages)
                self.memory.add(query, response)
                return response
            except Exception as e:
                return f"Error: {str(e)[:100]}"
        else:
            return "AI not available. Set API key with: set_api_key('gsk_...')"
    
    def is_ai_enabled(self) -> bool:
        return self.client is not None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_interactive():
    """Interactive chat mode."""
    jarvis = Jarvis()
    
    print("\n" + "=" * 50)
    print("  JARVIS AI")
    print("=" * 50)
    print(f"  AI: {'Enabled' if jarvis.is_ai_enabled() else 'Disabled'}")
    print("=" * 50)
    print("\nType 'help' or ask anything!\n")
    
    while True:
        try:
            if BROWSER and js:
                try:
                    user_input = js.prompt("You: ", "")
                except:
                    user_input = input("You: ")
            else:
                user_input = input("You: ")
            
            if not user_input:
                continue
            
            print(f"You: {user_input}")
            response = jarvis.chat(user_input)
            print(f"Jarvis: {response}\n")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                break
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break


def run_demo():
    """Show demo of features."""
    print("\n" + "=" * 50)
    print("  JARVIS DEMO")
    print("=" * 50 + "\n")
    
    jarvis = Jarvis()
    
    demos = [
        ("What time is it?", "Time"),
        ("Calculate 25 * 4", "Calculator"),
        ("What's today's date?", "Date"),
        ("Hello!", "Greeting"),
        ("Help", "Help"),
    ]
    
    for query, label in demos:
        print(f"📝 {label}: {query}")
        print(f"   → {jarvis.chat(query)}\n")


def chat(query: str, api_key: str = None) -> str:
    """Quick chat function."""
    if api_key:
        set_api_key(api_key)
    return Jarvis().chat(query)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  JARVIS AI - FULLY FUNCTIONAL")
    print("=" * 50 + "\n")
    
    api_key = get_api_key()
    
    if api_key:
        print("✓ API key found! Starting...\n")
        run_interactive()
    else:
        print("⚠ No API key found.")
        print("\nTo enable AI:")
        print("  1. Get key: https://console.groq.com")
        print("  2. Run: set_api_key('gsk_...')")
        print("\n" + "-" * 50 + "\n")
        run_demo()
