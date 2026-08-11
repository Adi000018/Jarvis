#!/usr/bin/env python3
"""
Jarvis AI Assistant - Browser/Pyodide Compatible Version
=========================================================

Works in browser Python compilers (Pyodide, etc.) with no external dependencies.

Features:
- Rule-based intents (time, date, calculator) - works offline
- Groq AI integration (if groq package is available + API key)
- Simple text-based interaction
- No voice/microphone (not available in browser)

Usage:
    # Basic usage
    from jarvis_browser import Jarvis, jarvis_chat
    
    # With API key
    os.environ["GROQ_API_KEY"] = "gsk_..."
    jarvis = Jarvis()
    print(jarvis.chat("Hello!"))
    
    # Without API key (demo mode with rule-based intents only)
    jarvis = Jarvis()
    print(jarvis.chat("What time is it?"))
"""

import os
import sys
import json
import re
import time
from collections import deque
from datetime import datetime, date as date_class
from typing import List, Dict, Optional, Any

# ============================================================================
# DETECT BROWSER/PYODIDE ENVIRONMENT
# ============================================================================

def is_browser_environment():
    """Detect if running in browser (Pyodide, etc.)."""
    # Check for Pyodide
    if hasattr(sys, 'modules') and 'pyodide' in sys.modules:
        return True
    # Check version string
    if 'pyodide' in sys.version.lower() or 'browser' in sys.version.lower():
        return True
    # Check for js module
    try:
        import js
        return True
    except ImportError:
        pass
    return False

BROWSER = is_browser_environment()
print(f"🔍 Environment: {'Browser/Pyodide' if BROWSER else 'Standard Python'}")

# Try to import optional packages
try:
    import js
    HAS_JS = True
except ImportError:
    js = None
    HAS_JS = False

# Try groq package
try:
    from groq import Groq
    HAS_GROQ = True
    print("✓ Groq package available")
except ImportError:
    Groq = None
    HAS_GROQ = False
    print("✗ Groq package not available (will use demo mode for AI responses)")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model": "llama-3.3-70b-versatile",
    "system_prompt": "You are Jarvis, a helpful personal assistant. Be concise and friendly.",
    "max_history": 10,
}


# ============================================================================
# LOGGING
# ============================================================================

class Logger:
    def __init__(self, name="Jarvis"):
        self.name = name
    
    def info(self, msg):
        print(f"[{self.name}] {msg}")
    
    def error(self, msg):
        print(f"[{self.name}] ERROR: {msg}")


logger = Logger()


# ============================================================================
# GET API KEY (Multiple sources)
# ============================================================================

def get_api_key() -> Optional[str]:
    """Get Groq API key from various sources."""
    
    # 1. Environment variable
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key and api_key.startswith("gsk_"):
        return api_key
    
    # 2. Browser localStorage
    if HAS_JS:
        try:
            key = js.localStorage.getItem("GROQ_API_KEY")
            if key and key.startswith("gsk_"):
                os.environ["GROQ_API_KEY"] = key
                return key
        except:
            pass
    
    # 3. Prompt user (browser only)
    if HAS_JS:
        try:
            key = js.prompt("Enter Groq API Key (starts with gsk_):", "")
            if key and key.startswith("gsk_"):
                os.environ["GROQ_API_KEY"] = key
                try:
                    js.localStorage.setItem("GROQ_API_KEY", key)
                except:
                    pass
                return key
        except:
            pass
    
    return None


# ============================================================================
# RULE-BASED INTENTS (No dependencies - always works)
# ============================================================================

class IntentHandler:
    """Handle simple queries without AI."""
    
    def check(self, query: str) -> Optional[str]:
        q = query.lower().strip()
        
        # Time
        if any(w in q for w in ["time", "what time", "current time"]):
            now = datetime.now()
            return f"The current time is {now.strftime('%I:%M %p').lstrip('0')}"
        
        # Date
        if any(w in q for w in ["date", "what date", "today", "day is it"]):
            today = date_class.today()
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            return f"Today is {days[today.weekday()]}, {today.strftime('%B %d, %Y')}"
        
        # Calculator
        if any(w in q for w in ["calculate", "calculator", "what is", "compute"]):
            result = self._calculate(q)
            if result is not None:
                return f"The answer is {result}"
        
        # Help
        if any(w in q for w in ["help", "what can you do", "capabilities"]):
            return (
                "Here's what I can do:\n"
                "• Tell time and date\n"
                "• Do calculations\n"
                "• Answer questions (with Groq API key)\n"
                "• Type 'exit' to quit"
            )
        
        # Greeting
        if any(w in q for w in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm Jarvis. How can I help you today?"
        
        return None
    
    def _calculate(self, query: str) -> Optional[float]:
        q = re.sub(r'\b(what is|calculate|compute|equals)\b', '', query.lower())
        q = re.sub(r'[?.,!\s]', '', q)
        
        match = re.search(r'(\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*\d+)*)', q)
        if match:
            expr = match.group(1).replace('^', '**')
            try:
                return round(eval(expr, {"__builtins__": {}}, {}), 4)
            except:
                pass
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
# GROQ BRAIN (If package available)
# ============================================================================

class GroqBrain:
    """Groq API integration."""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.client = Groq(api_key=api_key) if Groq else None
        self.api_key = api_key
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key.startswith("gsk_")
    
    def generate(self, messages: List[Dict]) -> str:
        if not self.client:
            return "AI not available. Please install the groq package or set GROQ_API_KEY."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return f"Error: {str(e)[:100]}"


# ============================================================================
# JARVIS MAIN CLASS
# ============================================================================

class Jarvis:
    """Main Jarvis assistant."""
    
    def __init__(self, mode: str = "online"):
        self.mode = mode
        self.intent_handler = IntentHandler()
        self.memory = Memory(CONFIG["max_history"])
        self.logger = Logger()
        
        # Try to get API key and initialize Groq
        api_key = get_api_key()
        if api_key and HAS_GROQ:
            self.brain = GroqBrain(api_key)
            self.use_ai = True
            logger.info(f"✓ Groq AI enabled (model: {self.brain.model})")
        else:
            self.brain = None
            self.use_ai = False
            if not api_key:
                logger.info("⚠ No API key - using rule-based mode only")
            else:
                logger.info("⚠ Groq package not installed - using rule-based mode only")
        
        self._show_banner()
    
    def _show_banner(self):
        print("\n" + "=" * 50)
        print("  JARVIS AI ASSISTANT")
        print("=" * 50)
        print(f"  Mode: {'AI (Groq)' if self.use_ai else 'Rule-based (demo)'}")
        print(f"  Input: Text")
        print("=" * 50)
        print("\nCommands:")
        print("  • Type 'exit' to quit")
        print("  • Type 'help' for capabilities")
        print("  • Ask me anything!\n")
    
    def chat(self, query: str) -> str:
        """Single chat - returns response."""
        query = query.strip()
        if not query:
            return ""
        
        # Check exit
        if query.lower() in ["exit", "quit", "bye", "goodbye", "stop"]:
            return "Goodbye! Have a great day."
        
        # Check rule-based intents first
        intent_response = self.intent_handler.check(query)
        if intent_response:
            self.memory.add(query, intent_response)
            return intent_response
        
        # Use AI or fallback
        if self.use_ai and self.brain:
            messages = self.memory.get_context()
            messages.append({"role": "user", "content": query})
            
            response = self.brain.generate(messages)
            self.memory.add(query, response)
            return response
        else:
            # No AI available - show helpful message
            self.memory.add(query, f"I'd love to help with that! To enable AI responses, install the groq package and set your GROQ_API_KEY.")
            return f"I can help with time, date, and calculations. For AI responses, install groq package and set GROQ_API_KEY."


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def run_interactive():
    """Run interactive chat session."""
    jarvis = Jarvis()
    
    while True:
        try:
            # Try different input methods
            if HAS_JS:
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
            
            # Check for exit
            if user_input.lower() in ["exit", "quit", "bye", "goodbye", "stop"]:
                break
                
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


# ============================================================================
# DEMO MODE (No API key needed)
# ============================================================================

def run_demo():
    """Run demo showing rule-based features."""
    print("\n" + "=" * 50)
    print("  JARVIS DEMO MODE")
    print("=" * 50)
    print("\nShowing rule-based features (no AI required):\n")
    
    examples = [
        "What time is it?",
        "Calculate 25 * 4",
        "What's today's date?",
        "Help",
    ]
    
    jarvis = Jarvis()
    
    for query in examples:
        print(f"You: {query}")
        response = jarvis.chat(query)
        print(f"Jarvis: {response}\n")


# ============================================================================
# QUICK TEST FUNCTION
# ============================================================================

def test():
    """Quick test of Jarvis."""
    print("\n" + "=" * 50)
    print("  JARVIS QUICK TEST")
    print("=" * 50 + "\n")
    
    jarvis = Jarvis()
    
    tests = [
        ("What time is it?", "time response"),
        ("Calculate 100 / 4", "calculation response"),
        ("Hello!", "greeting response"),
    ]
    
    for query, expected in tests:
        print(f"Test: {query}")
        response = jarvis.chat(query)
        print(f"Response: {response[:80]}...\n")
    
    print("=" * 50)
    print("Tests complete!")
    print("=" * 50)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Check for API key
    api_key = get_api_key()
    
    if api_key and HAS_GROQ:
        print(f"\n✓ Groq API key found! Starting interactive mode...\n")
        run_interactive()
    else:
        print("\n" + "=" * 50)
        print("  JARVIS - SETUP REQUIRED FOR AI MODE")
        print("=" * 50)
        print("\nTo enable AI responses:")
        print("1. Get free API key: https://console.groq.com")
        print("2. Install: pip install groq")
        print("3. Set key: export GROQ_API_KEY='gsk_...'")
        print("\n" + "-" * 50 + "\n")
        
        # Run demo
        run_demo()
