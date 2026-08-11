#!/usr/bin/env python3
"""
Jarvis AI Assistant - Optimized for "Coding Python" App
=========================================================

A streamlined version designed for Python learning/coding apps.

Features:
✅ Works without pip installs (uses only built-in modules)
✅ Time, Date, Calculator - instant responses
✅ AI integration via Groq (if API key available)
✅ Clean, simple interface for coding environments

Usage in Coding Python App:
----------------------------
1. Copy this entire file into your app
2. Set API key if you have one (optional)
3. Run and start chatting!

Quick Start:
------------
import os
os.environ["GROQ_API_KEY"] = "gsk_..."

import jarvis_coding_python as jarvis
jarvis.run()
"""

import os
import sys
import json
import re
from collections import deque
from datetime import datetime, date as date_class
from typing import List, Dict, Optional

# ============================================================================
# SETUP - Check for API key
# ============================================================================

API_KEY = os.environ.get("GROQ_API_KEY", "")

# Try to import groq (optional)
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    Groq = None
    HAS_GROQ = False


# ============================================================================
# GROQ CLIENT (if available)
# ============================================================================

class GroqClient:
    """Simple Groq API client."""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key) if Groq else None
        self.api_key = api_key
    
    def chat(self, model: str, messages: list, max_tokens: int = 512) -> str:
        """Send chat request."""
        if not self.client:
            return "API not available"
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)[:80]}"


# ============================================================================
# INTENT HANDLER
# ============================================================================

class Intents:
    """Handle common queries without AI."""
    
    def check(self, query: str) -> Optional[str]:
        q = query.lower().strip()
        
        # Time
        if "time" in q:
            now = datetime.now()
            return f"Current time: {now.strftime('%H:%M')}"
        
        # Date
        if "date" in q or "today" in q:
            today = date_class.today()
            return f"Today: {today.strftime('%Y-%m-%d')}"
        
        # Calculator
        if any(w in q for w in ["calculate", "math", "compute", "what is"]):
            result = self._math(q)
            if result is not None:
                return f"Result: {result}"
        
        # Python help
        if any(w in q for w in ["python", "code", "program", "how to"]):
            return self._python_help(q)
        
        # Greeting
        if any(w in q for w in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm Jarvis. Ready to help you code!"
        
        # Who are you
        if any(w in q for w in ["who are you", "your name", "what are you"]):
            return "I'm Jarvis, your Python coding assistant!"
        
        # Help
        if "help" in q:
            return (
                "I can help with:\n"
                "• Time & Date\n"
                "• Calculations\n"
                "• Python questions\n"
                "• General Q&A (with AI)\n\n"
                "Just ask!"
            )
        
        # Exit
        if any(w in q for w in ["exit", "quit", "bye"]):
            return "exit"
        
        return None
    
    def _math(self, query: str) -> Optional[float]:
        q = re.sub(r'\b(what is|calculate)\b', '', query.lower())
        q = re.sub(r'[?.,!\s]', '', q)
        
        match = re.search(r'(-?\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(-?\d+(?:\.\d+)?)', q)
        if match:
            a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
            if op == '+': return a + b
            if op == '-': return a - b
            if op == '*': return a * b
            if op == '/' and b != 0: return a / b
        return None
    
    def _python_help(self, query: str) -> str:
        """Provide Python-related help."""
        q = query.lower()
        
        if "print" in q:
            return "Use print() to display output: print('Hello')"
        if "loop" in q or "for" in q:
            return "For loop example:\nfor i in range(5):\n    print(i)"
        if "if" in q or "condition" in q:
            return "If statement:\nif x > 5:\n    print('Big')"
        if "list" in q:
            return "Create a list: my_list = [1, 2, 3]"
        if "function" in q or "def" in q:
            return "Define a function:\ndef greet(name):\n    return f'Hello, {name}'"
        if "import" in q:
            return "Import a module: import math\nThen use: math.sqrt(16)"
        
        return "I can help with Python basics. Ask me about: print, loops, if statements, lists, functions, imports..."


# ============================================================================
# MEMORY
# ============================================================================

class Memory:
    def __init__(self):
        self.history = deque(maxlen=10)
    
    def add(self, user: str, response: str):
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": response})
    
    def get_context(self) -> list:
        return list(self.history)


# ============================================================================
# JARVIS
# ============================================================================

class Jarvis:
    """Main Jarvis assistant."""
    
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model
        self.intents = Intents()
        self.memory = Memory()
        self.ai = None
        
        # Setup AI if possible
        if API_KEY and HAS_GROQ:
            self.ai = GroqClient(API_KEY)
    
    def ask(self, query: str) -> str:
        """Ask Jarvis a question."""
        query = query.strip()
        if not query:
            return ""
        
        # Check intents first
        intent = self.intents.check(query)
        if intent == "exit":
            return "Goodbye!"
        if intent:
            self.memory.add(query, intent)
            return intent
        
        # Use AI
        if self.ai:
            try:
                messages = self.memory.get_context()
                messages.append({"role": "user", "content": query})
                response = self.ai.chat(self.model, messages)
                self.memory.add(query, response)
                return response
            except:
                return "AI error. Try a simpler question."
        else:
            return "AI not available. Install 'groq' package and set GROQ_API_KEY."


# ============================================================================
# RUN INTERACTIVE
# ============================================================================

def run():
    """Run interactive chat."""
    jarvis = Jarvis()
    
    print("\n" + "=" * 40)
    print("  JARVIS - Coding Python Assistant")
    print("=" * 40)
    print(f"  AI: {'On' if jarvis.ai else 'Off'}")
    print("=" * 40)
    print("\nType 'help' or ask anything!\n")
    
    while True:
        try:
            q = input(">>> ").strip()
            if not q:
                continue
            
            response = jarvis.ask(q)
            print(f"\n{response}\n")
            
            if q.lower() in ["exit", "quit", "bye"]:
                break
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break


# ============================================================================
# QUICK TEST
# ============================================================================

def test():
    """Quick test of all features."""
    print("\n" + "=" * 40)
    print("  JARVIS TEST")
    print("=" * 40 + "\n")
    
    jarvis = Jarvis()
    
    tests = [
        "What time is it?",
        "Calculate 50 * 50",
        "What is today?",
        "Hello!",
        "Who are you?",
        "Help",
    ]
    
    for q in tests:
        print(f"Q: {q}")
        print(f"A: {jarvis.ask(q)}\n")


# ============================================================================

if __name__ == "__main__":
    # Check for API key
    if API_KEY:
        print("✓ API key found")
    else:
        print("⚠ No API key - AI disabled")
    
    # Run test or interactive
    run()
