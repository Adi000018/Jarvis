#!/usr/bin/env python3
"""
Jarvis AI Assistant - Coding Python App Edition
================================================

Works in any Python environment including online compilers.

Setup:
------
1. Set your API key at the top (line 15)
2. Run the script
3. See test results automatically!

QUICK TEST:
-----------
Just run this file - it will test all features automatically.
"""

import os
import sys
import json
import re
from collections import deque
from datetime import datetime, date as date_class
from typing import List, Dict, Optional

# ============================================================================
# SETUP - PUT YOUR API KEY HERE (Line 15)
# ============================================================================

# Get API key from environment, or set it here:
API_KEY = os.environ.get("GROQ_API_KEY", "")

# UNCOMMENT AND ADD YOUR KEY IF NEEDED:
# API_KEY = "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"

# ============================================================================
# TRY TO IMPORT GROQ (OPTIONAL - FOR AI FEATURES)
# ============================================================================

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    Groq = None
    HAS_GROQ = False
    print("⚠ groq package not installed (pip install groq)")


# ============================================================================
# GROQ CLIENT (IF AVAILABLE)
# ============================================================================

class GroqClient:
    """Simple Groq API client."""
    
    def __init__(self, api_key: str):
        if Groq and api_key.startswith("gsk_"):
            self.client = Groq(api_key=api_key)
            self.api_key = api_key
        else:
            self.client = None
            self.api_key = None
    
    def chat(self, model: str, messages: list, max_tokens: int = 512) -> str:
        """Send chat request."""
        if not self.client:
            return "AI not available"
        
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
# INTENT HANDLER (WORKS WITHOUT AI)
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
        
        # For specific simple queries, check intents first
        simple_intents = ["time", "date", "today", "calculate", "hello", "hi ", "hey ", "who are you", "your name", "what are you", "help", "exit", "quit", "bye"]
        
        q = query.lower()
        is_simple = any(s in q for s in simple_intents)
        
        # Check intents for simple queries
        if is_simple:
            intent = self.intents.check(query)
            if intent == "exit":
                return "Goodbye!"
            if intent:
                self.memory.add(query, intent)
                return intent
        
        # Use AI for everything else (or complex questions)
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
            # AI not available - check intents for everything
            intent = self.intents.check(query)
            if intent == "exit":
                return "Goodbye!"
            if intent:
                return intent
            return "AI not available. Install 'groq' package and set GROQ_API_KEY."


# ============================================================================
# AUTO-RUN TESTS (NO INPUT() NEEDED!)
# ============================================================================

def run_tests():
    """Run automatic tests - no user input needed!"""
    print("\n" + "=" * 50)
    print("  JARVIS - CODING PYTHON ASSISTANT")
    print("=" * 50)
    print(f"\n  AI: {'ON' if Jarvis().ai else 'OFF'}")
    print("=" * 50)
    print()
    
    jarvis = Jarvis()
    
    # Test questions
    tests = [
        ("What time is it?", "Time"),
        ("Calculate 25 * 4", "Calculator"),
        ("What is today?", "Date"),
        ("Hello!", "Greeting"),
        ("Who are you?", "Identity"),
        ("Help", "Help"),
        ("How to use loops in Python?", "Python Help"),
    ]
    
    print("TEST RESULTS:\n")
    print("-" * 50)
    
    for question, category in tests:
        print(f"\n[{category}]")
        print(f"Q: {question}")
        answer = jarvis.ask(question)
        print(f"A: {answer}")
        print("-" * 50)
    
    print("\n✅ All tests complete!")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # Show status
    if API_KEY:
        print("✓ API key found!")
    else:
        print("⚠ No API key - using rule-based mode only")
    
    if HAS_GROQ:
        print("✓ groq package installed!")
    else:
        print("⚠ groq not installed - AI features disabled")
    
    print()
    
    # Run automatic tests
    run_tests()
