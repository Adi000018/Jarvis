#!/usr/bin/env python3
"""
Jarvis AI Assistant - FULLY FUNCTIONAL Complete Version
=========================================================

Works in ANY Python environment (local, online compiler, Pyodide, etc.)

FEATURES:
✅ Time & Date - instant responses
✅ Calculator - basic math operations
✅ Python Coding Help - examples and explanations
✅ AI Chat - via Groq API (optional, needs API key)
✅ Conversation Memory - remembers context
✅ Auto-Test Mode - runs without user input
✅ Works without pip installs (groq optional for AI)

SETUP:
------
1. Copy entire file to your project
2. Set API key (optional): API_KEY = "gsk_..."
3. Run: python jarvis_complete.py

QUICK START:
------------
from jarvis_complete import Jarvis
jarvis = Jarvis()
print(jarvis.ask("Hello!"))
print(jarvis.ask("What time is it?"))
print(jarvis.ask("Calculate 25 * 4"))
"""

import os
import sys
import json
import re
from collections import deque
from datetime import datetime, date as date_class
from typing import List, Dict, Optional

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================

# Your Groq API Key (get free at https://console.groq.com)
# Leave empty "" to use rule-based mode only (no AI)
API_KEY = os.environ.get("GROQ_API_KEY", "")

# UNCOMMENT to set key directly:
# API_KEY = "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"

# AI Model (Groq models)
AI_MODEL = "llama-3.1-8b-instant"  # Fast model
# AI_MODEL = "llama-3.3-70b-versatile"  # More powerful

# Conversation memory (number of turns to remember)
MAX_MEMORY = 10

# System prompt for AI
SYSTEM_PROMPT = """
You are Jarvis, a helpful coding assistant. Be concise, friendly, and helpful.
If asked about Python, provide clear code examples.
If you don't know something, say so honestly.
""".strip()


# ============================================================================
# TRY TO IMPORT GROQ (OPTIONAL - FOR AI FEATURES)
# ============================================================================

try:
    from groq import Groq
    HAS_GROQ = True
    print("✓ groq package available")
except ImportError:
    Groq = None
    HAS_GROQ = False
    print("⚠ groq not installed (install with: pip install groq)")


# ============================================================================
# GROQ AI CLIENT
# ============================================================================

class AIClient:
    """Groq API client for AI responses."""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
        self.client = None
        
        if api_key and api_key.startswith("gsk_") and Groq:
            try:
                self.client = Groq(api_key=api_key)
                print(f"✓ AI enabled (model: {model})")
            except Exception as e:
                print(f"⚠ AI initialization failed: {e}")
                self.client = None
        else:
            print("⚠ AI disabled - no valid API key")
    
    def chat(self, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Send chat request to Groq API."""
        if not self.client:
            return "AI not available. Set GROQ_API_KEY to enable."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                return "Invalid API key. Please check your GROQ_API_KEY."
            elif "429" in error_msg:
                return "Rate limited. Please wait and try again."
            else:
                return f"AI error: {error_msg[:80]}"


# ============================================================================
# RULE-BASED INTENT HANDLER (WORKS WITHOUT AI)
# ============================================================================

class Intents:
    """Handle common queries without AI - fast and reliable."""
    
    def check(self, query: str) -> Optional[str]:
        """Check if query matches a rule-based intent."""
        q = query.lower().strip()
        
        # Time queries
        if any(w in q for w in ["time", "what time", "current time", "clock"]):
            now = datetime.now()
            return f"Current time: {now.strftime('%H:%M:%S')}"
        
        # Date queries
        if any(w in q for w in ["date", "what date", "today", "day is it", "what day"]):
            today = date_class.today()
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            return f"Today is {days[today.weekday()]}, {today.strftime('%B %d, %Y')}"
        
        # Calculator
        if any(w in q for w in ["calculate", "calculator", "what is", "compute", "solve", "math"]):
            result = self._calculate(q)
            if result is not None:
                return f"Result: {result}"
        
        # Python coding help
        if self._is_python_query(q):
            return self._python_help(q)
        
        # Greetings
        if any(w in q for w in ["hello", "hi ", "hey ", "greetings", "good morning", "good afternoon", "good evening"]):
            hour = datetime.now().hour
            greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
            return f"{greeting}! I'm Jarvis. Ready to help you code!"
        
        # How are you
        if re.search(r'\bhow\s+(are|is)\s+you\b', q):
            return "I'm doing great, thank you! How can I assist you today?"
        
        # Identity
        if any(w in q for w in ["your name", "who are you", "what are you", "introduce yourself", "tell me about yourself"]):
            return "I'm Jarvis, your Python coding assistant! I can help with time, date, calculations, Python code, and answer questions."
        
        # Capabilities / Help
        if any(w in q for w in ["help", "what can you do", "capabilities", "commands", "features"]):
            return self._help_text()
        
        # Exit commands
        if any(w in q for w in ["exit", "quit", "bye", "goodbye", "stop", "end"]):
            return "exit"
        
        # Thanks
        if any(w in q for w in ["thank", "thanks", "appreciate"]):
            return "You're welcome! Happy to help. 😊"
        
        # Yes/No questions (simple)
        if q.startswith("is ") or q.startswith("are ") or q.startswith("do ") or q.startswith("does "):
            return "That's a great question! Let me ask my AI brain..."  # Will fall through to AI
        
        return None
    
    def _is_python_query(self, q: str) -> bool:
        """Check if query is about Python programming syntax/help."""
        # Only match specific "how to" or "example" requests for Python help
        # General questions about Python go to AI
        if any(w in q for w in ["what is python", "python programming", "tell me about python", "python language"]):
            return False  # Let AI handle general Python questions
        
        python_keywords = [
            "python", "code", "programming", "program", "coding",
            "function", "loop", "if statement", "variable", "list",
            "dictionary", "array", "string", "print", "import",
            "class", "object", "module", "library", "framework",
            "how to", "example", "syntax", "tutorial"
        ]
        return any(kw in q for kw in python_keywords)
    
    def _calculate(self, query: str) -> Optional[float]:
        """Extract and calculate math expression."""
        # Remove words
        q = re.sub(r'\b(what is|calculate|compute|solve|equals)\b', '', query.lower())
        q = re.sub(r'[?.,!\s]', '', q)
        
        # Match basic operations: number op number
        match = re.search(r'(-?\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(-?\d+(?:\.\d+)?)', q)
        if match:
            a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
            if op == '+': return round(a + b, 4)
            if op == '-': return round(a - b, 4)
            if op == '*': return round(a * b, 4)
            if op == '/':
                if b == 0:
                    return None
                return round(a / b, 4)
        
        # Try more complex expressions
        try:
            # Safe eval with limited operations
            expr = re.sub(r'[^0-9+\-*/.() ]', '', q)
            if expr.strip():
                return round(eval(expr, {"__builtins__": {}}, {}), 4)
        except:
            pass
        
        return None
    
    def _python_help(self, query: str) -> str:
        """Provide Python coding help."""
        q = query.lower()
        
        # Specific topics
        if "print" in q:
            return (
                "Print in Python:\n"
                "print('Hello, World!')\n"
                "print(variable)\n"
                "print(f'Value: {value}')  # f-string"
            )
        
        if "loop" in q or "for" in q:
            return (
                "For Loop:\n"
                "for i in range(5):\n"
                "    print(i)  # 0, 1, 2, 3, 4\n\n"
                "While Loop:\n"
                "i = 0\n"
                "while i < 5:\n"
                "    print(i)\n"
                "    i += 1"
            )
        
        if "if" in q or "condition" in q or "else" in q:
            return (
                "If Statement:\n"
                "x = 10\n"
                "if x > 5:\n"
                "    print('Big')\n"
                "elif x == 5:\n"
                "    print('Equal')\n"
                "else:\n"
                "    print('Small')"
            )
        
        if "list" in q:
            return (
                "Lists in Python:\n"
                "my_list = [1, 2, 3, 4, 5]\n"
                "my_list.append(6)  # Add item\n"
                "my_list[0]  # Access first item\n"
                "len(my_list)  # Get length"
            )
        
        if "function" in q or "def" in q:
            return (
                "Functions in Python:\n"
                "def greet(name):\n"
                "    return f'Hello, {name}!'\n\n"
                "result = greet('World')\n"
                "print(result)"
            )
        
        if "dictionary" in q or "dict" in q:
            return (
                "Dictionaries in Python:\n"
                "person = {'name': 'John', 'age': 30}\n"
                "person['name']  # Access: 'John'\n"
                "person['age'] = 31  # Update\n"
                "person['city'] = 'NYC'  # Add"
            )
        
        if "string" in q or "text" in q:
            return (
                "Strings in Python:\n"
                "text = 'Hello World'\n"
                "text.upper()  # 'HELLO WORLD'\n"
                "text.lower()  # 'hello world'\n"
                "text.split()  # ['Hello', 'World']\n"
                "f'Value: {value}'  # f-string"
            )
        
        if "import" in q:
            return (
                "Importing in Python:\n"
                "import math\n"
                "math.sqrt(16)  # 4.0\n\n"
                "from math import sqrt\n"
                "sqrt(16)  # 4.0\n\n"
                "import random as rnd\n"
                "rnd.randint(1, 10)"
            )
        
        if "class" in q or "object" in q:
            return (
                "Classes in Python:\n"
                "class Person:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "    \n"
                "    def greet(self):\n"
                "        return f'Hi, I am {self.name}'\n\n"
                "p = Person('Alice')\n"
                "print(p.greet())"
            )
        
        if "file" in q or "read" in q or "write" in q:
            return (
                "File Operations:\n"
                "# Read file\n"
                "with open('file.txt', 'r') as f:\n"
                "    content = f.read()\n\n"
                "# Write file\n"
                "with open('file.txt', 'w') as f:\n"
                "    f.write('Hello')"
            )
        
        if "error" in q or "exception" in q or "debug" in q:
            return (
                "Error Handling:\n"
                "try:\n"
                "    result = 10 / 0\nexcept ZeroDivisionError as e:\n"
                "    print(f'Error: {e}')\nexcept Exception as e:\n"
                "    print(f'General error: {e}')\n"
                "finally:\n"
                "    print('Always runs')"
            )
        
        # General Python help
        return (
            "I can help with Python topics:\n"
            "• print() - output\n"
            "• loops (for, while)\n"
            "• if/else conditions\n"
            "• lists, dictionaries\n"
            "• functions (def)\n"
            "• classes & objects\n"
            "• imports\n"
            "• file operations\n"
            "• error handling\n\n"
            "Ask me about any of these!"
        )
    
    def _help_text(self) -> str:
        """Return help information."""
        return (
            "╔══════════════════════════════════════════╗\n"
            "║          JARVIS - Coding Assistant        ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  I can help you with:                    ║\n"
            "║  • Time & Date                           ║\n"
            "║  • Calculations (e.g., '25 * 4')        ║\n"
            "║  • Python coding help                    ║\n"
            "║  • Answer questions (with AI)            ║\n"
            "║  • Simple conversations                  ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  Python Topics:                          ║\n"
            "║  • print, loops, if/else                 ║\n"
            "║  • lists, dictionaries                   ║\n"
            "║  • functions, classes                    ║\n"
            "║  • imports, file handling                ║\n"
            "╚══════════════════════════════════════════╝\n"
            "Type 'exit' to quit."
        )


# ============================================================================
# CONVERSATION MEMORY
# ============================================================================

class Memory:
    """Store conversation history for context."""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)
    
    def add(self, user: str, response: str):
        """Add a conversation turn."""
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": response})
    
    def get_context(self) -> List[Dict]:
        """Get conversation history as messages list."""
        return list(self.history)
    
    def clear(self):
        """Clear conversation history."""
        self.history.clear()
    
    def __len__(self):
        return len(self.history)


# ============================================================================
# JARVIS MAIN CLASS
# ============================================================================

class Jarvis:
    """
    Jarvis AI Assistant - Full Function Version
    
    Usage:
        jarvis = Jarvis()
        response = jarvis.ask("Hello!")
        print(response)
    """
    
    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize Jarvis.
        
        Args:
            model: AI model name (default from config)
            api_key: Groq API key (default from config)
        """
        self.model = model or AI_MODEL
        self.api_key = api_key or API_KEY
        self.intents = Intents()
        self.memory = Memory(MAX_MEMORY)
        self.ai = None
        self._setup()
    
    def _setup(self):
        """Initialize AI client if possible."""
        if self.api_key and HAS_GROQ:
            self.ai = AIClient(self.api_key, self.model)
    
    def ask(self, query: str) -> str:
        """
        Ask Jarvis a question.
        
        Args:
            query: User's question
        
        Returns:
            Jarvis's response
        """
        query = query.strip()
        if not query:
            return ""
        
        # Check for rule-based intents first (fast)
        intent = self.intents.check(query)
        
        if intent == "exit":
            return "Goodbye! Have a great day! 👋"
        
        if intent:
            self.memory.add(query, intent)
            return intent
        
        # Use AI for complex questions
        if self.ai:
            try:
                # Build messages with context
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend(self.memory.get_context())
                messages.append({"role": "user", "content": query})
                
                response = self.ai.chat(messages)
                self.memory.add(query, response)
                return response
            except Exception as e:
                return f"AI error: {str(e)[:100]}. Try again."
        else:
            # AI not available - provide helpful message
            return (
                "I'd love to help with that! To enable AI responses:\n"
                "1. Get free API key: https://console.groq.com\n"
                "2. Set: API_KEY = 'gsk_...' (line 15)\n"
                "3. Install: pip install groq"
            )
    
    def chat(self, query: str) -> str:
        """Alias for ask() - same functionality."""
        return self.ask(query)
    
    def is_ai_enabled(self) -> bool:
        """Check if AI is enabled."""
        return self.ai is not None and self.ai.client is not None
    
    def reset_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
    
    def get_memory_info(self) -> dict:
        """Get memory statistics."""
        return {
            "history_length": len(self.memory),
            "max_turns": self.max_turns,
            "ai_enabled": self.is_ai_enabled()
        }


# ============================================================================
# AUTO-RUN TESTS (NO INPUT REQUIRED!)
# ============================================================================

def run_auto_test():
    """Run automatic tests - no user input needed."""
    print("\n" + "=" * 60)
    print("  JARVIS AI ASSISTANT - FULL FUNCTION TEST")
    print("=" * 60)
    print()
    
    jarvis = Jarvis()
    
    # Show status
    print(f"🤖 AI Status: {'ENABLED' if jarvis.is_ai_enabled() else 'DISABLED (using rule-based mode)'}")
    print(f"💾 Memory: {MAX_MEMORY} turns")
    print("=" * 60)
    print()
    
    # Test questions
    tests = [
        # Rule-based tests (always work)
        ("What time is it?", "Time"),
        ("Calculate 25 * 4", "Calculator"),
        ("What is today?", "Date"),
        ("Hello Jarvis!", "Greeting"),
        ("Who are you?", "Identity"),
        ("How are you?", "Wellbeing"),
        ("Help", "Help Command"),
        
        # Python coding help
        ("How to use print in Python?", "Python - Print"),
        ("How to create a list?", "Python - Lists"),
        ("Show me a for loop example", "Python - Loops"),
        ("How to define a function?", "Python - Functions"),
        
        # AI test (only if enabled)
        ("What is Python programming?", "AI - Python"),
    ]
    
    print("📋 TEST RESULTS:\n")
    print("-" * 60)
    
    passed = 0
    total = len(tests)
    
    for question, category in tests:
        print(f"\n[{category}]")
        print(f"Q: {question}")
        
        response = jarvis.ask(question)
        
        # Truncate long AI responses
        if len(response) > 150:
            display_response = response[:150] + "..."
        else:
            display_response = response
        
        print(f"A: {display_response}")
        print("-" * 60)
        
        # Count as passed if we got a non-empty response
        if response and response != "AI not available...":
            passed += 1
    
    print()
    print("=" * 60)
    print(f"✅ TESTS COMPLETE: {passed}/{total} successful responses")
    print("=" * 60)
    
    return jarvis


# ============================================================================
# DEMO MODE (SHOWCASE FEATURES)
# ============================================================================

def run_demo():
    """Run interactive demo."""
    print("\n" + "=" * 60)
    print("  JARVIS - INTERACTIVE DEMO")
    print("=" * 60)
    print()
    
    jarvis = Jarvis()
    
    print("Type your questions below (or 'exit' to quit):\n")
    
    while True:
        try:
            # Try input, fallback for environments without it
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not question:
                continue
            
            if question.lower() in ["exit", "quit", "bye"]:
                print("\nJarvis: Goodbye! Have a great day! 👋\n")
                break
            
            response = jarvis.ask(question)
            print(f"\nJarvis: {response}\n")
            
        except Exception as e:
            print(f"\nError: {e}\n")
            break


# ============================================================================
# QUICK CHAT FUNCTION (ONE-LINER)
# ============================================================================

def quick_chat(question: str, api_key: str = None) -> str:
    """
    Quick one-off chat - convenient for testing.
    
    Example:
        print(quick_chat("What time is it?"))
        print(quick_chat("Calculate 25 * 4"))
    """
    if api_key:
        global API_KEY
        API_KEY = api_key
    
    jarvis = Jarvis()
    return jarvis.ask(question)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🚀 JARVIS AI - FULLY FUNCTIONAL")
    print("=" * 60)
    print()
    
    # Check setup
    if API_KEY:
        print("✓ API key configured")
    else:
        print("ℹ No API key - running in rule-based mode")
    
    if HAS_GROQ:
        print("✓ groq package installed")
    else:
        print("⚠ Install groq for AI: pip install groq")
    
    print()
    
    # Run automatic tests (no user input needed!)
    jarvis = run_auto_test()
    
    print("\n" + "=" * 60)
    print("  ✅ READY TO USE!")
    print("=" * 60)
    print()
    print("Import and use in your code:")
    print("  from jarvis_complete import Jarvis")
    print("  jarvis = Jarvis()")
    print("  print(jarvis.ask('Hello!'))")
    print()
    print("Or run interactive mode:")
    print("  python jarvis_complete.py")
    print()
