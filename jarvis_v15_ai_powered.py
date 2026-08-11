#!/usr/bin/env python3
"""
Jarvis AI Agent v15.0 (AI-POWERED EDITION)
Integrated with Open-Source LLMs for Real Answers

Features:
- Real AI reasoning with Ollama or LlamaCpp
- Actual knowledge and context understanding
- Smart task management + intelligent conversation
- No API keys required - runs locally
- Support for multiple open-source models
- Fallback to local utilities when offline

Supported Models:
- Mistral 7B (recommended, fastest)
- Llama 2 7B/13B (most capable)
- Neural Chat 7B (optimized for chat)
- Phi 2 (small, fast)
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import difflib
import hashlib
import json
import logging
import math
import os
import re
import secrets
import socket
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, pstdev, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum

APP_VERSION = "15.0 AI-Powered (Open-Source Edition)"
APP_DIR = Path.home() / "jarvis_agent_v15"
FILES_DIR = APP_DIR / "files"
CONFIG_FILE = APP_DIR / "config.json"
MEMORY_FILE = APP_DIR / "memory.json"
NOTES_FILE = APP_DIR / "notes.txt"
TODO_FILE = APP_DIR / "todo.json"
REMINDER_FILE = APP_DIR / "reminders.json"
LOG_FILE = APP_DIR / "jarvis.log"
KNOWLEDGE_FILE = APP_DIR / "knowledge.json"

logger = logging.getLogger("jarvis_v15")


def setup_logging() -> None:
    """Configure logging."""
    APP_DIR.mkdir(exist_ok=True)
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)


# =============================================================================
# AI BACKEND INTEGRATION
# =============================================================================

class AIBackend:
    """Base class for AI backend integration."""
    
    def __init__(self):
        self.available = False
        self.model_name = "None"
    
    def query(self, prompt: str, context: str = "") -> str:
        """Query the AI with a prompt."""
        raise NotImplementedError
    
    def health_check(self) -> bool:
        """Check if backend is available."""
        raise NotImplementedError


class OllamaBackend(AIBackend):
    """Integration with Ollama (easiest to set up)."""
    
    def __init__(self, model: str = "mistral", host: str = "http://localhost:11434"):
        super().__init__()
        self.model = model
        self.host = host
        self.endpoint = f"{host}/api/generate"
        self.available = self.health_check()
        if self.available:
            self.model_name = f"Ollama ({model})"
            logger.info(f"✓ Connected to Ollama with {model}")
        else:
            logger.info("⚠ Ollama not available (see setup instructions)")
    
    def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = urllib.request.urlopen(
                f"{self.host}/api/tags", 
                timeout=2
            )
            return response.status == 200
        except Exception:
            return False
    
    def query(self, prompt: str, context: str = "") -> str:
        """Query Ollama API."""
        if not self.available:
            return self._offline_response()
        
        try:
            # Construct full prompt with context
            full_prompt = f"{context}\n\nUser: {prompt}\n\nAssistant:" if context else f"{prompt}\n\nAssistant:"
            
            # Prepare request
            data = json.dumps({
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
            }).encode('utf-8')
            
            # Make request
            req = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode())
                answer = result.get("response", "").strip()
                return answer if answer else "I couldn't generate a response."
                
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return self._offline_response()
    
    def _offline_response(self) -> str:
        return (
            "I'm currently offline. To enable AI features:\n\n"
            "Install Ollama:\n"
            "  1. Download from ollama.ai\n"
            "  2. Run: ollama pull mistral\n"
            "  3. Run: ollama serve\n"
            "  4. Restart Jarvis\n\n"
            "Then I can answer real questions!"
        )


class LlamaCppBackend(AIBackend):
    """Integration with LlamaCpp (Python bindings)."""
    
    def __init__(self, model_path: str = None):
        super().__init__()
        try:
            from llama_cpp import Llama
            self.Llama = Llama
            self.model_path = model_path or self._find_model()
            if self.model_path and Path(self.model_path).exists():
                self.llm = Llama(self.model_path, n_gpu_layers=-1)
                self.available = True
                self.model_name = f"LlamaCpp ({Path(model_path).name})"
                logger.info(f"✓ Loaded LlamaCpp model: {model_path}")
            else:
                logger.info("⚠ No model found for LlamaCpp")
        except ImportError:
            logger.info("⚠ LlamaCpp not installed (pip install llama-cpp-python)")
    
    def _find_model(self) -> Optional[str]:
        """Find a model in common locations."""
        common_paths = [
            Path.home() / "models" / "mistral-7b.gguf",
            Path.home() / "models" / "llama-2-7b.gguf",
            Path("/opt/models/mistral.gguf"),
        ]
        for path in common_paths:
            if path.exists():
                return str(path)
        return None
    
    def health_check(self) -> bool:
        """Check if LlamaCpp is available."""
        return self.available
    
    def query(self, prompt: str, context: str = "") -> str:
        """Query LlamaCpp model."""
        if not self.available:
            return "LlamaCpp backend not initialized."
        
        try:
            full_prompt = f"{context}\n\nUser: {prompt}" if context else prompt
            response = self.llm(
                full_prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
            )
            answer = response["choices"][0]["text"].strip()
            return answer if answer else "I couldn't generate a response."
        except Exception as e:
            logger.error(f"LlamaCpp error: {e}")
            return f"Error: {e}"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def now_text() -> str:
    """Get current timestamp."""
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_spaces(text: Any) -> str:
    """Remove extra whitespace."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "who", "what", "why", "how", "when", "where", "which", "to", "of",
        "in", "on", "at", "by", "for", "with", "from", "and", "or", "but",
    }
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    return [w for w in words if len(w) > 2 and w not in stopwords]


# =============================================================================
# CORE FEATURES (from v14)
# =============================================================================

class Notes:
    """Note management."""

    def __init__(self):
        self.notes: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict]:
        try:
            if NOTES_FILE.exists():
                return json.loads(NOTES_FILE.read_text(encoding="utf-8"))
        except:
            pass
        return []

    def add(self, note: str) -> str:
        """Add note."""
        self.notes.append({
            "id": len(self.notes) + 1,
            "text": note,
            "timestamp": now_text(),
        })
        self._save()
        return f"✓ Note saved: {note[:50]}..."

    def show(self) -> str:
        """Show notes."""
        if not self.notes:
            return "No notes yet."
        lines = ["📝 Your Notes:"]
        for note in self.notes[-10:]:
            lines.append(f"  • {note['text'][:60]}")
        return "\n".join(lines)

    def search(self, query: str) -> str:
        """Search notes."""
        results = [n for n in self.notes if query.lower() in n['text'].lower()]
        if not results:
            return f"No notes found matching '{query}'."
        lines = [f"Found {len(results)} note(s):"]
        for note in results[-5:]:
            lines.append(f"  • {note['text']}")
        return "\n".join(lines)

    def _save(self):
        NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
        NOTES_FILE.write_text(json.dumps(self.notes, indent=2), encoding="utf-8")


class Todo:
    """Task management."""

    def __init__(self):
        self.tasks = self._load()

    def _load(self) -> List[Dict]:
        try:
            if TODO_FILE.exists():
                return json.loads(TODO_FILE.read_text(encoding="utf-8"))
        except:
            pass
        return []

    def add(self, task: str, priority: str = "normal") -> str:
        """Add task."""
        self.tasks.append({
            "id": len(self.tasks) + 1,
            "task": task,
            "done": False,
            "priority": priority,
            "created": now_text(),
        })
        self._save()
        emoji = {"high": "🔴", "normal": "🟡", "low": "🟢"}[priority]
        return f"✓ Task added ({emoji} {priority}): {task}"

    def show(self) -> str:
        """Show tasks."""
        if not self.tasks:
            return "No tasks."
        lines = ["📋 Your Tasks:"]
        for t in sorted(self.tasks, key=lambda x: x.get("done", False)):
            status = "✓" if t.get("done") else "○"
            priority = {"high": "🔴", "normal": "🟡", "low": "🟢"}.get(t.get("priority"), "🟡")
            lines.append(f"  {status} [{t['id']}] {priority} {t['task']}")
        return "\n".join(lines)

    def done(self, task_id: int) -> str:
        """Mark task complete."""
        for t in self.tasks:
            if t["id"] == task_id:
                t["done"] = True
                self._save()
                return f"✓ Task completed: {t['task']}"
        return "Task not found."

    def _save(self):
        TODO_FILE.parent.mkdir(parents=True, exist_ok=True)
        TODO_FILE.write_text(json.dumps(self.tasks, indent=2), encoding="utf-8")


class Math:
    """Math utilities."""

    def calculate(self, expr: str) -> str:
        """Evaluate expression."""
        try:
            expr = expr.replace("^", "**")
            safe_dict = {
                "__builtins__": {},
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "pi": math.pi,
                "e": math.e,
            }
            result = eval(expr, safe_dict)
            return f"📊 {expr} = {result}"
        except Exception as e:
            return f"Calculation error: {e}"

    def stats(self, numbers_str: str) -> str:
        """Calculate statistics."""
        try:
            numbers = [float(n.strip()) for n in numbers_str.split(",")]
            if not numbers:
                return "No numbers provided."
            
            lines = [
                "📈 Statistics:",
                f"  Count: {len(numbers)}",
                f"  Sum: {sum(numbers):.2f}",
                f"  Mean: {mean(numbers):.2f}",
                f"  Median: {median(numbers):.2f}",
                f"  Min: {min(numbers):.2f}",
                f"  Max: {max(numbers):.2f}",
            ]
            if len(numbers) > 1:
                lines.append(f"  Std Dev: {pstdev(numbers):.2f}")
            return "\n".join(lines)
        except Exception as e:
            return f"Statistics error: {e}"


# =============================================================================
# MAIN JARVIS AGENT
# =============================================================================

class JarvisAIPoweredAgent:
    """Jarvis with real AI backend."""

    def __init__(self, user_name: str = "User", ai_backend: str = "ollama"):
        setup_logging()
        self.running = True
        self.user_name = user_name
        self.agent_name = "Jarvis"
        
        # Initialize AI backend
        if ai_backend == "ollama":
            self.ai = OllamaBackend()
        elif ai_backend == "llamacpp":
            self.ai = LlamaCppBackend()
        else:
            self.ai = OllamaBackend()
        
        # Features
        self.notes = Notes()
        self.todo = Todo()
        self.math = Math()
        self.conversation_history: List[Tuple[str, str]] = []

    def banner(self) -> None:
        """Display banner."""
        print(f"""
╔═════════════════════════════════════════════════════════════╗
║       Jarvis AI-Powered Agent v{APP_VERSION}       ║
║    Real AI Reasoning with Open-Source Models               ║
║    Backend: {self.ai.model_name:<35} ║
╚═════════════════════════════════════════════════════════════╝
        """)

    def dispatch(self, text: str) -> Tuple[Optional[str], Optional[bool]]:
        """Dispatch commands."""
        text_lower = text.lower()
        
        if text_lower in ["help", "?"]:
            return self.help_text(), True
        
        # Notes
        if text_lower.startswith("note "):
            return self.notes.add(text[5:].strip()), True
        if text_lower in ["notes", "show notes"]:
            return self.notes.show(), True
        if text_lower.startswith("search "):
            return self.notes.search(text[7:].strip()), True
        
        # Todos
        if text_lower.startswith("todo "):
            task = text[5:].strip()
            priority = "normal"
            if " --" in task:
                task, priority = task.rsplit(" --", 1)
            return self.todo.add(task, priority.strip()), True
        if text_lower in ["todos", "tasks"]:
            return self.todo.show(), True
        if text_lower.startswith("done "):
            try:
                task_id = int(text[5:].strip())
                return self.todo.done(task_id), True
            except:
                return "Usage: done <task_id>", False
        
        # Math
        if text_lower.startswith("calc "):
            return self.math.calculate(text[5:].strip()), True
        if text_lower.startswith("stats "):
            return self.math.stats(text[6:].strip()), True
        
        # Utilities
        if text_lower == "time":
            return f"🕐 {dt.datetime.now().strftime('%I:%M %p on %A')}", True
        if text_lower == "date":
            return f"📅 {dt.datetime.now().strftime('%B %d, %Y (%A)')}", True
        
        # Exit
        if text_lower in ["exit", "quit", "bye"]:
            return f"Goodbye {self.user_name}!", False
        
        return None, None

    def get_context(self) -> str:
        """Get conversation context for AI."""
        if len(self.conversation_history) < 2:
            return ""
        
        context_lines = []
        for user_msg, ai_msg in self.conversation_history[-3:]:
            context_lines.append(f"User: {user_msg}")
            context_lines.append(f"Assistant: {ai_msg[:100]}...")
        
        return "\n".join(context_lines)

    def handle(self, text: str) -> str:
        """Handle user input."""
        if not text.strip():
            return ""
        
        logger.info(f"USER: {text}")
        
        # Try command dispatch first
        response, success = self.dispatch(text)
        if success is not None:
            if response:
                self.conversation_history.append((text, response))
            return response
        
        # Query AI for real answers
        context = self.get_context()
        print(f"\n{self.agent_name}: [Thinking...]")  # Show it's processing
        
        if self.ai.available:
            response = self.ai.query(text, context)
        else:
            response = self.ai.query(text, context)  # Will show offline message
        
        self.conversation_history.append((text, response))
        logger.info(f"RESPONSE: {response[:100]}")
        
        return response

    def help_text(self) -> str:
        """Display help."""
        return f"""
╔═ JARVIS v15 COMMANDS ══════════════════════════════════════╗
│                                                             │
│ NOTES:    note <text> | notes | search <query>             │
│ TASKS:    todo <task> | todos | done <id>                  │
│ MATH:     calc <expr> | stats <n1,n2,n3>                   │
│ INFO:     time | date                                      │
│ AI:       Ask any question - Jarvis will use {self.ai.model_name} to answer! │
│ EXIT:     quit | bye | exit                                │
│                                                             │
│ Backend Status: {"✓ Online" if self.ai.available else "⚠ Offline"}                          │
│ {"Tip: Run 'ollama serve' to enable AI" if not self.ai.available else ""}                    │
╚═════════════════════════════════════════════════════════════╝
        """

    def run(self) -> None:
        """Main loop."""
        self.banner()
        print(f"👋 Hello {self.user_name}! I'm {self.agent_name}.")
        
        if self.ai.available:
            print(f"✓ Connected to {self.ai.model_name}")
            print("I can answer any question using AI reasoning!\n")
        else:
            print("⚠ AI backend offline. You can still use utilities.")
            print("Type 'help' for setup instructions.\n")
        
        print("Type 'help' for commands, or ask me anything!\n")
        
        while self.running:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                
                response = self.handle(user_input)
                if response:
                    print(f"\n{self.agent_name}: {response}\n")
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    self.running = False
                    
            except KeyboardInterrupt:
                print(f"\n{self.agent_name}: Goodbye!")
                self.running = False
            except Exception as e:
                print(f"\n{self.agent_name}: Error: {e}")
                logger.exception("Error in main loop")


def main() -> None:
    """Entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Jarvis AI-Powered Agent v15.0"
    )
    parser.add_argument("--name", default="User", help="Your name")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "llamacpp"],
                       help="AI backend (default: ollama)")
    args = parser.parse_args()
    
    agent = JarvisAIPoweredAgent(user_name=args.name, ai_backend=args.backend)
    try:
        agent.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        logger.exception("Fatal error")


if __name__ == "__main__":
    main()
