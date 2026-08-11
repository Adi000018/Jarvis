#!/usr/bin/env python3
"""
Jarvis AI Agent v16.0 — Professional Edition
==============================================

A local-first AI assistant that talks to a locally-running LLM through
Ollama (or, optionally, llama-cpp-python) and layers in a small set of
practical utilities: notes, tasks, and quick math.

Why "local-first"?
-------------------
No open-weight model can be embedded inside a plain Python file — a
usable model is several gigabytes, and the Ollama runtime itself is a
platform-specific compiled binary (Windows/macOS/Linux each need their
own). Nothing can make either of those "part of the .py file" and have
it still be a text file you can read and share.

What this script does instead is remove every *manual* step: run

    python jarvis_v16.py setup

and it will detect whether Ollama is installed, install it for your OS
if not, start the background server, and pull the model — all without
you touching a browser. After that one-time setup (which does need an
internet connection, the same as installing any other piece of
software), the assistant runs fully offline.

Usage
-----
    python jarvis_v16.py setup            # one-time: install + pull model
    python jarvis_v16.py run               # start the assistant
    python jarvis_v16.py run --model llama2 --name Alex
    python jarvis_v16.py status            # check backend health

Requires: Python 3.9+. No third-party packages needed for the Ollama
backend (uses the standard library only). llama-cpp-python is optional
and only required if you choose --backend llamacpp.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# CONSTANTS & PATHS
# =============================================================================

APP_VERSION = "16.0"
APP_DIR = Path.home() / ".jarvis_agent"
FILES_DIR = APP_DIR / "files"
CONFIG_FILE = APP_DIR / "config.json"
NOTES_FILE = APP_DIR / "notes.json"
TODO_FILE = APP_DIR / "todo.json"
LOG_FILE = APP_DIR / "jarvis.log"

DEFAULT_MODEL = "mistral"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_INSTALL_SCRIPT_URL = "https://ollama.com/install.sh"

logger = logging.getLogger("jarvis")


def setup_logging(verbose: bool = False) -> None:
    """Configure file logging. Safe to call more than once."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if logger.handlers:
        return
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class JarvisConfig:
    """Persisted user configuration."""
    user_name: str = "User"
    backend: str = "ollama"          # "ollama" | "llamacpp"
    ollama_model: str = DEFAULT_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    llamacpp_model_path: str = ""

    @classmethod
    def load(cls) -> "JarvisConfig":
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                return cls(**{**asdict(cls()), **data})
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Could not parse config, using defaults: %s", exc)
        return cls()

    def save(self) -> None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


# =============================================================================
# OLLAMA INSTALLER — automates the setup so nothing needs to be done by hand
# =============================================================================

class OllamaInstaller:
    """
    Detects, installs, launches, and provisions Ollama automatically.

    This does the closest thing possible to "no external download step":
    it performs the download and installation *for* the user, using the
    official channels for their OS, instead of asking them to visit a
    website. It still needs a network connection the first time, exactly
    like installing any other application — there is no way around that
    for a multi-gigabyte model file.
    """

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST):
        self.host = host.rstrip("/")

    # -- detection -----------------------------------------------------

    def is_binary_installed(self) -> bool:
        return shutil.which("ollama") is not None

    def is_server_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def has_model(self, model: str) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as resp:
                data = json.loads(resp.read().decode())
            names = {m.get("name", "").split(":")[0] for m in data.get("models", [])}
            return model in names
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False

    # -- installation ----------------------------------------------------

    def install(self) -> bool:
        """Install Ollama for the current OS. Returns True on success."""
        system = platform.system()
        print(f"→ Ollama not found. Installing for {system}...")
        try:
            if system in ("Linux", "Darwin"):
                # Official one-line installer: fetch and pipe to shell.
                cmd = f"curl -fsSL {OLLAMA_INSTALL_SCRIPT_URL} | sh"
                result = subprocess.run(cmd, shell=True, check=False)
                success = result.returncode == 0
            elif system == "Windows":
                # winget is preinstalled on modern Windows and handles the
                # download + silent install in one step.
                result = subprocess.run(
                    ["winget", "install", "-e", "--id", "Ollama.Ollama", "--silent"],
                    check=False,
                )
                success = result.returncode == 0
            else:
                print(f"⚠ Unsupported OS for auto-install: {system}")
                return False

            if success:
                print("✓ Ollama installed successfully.")
            else:
                print("✗ Automatic installation failed. See https://ollama.com/download")
            return success

        except FileNotFoundError as exc:
            print(f"✗ Required installer tool not found ({exc}). "
                  "Install manually from https://ollama.com/download")
            return False
        except Exception as exc:
            logger.error("Install error: %s", exc)
            print(f"✗ Installation error: {exc}")
            return False

    def start_server(self, wait_seconds: float = 15.0) -> bool:
        """Launch `ollama serve` in the background if it isn't already running."""
        if self.is_server_running():
            return True
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            return False

        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if self.is_server_running():
                return True
            time.sleep(0.5)
        return False

    def pull_model(self, model: str) -> bool:
        """Download a model's weights via `ollama pull`. Streams progress."""
        if self.has_model(model):
            return True
        print(f"→ Pulling model '{model}' (first run only, this can take a while)...")
        try:
            result = subprocess.run(["ollama", "pull", model], check=False)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def full_setup(self, model: str) -> bool:
        """Run the entire install → serve → pull pipeline. One command, no manual steps."""
        if not self.is_binary_installed():
            if not self.install():
                return False
        if not self.start_server():
            print("✗ Could not start the Ollama server.")
            return False
        if not self.pull_model(model):
            print(f"✗ Could not pull model '{model}'.")
            return False
        print(f"✓ Setup complete. Ollama is running with '{model}'.")
        return True


# =============================================================================
# AI BACKENDS
# =============================================================================

class AIBackend(ABC):
    """Common interface for local LLM backends."""

    model_name: str = "None"
    available: bool = False

    @abstractmethod
    def query(self, prompt: str, context: str = "") -> str:
        """Send a prompt (with optional conversation context) and return the reply."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is ready to serve requests."""


class OllamaBackend(AIBackend):
    """Talks to a local Ollama server over its HTTP API."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_OLLAMA_HOST,
                 timeout: float = 60.0, max_retries: int = 2):
        self.model = model
        self.host = host.rstrip("/")
        self.endpoint = f"{self.host}/api/generate"
        self.timeout = timeout
        self.max_retries = max_retries
        self.available = self.health_check()
        self.model_name = f"Ollama · {model}" if self.available else "Ollama (offline)"
        logger.info("Ollama backend %s (model=%s)",
                    "connected" if self.available else "unavailable", model)

    def health_check(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def query(self, prompt: str, context: str = "") -> str:
        if not self.available:
            return self._offline_message()

        full_prompt = f"{context}\n\nUser: {prompt}\n\nAssistant:" if context else prompt
        payload = json.dumps({
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.7, "top_p": 0.9, "top_k": 40},
        }).encode("utf-8")

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                req = urllib.request.Request(
                    self.endpoint, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode())
                answer = result.get("response", "").strip()
                return answer or "I couldn't generate a response for that."
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("Ollama query attempt %d failed: %s", attempt, exc)
                if attempt <= self.max_retries:
                    time.sleep(1.5 * attempt)

        logger.error("Ollama query failed after retries: %s", last_error)
        self.available = self.health_check()
        return self._offline_message()

    @staticmethod
    def _offline_message() -> str:
        return (
            "I can't reach the local AI model right now.\n"
            "Run:  python jarvis_v16.py setup\n"
            "to install Ollama and download a model automatically, "
            "then try again."
        )


class LlamaCppBackend(AIBackend):
    """Talks to a GGUF model directly via llama-cpp-python (no server needed)."""

    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self._llm = None
        self.available = False
        self.model_name = "LlamaCpp (not loaded)"

        try:
            from llama_cpp import Llama
        except ImportError:
            logger.info("llama-cpp-python not installed; run: pip install llama-cpp-python")
            return

        resolved = self._resolve_model_path(model_path)
        if not resolved:
            logger.info("No GGUF model file found for LlamaCpp backend.")
            return

        try:
            self._llm = Llama(model_path=str(resolved), n_gpu_layers=-1, verbose=False)
            self.available = True
            self.model_name = f"LlamaCpp · {resolved.name}"
            logger.info("Loaded LlamaCpp model: %s", resolved)
        except Exception as exc:
            logger.error("Failed to load LlamaCpp model: %s", exc)

    @staticmethod
    def _resolve_model_path(model_path: str) -> Optional[Path]:
        if model_path and Path(model_path).exists():
            return Path(model_path)
        for candidate in (Path.home() / "models").glob("*.gguf") if (Path.home() / "models").exists() else []:
            return candidate
        return None

    def health_check(self) -> bool:
        return self.available

    def query(self, prompt: str, context: str = "") -> str:
        if not self.available or self._llm is None:
            return "LlamaCpp backend is not loaded. Provide a valid --model-path."
        full_prompt = f"{context}\n\nUser: {prompt}\n\nAssistant:" if context else prompt
        try:
            result = self._llm(full_prompt, max_tokens=512, temperature=0.7, top_p=0.9)
            answer = result["choices"][0]["text"].strip()
            return answer or "I couldn't generate a response for that."
        except Exception as exc:
            logger.error("LlamaCpp query error: %s", exc)
            return f"Local model error: {exc}"


def build_backend(config: JarvisConfig) -> AIBackend:
    """Factory: construct the configured backend."""
    if config.backend == "llamacpp":
        return LlamaCppBackend(config.llamacpp_model_path)
    return OllamaBackend(model=config.ollama_model, host=config.ollama_host)


# =============================================================================
# FEATURES: notes, tasks, quick math
# =============================================================================

class NotesManager:
    """Simple persisted note-taking."""

    def __init__(self, path: Path = NOTES_FILE):
        self._path = path
        self._notes: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("notes.json was corrupt; starting fresh.")
        return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._notes, indent=2), encoding="utf-8")

    def add(self, text: str) -> str:
        if not text:
            return "Note text cannot be empty."
        entry = {"id": len(self._notes) + 1, "text": text,
                  "timestamp": datetime.now().isoformat(timespec="seconds")}
        self._notes.append(entry)
        self._save()
        return f"Saved note #{entry['id']}."

    def list_recent(self, limit: int = 10) -> str:
        if not self._notes:
            return "No notes yet."
        lines = [f"  [{n['id']}] {n['text']}" for n in self._notes[-limit:]]
        return "Recent notes:\n" + "\n".join(lines)

    def search(self, query: str) -> str:
        matches = [n for n in self._notes if query.lower() in n["text"].lower()]
        if not matches:
            return f"No notes match '{query}'."
        lines = [f"  [{n['id']}] {n['text']}" for n in matches]
        return f"Found {len(matches)} match(es):\n" + "\n".join(lines)


class TaskManager:
    """Simple persisted to-do list."""

    _PRIORITIES = {"high", "normal", "low"}

    def __init__(self, path: Path = TODO_FILE):
        self._path = path
        self._tasks: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("todo.json was corrupt; starting fresh.")
        return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._tasks, indent=2), encoding="utf-8")

    def add(self, task: str, priority: str = "normal") -> str:
        priority = priority.lower().strip() or "normal"
        if priority not in self._PRIORITIES:
            priority = "normal"
        entry = {"id": len(self._tasks) + 1, "task": task, "done": False,
                  "priority": priority, "created": datetime.now().isoformat(timespec="seconds")}
        self._tasks.append(entry)
        self._save()
        return f"Added task #{entry['id']} ({priority}): {task}"

    def list_all(self) -> str:
        if not self._tasks:
            return "No tasks."
        ordered = sorted(self._tasks, key=lambda t: t["done"])
        lines = []
        for t in ordered:
            mark = "x" if t["done"] else " "
            lines.append(f"  [{mark}] #{t['id']} ({t['priority']}) {t['task']}")
        return "Tasks:\n" + "\n".join(lines)

    def complete(self, task_id: int) -> str:
        for t in self._tasks:
            if t["id"] == task_id:
                t["done"] = True
                self._save()
                return f"Completed: {t['task']}"
        return f"No task with id {task_id}."


class Calculator:
    """Safe arithmetic and basic statistics — no eval() of arbitrary code."""

    _ALLOWED_NAMES: Dict[str, Any] = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "exp": math.exp,
        "pi": math.pi, "e": math.e, "abs": abs, "round": round,
    }

    def evaluate(self, expression: str) -> str:
        expr = expression.replace("^", "**")
        if re_has_disallowed_chars(expr):
            return "Expression contains characters that aren't allowed."
        try:
            result = eval(expr, {"__builtins__": {}}, self._ALLOWED_NAMES)
            return f"{expression} = {result}"
        except Exception as exc:
            return f"Couldn't evaluate that: {exc}"

    def stats(self, numbers_csv: str) -> str:
        try:
            numbers = [float(n.strip()) for n in numbers_csv.split(",") if n.strip()]
        except ValueError:
            return "Please provide a comma-separated list of numbers."
        if not numbers:
            return "No numbers provided."
        lines = [
            f"  Count : {len(numbers)}",
            f"  Sum   : {sum(numbers):.2f}",
            f"  Mean  : {mean(numbers):.2f}",
            f"  Median: {median(numbers):.2f}",
            f"  Min   : {min(numbers):.2f}",
            f"  Max   : {max(numbers):.2f}",
        ]
        if len(numbers) > 1:
            lines.append(f"  StdDev: {pstdev(numbers):.2f}")
        return "Statistics:\n" + "\n".join(lines)


import re as _re


def re_has_disallowed_chars(expr: str) -> bool:
    """Guard against anything beyond arithmetic / whitelisted function names."""
    return bool(_re.search(r"[^0-9a-zA-Z_+\-*/().,\s]", expr))


# =============================================================================
# MAIN AGENT
# =============================================================================

@dataclass
class Command:
    handler: Callable[["JarvisAgent", str], str]
    help: str


class JarvisAgent:
    """Coordinates commands, conversation history, and the AI backend."""

    def __init__(self, config: JarvisConfig):
        self.config = config
        self.agent_name = "Jarvis"
        self.ai = build_backend(config)
        self.notes = NotesManager()
        self.tasks = TaskManager()
        self.calc = Calculator()
        self.history: List[Tuple[str, str]] = []
        self.commands: Dict[str, Command] = self._build_commands()

    # -- command table ---------------------------------------------------

    def _build_commands(self) -> Dict[str, Command]:
        return {
            "help": Command(lambda self_, arg: self_.help_text(), "Show this help"),
            "note": Command(lambda self_, arg: self_.notes.add(arg), "note <text> — save a note"),
            "notes": Command(lambda self_, arg: self_.notes.list_recent(), "notes — show recent notes"),
            "search": Command(lambda self_, arg: self_.notes.search(arg), "search <query> — search notes"),
            "todo": Command(lambda self_, arg: self_._add_task(arg), "todo <task> [--priority] — add a task"),
            "todos": Command(lambda self_, arg: self_.tasks.list_all(), "todos — list tasks"),
            "done": Command(lambda self_, arg: self_._complete_task(arg), "done <id> — mark a task complete"),
            "calc": Command(lambda self_, arg: self_.calc.evaluate(arg), "calc <expr> — evaluate an expression"),
            "stats": Command(lambda self_, arg: self_.calc.stats(arg), "stats <n1,n2,...> — quick statistics"),
            "time": Command(lambda self_, arg: datetime.now().strftime("%I:%M %p on %A"), "time — current time"),
            "date": Command(lambda self_, arg: datetime.now().strftime("%B %d, %Y (%A)"), "date — today's date"),
            "status": Command(lambda self_, arg: self_.status_text(), "status — backend health"),
        }

    def _add_task(self, arg: str) -> str:
        task, _, priority = arg.partition("--")
        return self.tasks.add(task.strip(), priority.strip() or "normal")

    def _complete_task(self, arg: str) -> str:
        try:
            return self.tasks.complete(int(arg.strip()))
        except ValueError:
            return "Usage: done <task_id>"

    # -- dispatch ---------------------------------------------------------

    def dispatch(self, text: str) -> Optional[str]:
        """Return a command's output, or None if `text` isn't a recognized command."""
        stripped = text.strip()
        if not stripped:
            return ""
        head, _, rest = stripped.partition(" ")
        cmd = self.commands.get(head.lower())
        if cmd is None:
            return None
        return cmd.handler(self, rest.strip())

    def build_context(self, turns: int = 3) -> str:
        if len(self.history) < 2:
            return ""
        lines = []
        for user_msg, ai_msg in self.history[-turns:]:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {ai_msg[:200]}")
        return "\n".join(lines)

    def handle(self, text: str) -> str:
        if not text.strip():
            return ""
        logger.info("USER: %s", text)

        command_result = self.dispatch(text)
        if command_result is not None:
            if command_result:
                self.history.append((text, command_result))
            return command_result

        response = self.ai.query(text, self.build_context())
        self.history.append((text, response))
        logger.info("REPLY: %s", response[:200])
        return response

    # -- presentation -------------------------------------------------------

    def banner(self) -> str:
        status = "online" if self.ai.available else "offline"
        return (
            f"Jarvis v{APP_VERSION} — backend: {self.ai.model_name} ({status})"
        )

    def help_text(self) -> str:
        lines = ["Commands:"]
        for name, cmd in self.commands.items():
            lines.append(f"  {name:<8} {cmd.help}")
        lines.append("  exit     quit | bye | exit — leave")
        lines.append("Anything else is sent to the AI model as a free-form question.")
        return "\n".join(lines)

    def status_text(self) -> str:
        ok = self.ai.health_check()
        self.ai.available = ok
        return (
            f"Backend : {self.ai.model_name}\n"
            f"Status  : {'reachable' if ok else 'unreachable'}\n"
            f"Fix     : run 'python jarvis_v16.py setup' if unreachable"
        )

    def run(self) -> None:
        print(self.banner())
        print(f"Hi {self.config.user_name}, I'm {self.agent_name}. Type 'help' for commands.\n")
        while True:
            try:
                user_input = input("you> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                return

            if user_input.lower() in ("exit", "quit", "bye"):
                print("Goodbye.")
                return

            reply = self.handle(user_input)
            if reply:
                print(f"{self.agent_name}> {reply}\n")


# =============================================================================
# CLI
# =============================================================================

def cmd_setup(args: argparse.Namespace) -> None:
    config = JarvisConfig.load()
    if args.model:
        config.ollama_model = args.model
    config.save()

    installer = OllamaInstaller(host=config.ollama_host)
    ok = installer.full_setup(config.ollama_model)
    sys.exit(0 if ok else 1)


def cmd_status(args: argparse.Namespace) -> None:
    config = JarvisConfig.load()
    backend = build_backend(config)
    print(f"Backend: {backend.model_name}")
    print(f"Reachable: {backend.health_check()}")


def cmd_run(args: argparse.Namespace) -> None:
    config = JarvisConfig.load()
    if args.name:
        config.user_name = args.name
    if args.backend:
        config.backend = args.backend
    if args.model:
        config.ollama_model = args.model
    if args.model_path:
        config.llamacpp_model_path = args.model_path
    config.save()

    agent = JarvisAgent(config)
    agent.run()


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        prog="jarvis_v16.py",
        description="Jarvis — a local-first AI assistant powered by Ollama.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Install Ollama and pull a model automatically")
    p_setup.add_argument("--model", default=None, help=f"Model to install (default: {DEFAULT_MODEL})")
    p_setup.set_defaults(func=cmd_setup)

    p_status = sub.add_parser("status", help="Check backend health")
    p_status.set_defaults(func=cmd_status)

    p_run = sub.add_parser("run", help="Start the assistant")
    p_run.add_argument("--name", default=None, help="Your name")
    p_run.add_argument("--backend", choices=["ollama", "llamacpp"], default=None)
    p_run.add_argument("--model", default=None, help="Ollama model name")
    p_run.add_argument("--model-path", default=None, dest="model_path", help="Path to a GGUF file (llamacpp backend)")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        logger.exception("Fatal error")
        print(f"Fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
