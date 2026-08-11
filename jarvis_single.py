#!/usr/bin/env python3
"""
Jarvis AI Assistant - Single File Version
==========================================

A voice-driven personal assistant with Groq (online) + Ollama (offline fallback).
All code in one file for easy use in Python compilers/IDEs.

Usage:
    python jarvis_single.py              # Start Jarvis in interactive mode
    python jarvis_single.py --mode offline   # Force offline mode
    python jarvis_single.py --text "Hello"  # Text input mode (no voice)
    python jarvis_single.py --check          # Check environment

Requirements:
    pip install groq python-dotenv pyyaml pyttsx3 faster-whisper speechrecognition ollama duckduckgo-search wikipedia
"""

import os
import sys
import json
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
import wave
import io
import logging
import logging.handlers
import argparse
from collections import deque
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# Try to import optional dependencies
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
    print("Warning: python-dotenv not installed, API key must be in environment")

try:
    import yaml
except ImportError:
    yaml = None
    print("Warning: PyYAML not installed, using default config")

try:
    import speech_recognition as sr
except ImportError:
    sr = None
    print("Warning: speech_recognition not installed")

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    print("Warning: pyttsx3 not installed")

try:
    from groq import Groq
except ImportError:
    Groq = None
    print("Warning: groq not installed")

try:
    import ollama
except ImportError:
    ollama = None
    print("Warning: ollama not installed")


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "default_mode": "online",
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "fallback_model": "llama-3.1-8b-instant",
        "max_retries": 2,
        "timeout_seconds": 30,
    },
    "ollama": {
        "host": "http://localhost:11434",
        "default_model": "qwen2.5:3b",
        "timeout_seconds": 60,
    },
    "stt": {
        "engine": "faster-whisper",
        "whisper_model": "base",
        "language": "en",
    },
    "tts": {
        "engine": "pyttsx3",
        "voice_rate": 175,
        "voice_volume": 1.0,
    },
    "conversation": {
        "max_history_turns": 10,
        "system_prompt": "You are Jarvis, a helpful personal assistant. Be concise, friendly, and helpful. If the user asks something you don't know, say so honestly. Always respond in a natural, conversational tone.",
    },
    "logging": {
        "level": "INFO",
        "file": "jarvis.log",
        "max_bytes": 5242880,
        "backup_count": 3,
    },
    "transcript": {
        "enabled": True,
        "file": "conversation_transcript.json",
    },
}


# ============================================================================
# LOGGING
# ============================================================================

class JarvisLogger:
    """Simple rotating file logger."""
    
    def __init__(self, name: str = "jarvis", config: dict = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger(name)
        
        if not self.logger.handlers:
            log_config = self.config.get("logging", {})
            level_str = log_config.get("level", "INFO").upper()
            level = getattr(logging, level_str, logging.INFO)
            
            self.logger.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            # Console handler
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(level)
            console.setFormatter(formatter)
            self.logger.addHandler(console)
            
            # File handler (rotating)
            log_file = log_config.get("file", "jarvis.log")
            max_bytes = log_config.get("max_bytes", 5 * 1024 * 1024)
            backup_count = log_config.get("backup_count", 3)
            
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg): self.logger.debug(msg)
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def critical(self, msg): self.logger.critical(msg)


# ============================================================================
# CONNECTIVITY
# ============================================================================

def check_internet_connectivity(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> bool:
    """Check if internet connection is available."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except (socket.timeout, socket.error, OSError):
        return False


def check_network_status() -> dict:
    """Get comprehensive network status."""
    dns_ok = check_internet_connectivity()
    http_ok = False
    status_code = 0
    
    try:
        req = Request("https://www.google.com", headers={'User-Agent': 'Jarvis/1.0'})
        with urlopen(req, timeout=5.0) as response:
            http_ok = True
            status_code = response.status
    except (HTTPError, URLError, socket.timeout, OSError):
        pass
    
    return {
        "dns_connected": dns_ok,
        "http_connected": http_ok,
        "status_code": status_code,
        "online": dns_ok and http_ok,
    }


# ============================================================================
# MEMORY
# ============================================================================

class ConversationMemory:
    """Manages conversation history for context awareness."""
    
    def __init__(self, max_turns: int = 10, transcript_file: Optional[str] = None, logger: JarvisLogger = None):
        self.max_turns = max_turns
        self.transcript_file = transcript_file
        self._history = deque(maxlen=max_turns)
        self._turn_count = 0
        self.logger = logger or JarvisLogger("memory")
        
        if self.transcript_file:
            self._ensure_transcript_file()
    
    def _ensure_transcript_file(self):
        if not os.path.exists(self.transcript_file):
            try:
                with open(self.transcript_file, 'w') as f:
                    json.dump({"conversation": [], "created_at": datetime.now().isoformat()}, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Could not create transcript file: {e}")
    
    def add_turn(self, user_input: str, assistant_response: str):
        turn = {
            "turn": self._turn_count + 1,
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        }
        self._history.append(turn)
        
        response_turn = {
            "turn": self._turn_count + 1,
            "role": "assistant",
            "content": assistant_response,
            "timestamp": datetime.now().isoformat()
        }
        self._history.append(response_turn)
        
        self._turn_count += 1
        
        if self.transcript_file:
            self._save_transcript()
        
        self.logger.debug(f"Added turn {self._turn_count}: {user_input[:50]}...")
    
    def _save_transcript(self):
        try:
            transcript_data = {
                "conversation": list(self._history),
                "total_turns": self._turn_count,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.transcript_file, 'w') as f:
                json.dump(transcript_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save transcript: {e}")
    
    def get_history(self) -> List[Dict]:
        return list(self._history)
    
    def get_context_for_llm(self) -> List[Dict]:
        messages = []
        for turn in self._history:
            role = "user" if turn["role"] == "user" else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        return messages
    
    def clear(self):
        self._history.clear()
        self._turn_count = 0
        if self.transcript_file:
            self._ensure_transcript_file()
        self.logger.info("Conversation memory cleared")
    
    @property
    def turn_count(self) -> int:
        return self._turn_count
    
    @property
    def is_empty(self) -> bool:
        return len(self._history) == 0


# ============================================================================
# INTENT HANDLER
# ============================================================================

class IntentHandler:
    """Handles rule-based intents without needing the LLM."""
    
    def __init__(self, logger: JarvisLogger = None):
        self.logger = logger or JarvisLogger("intents")
        self._active_timers: Dict[str, threading.Thread] = {}
        self._stop_timers: Dict[str, threading.Event] = {}
        self._timer_completed = False
        self._timer_message = ""
    
    def check_intent(self, query: str) -> Optional[Dict[str, Any]]:
        query_lower = query.lower().strip()
        
        # Time
        if any(word in query_lower for word in ["time", "what time", "current time"]):
            return self._handle_time()
        
        # Date
        if any(word in query_lower for word in ["date", "what date", "today", "day is it"]):
            return self._handle_date()
        
        # Calculator
        if any(word in query_lower for word in ["calculate", "calculator", "what is", "compute"]):
            math_result = self._extract_math(query)
            if math_result is not None:
                return self._response(f"The answer is {math_result}")
        
        # Open websites
        if any(word in query_lower for word in ["open", "go to", "visit"]) and self._extract_url(query):
            return self._handle_open_website(self._extract_url(query))
        
        # Open apps
        if any(word in query_lower for word in ["open", "launch", "start"]):
            app = self._extract_app(query)
            if app:
                return self._handle_open_app(app)
        
        # Timer
        if any(word in query_lower for word in ["timer", "set timer", "countdown"]):
            return self._handle_timer(query)
        
        # Stop timer
        if any(word in query_lower for word in ["stop timer", "cancel timer"]):
            return self._handle_stop_timer()
        
        # Volume
        if any(word in query_lower for word in ["volume up", "increase volume", "louder"]):
            return self._handle_volume("up")
        if any(word in query_lower for word in ["volume down", "decrease volume", "quieter"]):
            return self._handle_volume("down")
        if any(word in query_lower for word in ["mute", "unmute", "volume off"]):
            return self._handle_volume("mute")
        
        # Play music
        if any(word in query_lower for word in ["play music", "play song", "start music"]):
            return self._handle_play_music()
        
        # Help
        if any(word in query_lower for word in ["help", "what can you do", "capabilities"]):
            return self._handle_help()
        
        return None
    
    def _response(self, text: str, intent_name: str = "rule_based") -> Dict[str, Any]:
        return {"success": True, "response": text, "intent": intent_name, "use_llm": False}
    
    def _handle_time(self) -> Dict[str, Any]:
        now = datetime.now()
        time_str = now.strftime("%I:%M %p").lstrip("0")
        return self._response(f"The current time is {time_str}")
    
    def _handle_date(self) -> Dict[str, Any]:
        today = datetime.date.today()
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        date_str = today.strftime("%B %d, %Y")
        day_name = day_names[today.weekday()]
        return self._response(f"Today is {day_name}, {date_str}")
    
    def _extract_math(self, query: str) -> Optional[float]:
        math_query = re.sub(r'\b(what is|calculate|compute|equals|equal to)\b', '', query.lower())
        math_query = re.sub(r'[\?\.,\!\s]', '', math_query)
        
        patterns = [r'(\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?)*)']
        
        for pattern in patterns:
            match = re.search(pattern, math_query)
            if match:
                expr = match.group(1).replace('^', '**')
                try:
                    result = eval(expr, {"__builtins__": {}}, {})
                    return round(result, 4)
                except:
                    return None
        return None
    
    def _extract_url(self, query: str) -> Optional[str]:
        url_pattern = r'https?://[^\s]+'
        match = re.search(url_pattern, query)
        if match:
            return match.group(0)
        
        websites = {
            "google": "https://www.google.com", "youtube": "https://www.youtube.com",
            "gmail": "https://mail.google.com", "github": "https://github.com",
            "twitter": "https://twitter.com", "facebook": "https://facebook.com",
            "instagram": "https://instagram.com", "reddit": "https://reddit.com",
            "amazon": "https://amazon.com", "wikipedia": "https://wikipedia.org",
            "weather": "https://weather.com", "news": "https://news.google.com",
        }
        
        query_words = query.lower().split()
        for word in query_words:
            if word in websites:
                return websites[word]
        return None
    
    def _extract_app(self, query: str) -> Optional[str]:
        apps = {
            "calculator": "calc.exe", "notepad": "notepad.exe",
            "paint": "mspaint.exe", "cmd": "cmd.exe", "terminal": "cmd.exe",
            "powershell": "powershell.exe", "control panel": "control.exe",
        }
        
        query_lower = query.lower()
        for app_name, exe_name in apps.items():
            if app_name in query_lower:
                return exe_name
        return None
    
    def _handle_open_website(self, url: str) -> Dict[str, Any]:
        try:
            import webbrowser
            webbrowser.open(url)
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            return self._response(f"Opening {domain} in your browser")
        except Exception as e:
            self.logger.error(f"Failed to open website: {e}")
            return self._response("I couldn't open that website.")
    
    def _handle_open_app(self, app: str) -> Dict[str, Any]:
        try:
            if sys.platform == "win32":
                subprocess.Popen(app)
                app_name = app.replace(".exe", "").title()
                return self._response(f"Opening {app_name}")
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-a", app.replace(".exe", "")])
                return self._response(f"Opening {app}")
            else:
                return self._response("App opening is only supported on Windows for now")
        except Exception as e:
            self.logger.error(f"Failed to open app: {e}")
            return self._response("I couldn't open that application.")
    
    def _handle_timer(self, query: str) -> Dict[str, Any]:
        minutes = 0
        seconds = 0
        
        min_match = re.search(r'(\d+)\s*minute', query.lower())
        sec_match = re.search(r'(\d+)\s*second', query.lower())
        
        if min_match:
            minutes = int(min_match.group(1))
        if sec_match:
            seconds = int(sec_match.group(1))
        
        if minutes == 0 and seconds == 0:
            minutes = 1
        
        total_seconds = minutes * 60 + seconds
        timer_id = f"timer_{time.time()}"
        stop_event = threading.Event()
        self._stop_timers[timer_id] = stop_event
        
        def timer_thread():
            elapsed = 0
            while elapsed < total_seconds and not stop_event.is_set():
                time.sleep(1)
                elapsed += 1
            if not stop_event.is_set():
                self._timer_completed = True
                self._timer_message = f"Timer completed! {minutes} minute(s) and {seconds} second(s) have passed."
        
        self._timer_completed = False
        self._timer_message = ""
        self._active_timers[timer_id] = threading.Thread(target=timer_thread)
        self._active_timers[timer_id].start()
        
        time_str = f"{minutes} minute(s) and {seconds} second(s)" if minutes > 0 and seconds > 0 else \
                   f"{minutes} minute(s)" if minutes > 0 else f"{seconds} second(s)"
        
        return self._response(f"Starting a timer for {time_str}. I'll let you know when it's done.")
    
    def _handle_stop_timer(self) -> Dict[str, Any]:
        stopped_any = False
        for timer_id, stop_event in self._stop_timers.items():
            stop_event.set()
            stopped_any = True
        
        if stopped_any:
            self._timer_completed = True
            self._timer_message = "Timer stopped."
            return self._response("Timer stopped.")
        return self._response("There's no active timer to stop.")
    
    def _handle_volume(self, action: str) -> Dict[str, Any]:
        volume_responses = {
            "up": "I'd turn the volume up, but I need additional setup for that on this system.",
            "down": "I'd turn the volume down, but I need additional setup for that on this system.",
            "mute": "I'd mute the audio, but I need additional setup for that on this system.",
            "query": "Volume query requires additional setup on this system.",
        }
        return self._response(volume_responses.get(action, "Volume adjustment needs setup."))
    
    def _handle_play_music(self) -> Dict[str, Any]:
        import webbrowser
        music_services = ["https://open.spotify.com", "https://music.youtube.com"]
        
        for url in music_services:
            try:
                webbrowser.open(url)
                return self._response(f"Opening {url.replace('https://', '')} in your browser")
            except:
                continue
        return self._response("I can't play music directly, but I can open a music service in your browser.")
    
    def _handle_help(self) -> Dict[str, Any]:
        help_text = (
            "Here's what I can do:\n\n"
            "• Tell you the time and date\n"
            "• Do basic calculations\n"
            "• Open websites and applications\n"
            "• Set timers\n"
            "• And through the AI brain: answer questions, have conversations, and more!\n\n"
            "Try asking me anything!"
        )
        return self._response(help_text)


# ============================================================================
# SPEECH-TO-TEXT
# ============================================================================

class SpeechToText:
    """Handles speech recognition from microphone input."""
    
    def __init__(self, engine: str = "faster-whisper", model_size: str = "base", language: str = "en", logger: JarvisLogger = None):
        self.engine = engine
        self.model_size = model_size
        self.language = language
        self.logger = logger or JarvisLogger("stt")
        self.recognizer = None
        self.whisper_model = None
        
        if sr is None:
            self.logger.error("speech_recognition not installed")
            return
        
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        
        if engine == "faster-whisper":
            self._init_whisper()
    
    def _init_whisper(self):
        try:
            from faster_whisper import WhisperModel
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            compute_type = "int8" if os.name == "nt" else "int8"
            self.whisper_model = WhisperModel(self.model_size, device="cpu", compute_type=compute_type)
            self.logger.info("Whisper model loaded successfully")
        except ImportError:
            self.logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
    
    def transcribe_from_mic(self, timeout: float = 10.0, phrase_time_limit: float = 30.0) -> Optional[str]:
        if sr is None:
            return None
        
        with sr.Microphone() as source:
            self.logger.info("Listening for speech...")
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                self.logger.debug("Audio captured, processing...")
            except sr.WaitTimeoutError:
                self.logger.warning("No speech detected within timeout")
                return None
            except Exception as e:
                self.logger.error(f"Error capturing audio: {e}")
                return None
        
        return self.transcribe_audio(audio)
    
    def transcribe_audio(self, audio_data: sr.AudioData) -> Optional[str]:
        if self.engine == "faster-whisper" and self.whisper_model:
            return self._transcribe_whisper(audio_data)
        return self._transcribe_google(audio_data)
    
    def _transcribe_whisper(self, audio_data: sr.AudioData) -> Optional[str]:
        try:
            import numpy as np
            
            wav_buffer = io.BytesIO()
            wf = wave.open(wav_buffer, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(audio_data.sample_rate)
            
            raw_data = audio_data.get_raw_data()
            wf.writeframes(raw_data)
            wf.close()
            
            wav_buffer.seek(0)
            
            segments, info = self.whisper_model.transcribe(wav_buffer, language=self.language, beam_size=5, vad_filter=True)
            
            text = " ".join([segment.text for segment in segments])
            
            if text.strip():
                self.logger.debug(f"Whisper transcription: {text[:50]}...")
                return text.strip()
            return None
        except Exception as e:
            self.logger.error(f"Whisper transcription error: {e}")
            return None
    
    def _transcribe_google(self, audio_data: sr.AudioData) -> Optional[str]:
        try:
            self.logger.debug("Using Google Speech Recognition (online)")
            text = self.recognizer.recognize_google(audio_data, language=self.language)
            if text:
                self.logger.debug(f"Google transcription: {text[:50]}...")
                return text
            return None
        except sr.UnknownValueError:
            self.logger.warning("Google could not understand audio")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Google API error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Google transcription error: {e}")
            return None
    
    def adjust_for_ambient_noise(self, duration: float = 1.0):
        if sr is None:
            return
        with sr.Microphone() as source:
            self.logger.info(f"Adjusting for ambient noise ({duration}s)...")
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            self.logger.info("Ambient noise adjustment complete")


# ============================================================================
# TEXT-TO-SPEECH
# ============================================================================

class TextToSpeech:
    """Handles text-to-speech conversion and playback."""
    
    def __init__(self, engine: str = "pyttsx3", voice_rate: int = 175, voice_volume: float = 1.0, logger: JarvisLogger = None):
        self.engine_name = engine
        self.voice_rate = voice_rate
        self.voice_volume = voice_volume
        self.logger = logger or JarvisLogger("tts")
        self._engine = None
        self._voices = []
        
        if engine == "pyttsx3":
            self._init_pyttsx3()
    
    def _init_pyttsx3(self):
        if pyttsx3 is None:
            self.logger.error("pyttsx3 not installed. Install with: pip install pyttsx3")
            return
        
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.voice_rate)
            self._engine.setProperty('volume', self.voice_volume)
            
            self._voices = self._engine.getProperty('voices')
            self._select_default_voice()
            
            self.logger.info("pyttsx3 engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize pyttsx3: {e}")
            self._engine = None
    
    def _select_default_voice(self):
        if not self._voices:
            return
        
        preferred_voices = [
            lambda v: "female" in v.name.lower(),
            lambda v: "zira" in v.name.lower() or "samantha" in v.name.lower(),
            lambda v: "en" in v.id.lower() or "english" in v.name.lower(),
        ]
        
        for predicate in preferred_voices:
            for voice in self._voices:
                if predicate(voice):
                    try:
                        self._engine.setProperty('voice', voice.id)
                        self.logger.debug(f"Selected voice: {voice.name}")
                        return
                    except:
                        continue
        
        try:
            self._engine.setProperty('voice', self._voices[0].id)
            self.logger.debug(f"Using default voice: {self._voices[0].name}")
        except:
            pass
    
    def speak(self, text: str, blocking: bool = True) -> bool:
        if not text or not text.strip():
            self.logger.warning("Empty text, nothing to speak")
            return False
        
        if not self._engine:
            self.logger.warning("TTS engine not available")
            return False
        
        try:
            self.logger.debug(f"Speaking: {text[:50]}...")
            
            if self.engine_name == "pyttsx3":
                self._engine.say(text)
                if blocking:
                    self._engine.runAndWait()
                else:
                    self._engine.startLoop(False)
                return True
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            return False
        
        return False
    
    def save_to_file(self, text: str, filepath: str) -> bool:
        if not self._engine:
            return False
        
        try:
            if self.engine_name == "pyttsx3":
                self._engine.save_to_file(text, filepath)
                self._engine.runAndWait()
                self.logger.info(f"Saved speech to: {filepath}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to save speech: {e}")
            return False
        
        return False
    
    def is_available(self) -> bool:
        return self._engine is not None


# ============================================================================
# BRAIN IMPLEMENTATIONS
# ============================================================================

@dataclass
class BrainResponse:
    text: str
    source: str
    used_fallback: bool = False
    error: Optional[str] = None


class BaseBrain(ABC):
    """Abstract base class for all brain implementations."""
    
    def __init__(self, model_name: str, system_prompt: str = "", logger: JarvisLogger = None):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.logger = logger or JarvisLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.7) -> BrainResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class GroqBrain(BaseBrain):
    """Online brain using Groq API."""
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", 
                 system_prompt: str = "",
                 max_retries: int = 2,
                 timeout: int = 30,
                 logger: JarvisLogger = None):
        super().__init__(model_name, system_prompt, logger)
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            if Groq is None:
                self.logger.error("groq package not installed")
                return None
            
            api_key = os.environ.get("GROQ_API_KEY")
            
            if not api_key:
                self.logger.warning("GROQ_API_KEY not set in environment")
                return None
            
            try:
                self._client = Groq(api_key=api_key)
            except Exception as e:
                self.logger.error(f"Failed to create Groq client: {e}")
                return None
        
        return self._client
    
    def is_available(self) -> bool:
        client = self._get_client()
        return client is not None
    
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.7) -> BrainResponse:
        client = self._get_client()
        if not client:
            return BrainResponse(
                text="I don't have access to the Groq API. Please set your GROQ_API_KEY environment variable.",
                source="groq",
                error="Missing API key"
            )
        
        full_messages = []
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})
        full_messages.extend(messages)
        
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self.timeout
                )
                
                text = response.choices[0].message.content
                self.logger.debug(f"Groq response received (attempt {attempt + 1})")
                return BrainResponse(text=text, source="groq")
                
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"Groq API error (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                
                if "401" in error_msg or "Unauthorized" in error_msg:
                    return BrainResponse(
                        text="I'm having trouble with my API key. Please check your GROQ_API_KEY setting.",
                        source="groq",
                        error="Invalid API key"
                    )
                elif "429" in error_msg or "rate limit" in error_msg.lower():
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    return BrainResponse(
                        text="I'm getting rate limited. Let me try offline mode instead.",
                        source="groq",
                        error="Rate limited",
                        used_fallback=True
                    )
                elif "timeout" in error_msg.lower() or "network" in error_msg.lower():
                    if attempt < self.max_retries:
                        time.sleep(1)
                        continue
                    return BrainResponse(
                        text="I'm having network trouble. Let me try offline mode.",
                        source="groq",
                        error="Network error",
                        used_fallback=True
                    )
                else:
                    if attempt < self.max_retries:
                        time.sleep(1)
                        continue
                    return BrainResponse(
                        text="I encountered an error. Let me try offline mode.",
                        source="groq",
                        error=error_msg,
                        used_fallback=True
                    )
        
        return BrainResponse(
            text="I'm sorry, I couldn't get a response from the online service.",
            source="groq",
            error="Max retries exceeded"
        )


class OfflineBrain(BaseBrain):
    """Offline brain using local Ollama instance."""
    
    def __init__(self, model_name: str = "qwen2.5:3b",
                 system_prompt: str = "",
                 host: str = "http://localhost:11434",
                 timeout: int = 60,
                 logger: JarvisLogger = None):
        super().__init__(model_name, system_prompt, logger)
        self.host = host
        self.timeout = timeout
    
    def _get_client(self):
        if ollama is None:
            self.logger.error("ollama package not installed. Install with: pip install ollama")
            return None
        return ollama
    
    def is_available(self) -> bool:
        client = self._get_client()
        if not client:
            return False
        
        try:
            client.list()
            return True
        except Exception as e:
            self.logger.debug(f"Ollama not available: {e}")
            return False
    
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.7) -> BrainResponse:
        client = self._get_client()
        if not client:
            return BrainResponse(
                text="Offline mode is not available. Please install Ollama and pull a model.",
                source="ollama",
                error="Ollama not installed"
            )
        
        ollama_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        
        try:
            response = client.chat(
                model=self.model_name,
                messages=ollama_messages,
                options={"num_predict": max_tokens, "temperature": temperature},
                stream=False
            )
            
            text = response.get("message", {}).get("content", "")
            if not text:
                text = response.get("response", "")
            
            self.logger.debug(f"Ollama response received")
            return BrainResponse(text=text, source="ollama")
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Ollama error: {error_msg}")
            
            if "not found" in error_msg.lower() or "model" in error_msg.lower():
                return BrainResponse(
                    text=f"The model '{self.model_name}' is not available. Please pull it with: ollama pull {self.model_name}",
                    source="ollama",
                    error="Model not found"
                )
            
            return BrainResponse(
                text="I'm having trouble with the offline mode. Please make sure Ollama is running.",
                source="ollama",
                error=error_msg
            )


# ============================================================================
# MODE ROUTER
# ============================================================================

class ModeRouter:
    """Routes queries to the appropriate brain with fallback chain."""
    
    def __init__(self, config: dict, logger: JarvisLogger = None):
        self.config = config
        self.logger = logger or JarvisLogger("router")
        
        groq_config = config.get("groq", {})
        ollama_config = config.get("ollama", {})
        conv_config = config.get("conversation", {})
        
        self.groq_brain = GroqBrain(
            model_name=groq_config.get("default_model", "llama-3.3-70b-versatile"),
            system_prompt=conv_config.get("system_prompt", ""),
            max_retries=groq_config.get("max_retries", 2),
            timeout=groq_config.get("timeout_seconds", 30),
            logger=self.logger
        )
        
        self.ollama_brain = OfflineBrain(
            model_name=ollama_config.get("default_model", "qwen2.5:3b"),
            system_prompt=conv_config.get("system_prompt", ""),
            host=ollama_config.get("host", "http://localhost:11434"),
            timeout=ollama_config.get("timeout_seconds", 60),
            logger=self.logger
        )
        
        self._mode = config.get("default_mode", "online")
        self._force_offline = False
        self._force_online = False
    
    @property
    def mode(self) -> str:
        if self._force_offline:
            return "offline"
        if self._force_online:
            return "online"
        return self._mode
    
    def set_mode(self, mode: str):
        mode = mode.lower()
        if mode == "online":
            self._force_online = True
            self._force_offline = False
            self.logger.info("Mode set to ONLINE (manual override)")
        elif mode == "offline":
            self._force_offline = True
            self._force_online = False
            self.logger.info("Mode set to OFFLINE (manual override)")
        else:
            self.logger.warning(f"Unknown mode: {mode}")
    
    def reset_mode_override(self):
        self._force_offline = False
        self._force_online = False
        self.logger.info("Mode override reset")
    
    def handle(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> BrainResponse:
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})
        
        primary_brain = self.groq_brain if self.mode == "online" else self.ollama_brain
        fallback_brain = self.ollama_brain if self.mode == "online" else self.groq_brain
        
        if not primary_brain.is_available():
            self.logger.warning(f"Primary brain ({self.mode}) not available, trying fallback")
            fallback_result = self._try_brain(fallback_brain, messages)
            if fallback_result and not fallback_result.error:
                return fallback_result
            return fallback_result if fallback_result else BrainResponse(
                text="I'm having trouble processing your request. Both online and offline modes are unavailable.",
                source="unknown",
                error="No brains available"
            )
        
        result = self._try_brain(primary_brain, messages)
        
        if result.error or result.text.startswith("I'm having trouble") or result.text.startswith("I'm sorry"):
            if not self._force_online and not self._force_offline:
                self.logger.info(f"Falling back from {self.mode} to alternative mode")
                fallback_result = self._try_brain(fallback_brain, messages)
                if fallback_result and not fallback_result.error:
                    if fallback_result.source == "ollama":
                        fallback_result.text = f"(Switching to offline mode) {fallback_result.text}"
                    return fallback_result
        
        return result if result else BrainResponse(
            text="I couldn't generate a response. Please try again.",
            source=self.mode,
            error="No response generated"
        )
    
    def _try_brain(self, brain: BaseBrain, messages: List[Dict[str, str]]) -> Optional[BrainResponse]:
        if not brain.is_available():
            self.logger.debug(f"Brain {brain.__class__.__name__} not available")
            return None
        
        try:
            return brain.generate(messages)
        except Exception as e:
            self.logger.error(f"Error from {brain.__class__.__name__}: {e}")
            return BrainResponse(
                text="I encountered an error processing your request.",
                source=brain.__class__.__name__.replace("Brain", "").lower(),
                error=str(e)
            )
    
    def check_mode_availability(self) -> Dict[str, bool]:
        internet_available = check_internet_connectivity()
        return {
            "online": self.groq_brain.is_available() and internet_available,
            "offline": self.ollama_brain.is_available(),
            "internet": internet_available
        }


# ============================================================================
# MAIN JARVIS CLASS
# ============================================================================

class Jarvis:
    """Main Jarvis assistant class."""
    
    def __init__(self, mode: str = None, text_mode: bool = False, config: dict = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = JarvisLogger("jarvis", self.config)
        self.text_mode = text_mode
        self.running = True
        
        # Initialize components
        self.router = ModeRouter(self.config, self.logger)
        self.stt = SpeechToText(
            engine=self.config.get("stt", {}).get("engine", "faster-whisper"),
            model_size=self.config.get("stt", {}).get("whisper_model", "base"),
            language=self.config.get("stt", {}).get("language", "en"),
            logger=self.logger
        )
        self.tts = TextToSpeech(
            engine=self.config.get("tts", {}).get("engine", "pyttsx3"),
            voice_rate=self.config.get("tts", {}).get("voice_rate", 175),
            voice_volume=self.config.get("tts", {}).get("voice_volume", 1.0),
            logger=self.logger
        )
        self.intent_handler = IntentHandler(self.logger)
        
        transcript_config = self.config.get("transcript", {})
        transcript_file = transcript_config.get("file") if transcript_config.get("enabled", True) else None
        max_turns = self.config.get("conversation", {}).get("max_history_turns", 10)
        self.memory = ConversationMemory(max_turns, transcript_file, self.logger)
        
        if mode:
            self.router.set_mode(mode)
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("=" * 50)
        self.logger.info("Jarvis AI Assistant Starting...")
        self.logger.info(f"Mode: {self.router.mode.upper()}")
        self.logger.info(f"Text mode: {text_mode}")
        self.logger.info(f"Groq available: {self.router.groq_brain.is_available()}")
        self.logger.info(f"Ollama available: {self.router.ollama_brain.is_available()}")
        self.logger.info("=" * 50)
    
    def _signal_handler(self, signum, frame):
        self.logger.info("Shutdown signal received...")
        self.running = False
    
    def process_input(self, query: str) -> str:
        query = query.strip()
        if not query:
            return ""
        
        mode_change = self._check_mode_commands(query)
        if mode_change:
            return mode_change
        
        if query.lower() in ["exit", "quit", "bye", "goodbye", "stop"]:
            self.logger.info("Exit command received")
            self.running = False
            return "Goodbye! Have a great day."
        
        intent_result = self.intent_handler.check_intent(query)
        if intent_result:
            self.memory.add_turn(query, intent_result["response"])
            return intent_result["response"]
        
        timer_message = self._check_timer()
        if timer_message:
            self.memory.add_turn(query, timer_message)
            return timer_message
        
        self.logger.info(f"Processing query with {self.router.mode} mode...")
        history = self.memory.get_context_for_llm()
        response = self.router.handle(query, history)
        
        if response.error and response.text.startswith("I'm having trouble"):
            self.memory.add_turn(query, response.text)
            return response.text
        
        self.memory.add_turn(query, response.text)
        return response.text
    
    def _check_mode_commands(self, query: str) -> str:
        query_lower = query.lower().strip()
        
        if query_lower in ["go offline", "switch to offline", "use offline mode", "offline mode"]:
            self.router.set_mode("offline")
            return "Switching to offline mode. I'll use the local Ollama model from now on."
        
        elif query_lower in ["go online", "switch to online", "use online mode", "online mode"]:
            if not os.environ.get("GROQ_API_KEY"):
                return "I can't go online without a Groq API key. Please set GROQ_API_KEY in your environment."
            self.router.set_mode("online")
            return "Switching to online mode. I'll use the Groq API from now on."
        
        elif query_lower in ["reset mode", "default mode", "auto mode"]:
            self.router.reset_mode_override()
            return f"Mode reset to default ({self.router._mode})."
        
        return None
    
    def _check_timer(self) -> str:
        handler = self.intent_handler
        if hasattr(handler, '_timer_completed') and handler._timer_completed:
            handler._timer_completed = False
            message = getattr(handler, '_timer_message', "Timer completed!")
            return message
        return None
    
    def _speak_response(self, text: str):
        if self.tts.is_available():
            try:
                self.tts.speak(text)
            except Exception as e:
                self.logger.warning(f"TTS failed: {e}")
    
    def run_interactive(self):
        print("\n" + "=" * 50)
        print("  JARVIS AI ASSISTANT")
        print("=" * 50)
        print(f"  Mode: {self.router.mode.upper()}")
        print(f"  Input: {'Text' if self.text_mode else 'Voice (press Enter to talk)'}")
        print("=" * 50)
        print("\nCommands:")
        print("  • Say 'go online' or 'go offline' to switch modes")
        print("  • Say 'exit' or press Ctrl+C to quit")
        print("  • Ask me anything!")
        print("\n" + "-" * 50 + "\n")
        
        if not self.text_mode and self.stt:
            try:
                self.stt.adjust_for_ambient_noise(duration=1.0)
            except:
                pass
        
        while self.running:
            try:
                if self.text_mode:
                    user_input = input("\nYou: ").strip()
                else:
                    print("\nPress Enter and speak (or type 'text:' for text input)...")
                    input()
                    try:
                        user_input = self.stt.transcribe_from_mic(timeout=10.0)
                        if user_input is None:
                            print("No speech detected. Try again or type 'text:' for text input.")
                            continue
                    except KeyboardInterrupt:
                        continue
                    except Exception as e:
                        self.logger.error(f"STT error: {e}")
                        print("Voice input failed. Type 'text:' followed by your message for text input.")
                        continue
                    
                    if user_input.lower().startswith("text:"):
                        user_input = user_input[5:].strip()
                    elif not user_input:
                        continue
                
                if not user_input:
                    continue
                
                print(f"\nYou: {user_input}")
                sys.stdout.flush()
                
                response = self.process_input(user_input)
                
                if response:
                    print(f"\nJarvis: {response}\n")
                    sys.stdout.flush()
                    
                    if not self.text_mode:
                        self._speak_response(response)
                
            except KeyboardInterrupt:
                print("\n\nInterrupt received...")
                self.running = False
            except EOFError:
                self.logger.info("EOF received")
                self.running = False
            except Exception as e:
                self.logger.error(f"Error in conversation loop: {e}")
                print(f"\nJarvis: I encountered an error. Please try again.\n")
        
        self.shutdown()
    
    def shutdown(self):
        self.logger.info("Shutting down Jarvis...")
        if hasattr(self, 'memory'):
            self.memory.clear()
        print("\nGoodbye!")
        sys.exit(0)


# ============================================================================
# ENVIRONMENT CHECK
# ============================================================================

def check_environment():
    """Check and report on environment setup."""
    print("=" * 50)
    print("  JARVIS ENVIRONMENT CHECK")
    print("=" * 50)
    
    print(f"\nPython version: {sys.version}")
    
    packages = {
        "groq": "Groq API client",
        "pyyaml": "YAML config parsing",
        "python_dotenv": "Environment variable loading",
        "pyttsx3": "Text-to-speech",
        "speech_recognition": "Speech recognition",
        "faster_whisper": "Whisper STT model",
    }
    
    print("\nPackage checks:")
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}: {desc}")
        except ImportError:
            print(f"  ✗ {pkg}: {desc} - NOT INSTALLED")
    
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        print(f"\n  ✓ Groq API key: configured (starts with {api_key[:8]}...)")
    else:
        print("\n  ✗ Groq API key: NOT SET")
        print("    Get a free key at: https://console.groq.com")
    
    try:
        if ollama:
            models = ollama.list()
            if models.get('models'):
                print(f"\n  ✓ Ollama: running with {len(models['models'])} model(s)")
                for m in models['models'][:3]:
                    print(f"    - {m['name']}")
            else:
                print("\n  ⚠ Ollama: running but no models pulled")
                print("    Pull a model: ollama pull qwen2.5:3b")
        else:
            print("\n  ✗ Ollama: Python package not installed")
    except Exception as e:
        print(f"\n  ✗ Ollama: not available ({e})")
    
    print("\n" + "=" * 50)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Jarvis AI Assistant - Voice-driven personal assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python jarvis_single.py                    # Start in default mode
  python jarvis_single.py --mode offline     # Force offline mode
  python jarvis_single.py --text             # Text-only mode (no voice)
  python jarvis_single.py --check            # Check environment setup
        """
    )
    
    parser.add_argument("--mode", choices=["online", "offline"], help="Initial mode")
    parser.add_argument("--text", action="store_true", help="Text-only mode (no microphone)")
    parser.add_argument("--check", action="store_true", help="Check environment and exit")
    
    args = parser.parse_args()
    
    if args.check:
        check_environment()
        return
    
    if not os.environ.get("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not set. Online mode will not be available.")
        print("Get a free key at: https://console.groq.com\n")
    
    try:
        jarvis = Jarvis(mode=args.mode, text_mode=args.text)
        jarvis.run_interactive()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
