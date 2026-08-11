# Master Prompt: Build a "Jarvis" AI Assistant (Python, Groq Online + Local Offline Fallback)

Copy everything below the line into an AI coding assistant (Claude Code, Cursor, etc.) or use it as your own build spec.

---

## SYSTEM / PROJECT PROMPT

You are a senior Python engineer. Build a modular, voice-driven personal assistant called **Jarvis** with two brains, switchable at runtime or via config:

- **ONLINE MODE (primary)** — uses the **Groq API** (free tier, OpenAI-compatible, very low latency) as the LLM brain. Requires a `GROQ_API_KEY`.
- **OFFLINE MODE (fallback)** — uses a local **Ollama** model, no key, no internet, for when there's no connectivity or the user forces offline mode.

### 1. Architecture (pipeline)

```
Mic → Speech-to-Text → Mode Router → Brain → Text-to-Speech → Speaker
                             │
                ┌────────────┴────────────┐
            ONLINE (default)          OFFLINE (fallback)
         Groq API (needs key)      Ollama local model (no key)
```

### 2. Component choices

| Function | Online (primary) | Offline (fallback) |
|---|---|---|
| Brain / LLM | **Groq API** via the `groq` Python SDK (models like `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`) | `Ollama` local model (`qwen2.5:3b`, `phi3`, `llama3.2:3b`) via `ollama` package |
| Speech-to-Text | `faster-whisper` or `vosk` (local either way, doesn't need internet) | same |
| Text-to-Speech | `pyttsx3` (simple, offline) or `Coqui XTTS`/`Kokoro` for nicer voice | same |
| Extra web context (optional) | `duckduckgo_search`, `wikipedia`, `wttr.in` — free, no key | not available offline |

### 3. API key handling (Groq)

- Get a free key at https://console.groq.com — Groq's free tier does not require a credit card at signup.
- **Never hardcode the key.** Load it from an environment variable `GROQ_API_KEY`, or a local `.env` file loaded with `python-dotenv`.
- Add `.env` to `.gitignore` immediately — this is critical if the project is ever pushed to GitHub.
- On startup, check `os.environ.get("GROQ_API_KEY")`. If missing:
  - In online mode, warn the user clearly and either prompt them to paste a key (store it in `.env` for next run) or auto-fall back to offline mode.
- Wrap all Groq calls in try/except for: missing key, invalid key (401), rate limit (429), and network errors — on any of these, log the error and fall back to the offline Ollama brain for that turn (graceful degradation, not a crash).

### 4. Mode-switching design

- Config value (env var, `config.yaml`, or CLI flag `--mode online|offline`) sets the default.
- `ModeRouter` class with one interface `handle(query: str) -> str`, two implementations behind a common `BaseBrain` abstract class (`GroqBrain`, `OfflineBrain`) — strategy pattern.
- Auto-fallback chain: online mode requested → check `GROQ_API_KEY` present → check connectivity (simple DNS/socket check) → call Groq → on any failure, fall back to `OfflineBrain` and log a warning. User should barely notice except a slightly different voice/response style.
- Manual override: user can say/type "go offline" or "go online" mid-session to force a mode for the rest of the session.

### 5. Required features

1. Push-to-talk activation to start (press Enter or hotkey); wake-word can be added later.
2. Conversation loop with short-term memory (last N turns) kept in a deque, passed as message history to Groq (and to the local model in offline mode) for context.
3. Text or voice input, both supported.
4. Rule-based intents handled without the LLM for speed (time, open app/website, set timer, volume) — checked before falling back to the LLM.
5. Robust error handling: mic not found, missing/invalid API key, Ollama not running (offline mode), no internet (online mode), model not pulled locally.
6. Config file for: default mode, Groq model name, local Ollama model name, TTS engine, whisper model size.
7. Logging to a rotating log file; optional conversation transcript saved as JSON.
8. Clean project structure:

```
jarvis/
├── main.py
├── config.yaml
├── .env.example          # GROQ_API_KEY=your_key_here
├── .gitignore             # must include .env
├── requirements.txt
├── core/
│   ├── router.py           # ModeRouter / BaseBrain / GroqBrain / OfflineBrain
│   ├── stt.py
│   ├── tts.py
│   ├── intents.py
│   └── memory.py
├── utils/
│   ├── connectivity.py
│   └── logger.py
└── tests/
```

### 6. Non-functional requirements

- Python 3.10+, cross-platform (Windows/Linux/macOS), call out OS-specific caveats.
- Groq free tier has rate limits — mention them in `SETUP.md` and handle 429s gracefully (short retry/backoff, then fall back offline for that turn if still failing).
- Recommend a small Ollama model (`qwen2.5:3b` or `phi3`) for the offline fallback so it stays lightweight since it's rarely the primary path.

### 7. Deliverables I want from you

1. Full folder/file structure above, fully coded (not pseudocode).
2. `requirements.txt` (include `groq`, `python-dotenv`, plus STT/TTS libs).
3. `SETUP.md`: how to get a Groq key, set up `.env`, install Ollama + pull a fallback model, install pip deps, mic permissions.
4. Clear explanation of the fallback chain (key missing → connectivity fail → API error → offline).
5. Comments on any tricky audio/threading code.

Ask me clarifying questions only if something is genuinely ambiguous; otherwise use sensible defaults and proceed.

---

## Notes for you

- **Groq needs a `GROQ_API_KEY`** — free at console.groq.com, no card required for the free tier as of writing (verify current terms yourself before relying on it).
- Offline Ollama stays as the safety net so the assistant never fully breaks without internet — it's a fallback, not a second primary mode.
- Same TTS/STT wrapper design as before keeps them swappable without touching the brain logic.
- If you later want to drop Groq's free-tier limits, the `GroqBrain` class is the only place model/provider config lives — easy to swap providers later.
