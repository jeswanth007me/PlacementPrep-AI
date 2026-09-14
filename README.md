# PlacementPrep AI — Agentic Interview & Career Preparation Assistant

**PlacementPrep AI** is an agentic AI interview preparation system designed to help students prepare for campus placements through adaptive mock interviews.

---

## Current Status: Milestone 1 (Conversational Interview Agent Core)

This repository currently implements **Milestone 1**: a working, minimal, conversational mock interview agent.

### Milestone 1 Capabilities
- **Focused Technical Questions**: Asks concise, one-at-a-time technical interview questions (default subject: Python).
- **Contextual Follow-ups**: Listens to the candidate's answers and asks intelligent follow-up questions probing depth, edge cases, and trade-offs.
- **Adaptive Difficulty**: Dynamically adapts the difficulty of the next question based on the candidate's response.
- **Session Memory Persistence**: Preserves full multi-turn conversation history across turns using LangGraph's `InMemorySaver` checkpointer and an explicit `thread_id`.
- **Interactive CLI**: Interactive terminal interface to practice mock interviews directly.

### Planned / Future Milestones (Not Yet Implemented)
- Candidate resume ingestion and profile building
- RAG (Retrieval-Augmented Generation) knowledge base for preparation materials
- Automated rubric-based evaluation and scoring reports
- Voice input / output (STT & TTS)
- Web search grounding
- Web frontend UI and deployment

---

## Technologies Used

- **Language**: Python 3.10+ (tested on Python 3.11)
- **Agent Framework**: LangChain (`langchain-core`, `langchain`)
- **State & Workflow Management**: LangGraph (`langgraph`, `langgraph-checkpoint`)
- **LLM**: Google Gemini (`gemini-3.5-flash`) via `langchain-google-genai`
- **Memory**: LangGraph `InMemorySaver` (in-RAM checkpointer)
- **Configuration**: `python-dotenv` for local environment management
- **Testing**: Python standard library `unittest`

---

## Project Structure

```
PlacementPrep-AI/
├── backend/
│   ├── agent.py       # StateGraph definition, Gemini client, and CLI runner
│   ├── prompts.py     # Interviewer system prompts and behavioral rules
│   └── memory.py      # InMemorySaver checkpointer and session config helpers
├── tests/
│   └── test_agent.py  # Automated unit and integration tests
├── .env.example       # Template for local environment variables (safe to commit)
├── .gitignore         # Ignores .env, virtual environments, and caches
├── requirements.txt   # Minimal dependencies
└── README.md          # Project documentation
```

---

## How Memory Works

1. **StateGraph with MessagesState**:
   The conversation is structured as a LangGraph state graph. The state contains a `messages` list managed with LangGraph's `add_messages` reducer. When the interviewer node produces a new response, it is appended to the message list rather than overwriting previous messages.

2. **InMemorySaver Checkpointer**:
   When the graph is compiled (`workflow.compile(checkpointer=checkpointer)`), the `InMemorySaver` stores snapshots of the conversation state in memory, indexed by session.

3. **Session Tracking via `thread_id`**:
   Every invocation provides a config dictionary:
   ```python
   config = {"configurable": {"thread_id": "interview-session-1"}}
   ```
   LangGraph loads all accumulated prior turns matching the `thread_id` before invoking the model, providing full conversation context to Gemini.

---

## Getting Started (Setup for Person 2 / New Clones)

### 1. Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key (obtain from [Google AI Studio](https://aistudio.google.com/))

### 2. Clone the Repository
```bash
git clone <repository-url>
cd placement-prep-ai
```

### 3. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Local Environment Variables

Create your local `.env` file from `.env.example`:

**Windows:**
```powershell
copy .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

Open `.env` in a text editor and insert your Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

> **Security Note:** The `.env` file contains secret credentials and is ignored by Git. Never commit `.env` or share your API key.

### 6. Run Automated Tests

The test suite verifies prompt generation, memory persistence, multi-turn accumulation, session isolation, and API key validation:

```bash
python -m unittest tests/test_agent.py
```

Expected output:
```
Ran 7 tests in ...
OK
```

### 7. Run the Conversational Interview Agent

Start the interactive CLI interview session:

```bash
python backend/agent.py
```

- The agent will greet you and ask the first Python question.
- Type your answer and press Enter.
- The agent remembers previous answers and asks an adaptive follow-up.
- Type `exit` or `quit` at any time to end the session.
