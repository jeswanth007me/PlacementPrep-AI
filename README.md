# PlacementPrep AI — Agentic Interview & Career Preparation Assistant

**PlacementPrep AI** is an agentic AI interview preparation system designed to help students prepare for campus placements through adaptive, personalized mock interviews.

---

## Current Status: Milestone 1, 2 & 3 Completed

### Milestone 1: Conversational Interview Agent Core
- **Focused Technical Questions**: Asks concise, single-question technical interview questions (default: Python).
- **Contextual Follow-ups**: Listens to the candidate's answers and asks intelligent follow-up questions probing depth, edge cases, and trade-offs.
- **Adaptive Difficulty**: Dynamically adapts the difficulty of the next question based on the candidate's response.
- **Session Memory Persistence**: Preserves full multi-turn conversation history across turns using LangGraph's `InMemorySaver` checkpointer and an explicit `thread_id`.

### Milestone 2: Candidate Profile & Resume Analysis
- **Structured Resume Extraction**: Decoupled function `analyze_resume(resume_text: str, target_role: str) -> CandidateProfile` powered by Google Gemini 3.5 Flash with structured outputs.
- **Pydantic Validation**: Enforces validated schemas for `CandidateProfile` and `ProjectSummary`.
- **Personalized Interviewing**: Injects the candidate's name, target role, technical skills, and project summaries into the interviewer system prompt.
- **Role-Grounded & Project-Specific Questions**: The interviewer directly references the candidate's actual projects and tech stack.
- **Strategic Probing**: Identifies topics and trade-offs to probe during the interview without assuming or claiming that they are confirmed weaknesses.
- **Backward Compatible**: Milestone 1 generic interviews continue to work seamlessly when no resume is supplied.

### Milestone 3: Agentic RAG + Adaptive Decision Logic
- **Autonomous Retrieval Decision**: The agent natively decides *when* and *if* to call `retrieve_prep_material(query, topic)`. Retrieval is never blindly forced on every turn.
- **Confidence-Gated Material**: Uses cosine similarity over Google Gemini 3072-dimensional embeddings (`gemini-embedding-001`). Material is accepted only if confidence >= 0.70. Sub-threshold results trigger explicit fallback to candidate context and conversation history.
- **Structured Answer Evaluation**: Pre-turn evaluation node (`evaluate_candidate_answer`) objectively rates candidate answers as `strong`, `acceptable`, or `weak` with concise rationale.
- **Deterministic Difficulty Adaptation**: Adjusts question difficulty on a 1–5 scale based strictly on evaluation rating (+1 for strong, -1 for weak, 0 for acceptable; clamped [1, 5]).
- **Transparent Decision Logs**: Every turn logs evaluation rating, difficulty transitions, retrieval queries, similarity scores, and fallback flags.

### Planned / Future Milestones (Not Yet Implemented)
- Automated final interview preparation & scoring report
- Web frontend UI and deployment (Person 2 integration)

---

## Technologies Used

- **Language**: Python 3.10+ (tested on Python 3.11)
- **Agent Framework**: LangChain (`langchain-core`, `langchain`)
- **State & Workflow Management**: LangGraph (`langgraph`, `langgraph-checkpoint`)
- **LLM**: Google Gemini (`gemini-3.5-flash`) via `langchain-google-genai`
- **Embeddings**: Google Gemini (`gemini-embedding-001`) via `google-genai`
- **Data Validation**: Pydantic v2
- **Memory**: LangGraph `InMemorySaver` (in-RAM checkpointer)
- **Configuration**: `python-dotenv` for local environment management
- **Testing**: Python standard library `unittest`

---

## Project Structure

```
PlacementPrep-AI/
├── backend/
│   ├── agent.py               # StateGraph workflow, tools, interviewer nodes, and CLI
│   ├── models.py              # Pydantic models (CandidateProfile, AnswerEvaluation, DecisionLogEntry)
│   ├── parser.py              # analyze_resume() structured extraction via Gemini
│   ├── evaluation.py          # Answer evaluation node & deterministic difficulty scaling
│   ├── rag.py                 # Embedding generation, vector similarity & confidence-gated retrieval
│   ├── prompts.py             # System prompts with level guidelines & agentic retrieval rules
│   └── memory.py              # InMemorySaver checkpointer and session config helpers
├── data/
│   ├── sample_resume.txt      # Realistic sample student resume for testing & demo
│   └── prep_embeddings.json   # Cached embeddings for preparation knowledge base
├── tests/
│   ├── __init__.py            # Test package marker
│   ├── test_agent.py          # Unit & integration tests for Milestone 1
│   ├── test_resume_parser.py  # Unit & integration tests for Milestone 2
│   └── test_rag.py            # Unit & integration tests for Milestone 3 (RAG & evaluation)
├── .env.example               # Template for local environment variables (safe to commit)
├── .gitignore                 # Ignores .env, virtual environments, and caches
├── requirements.txt           # Minimal dependencies
└── README.md                  # Project documentation
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

## Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key (obtain from [Google AI Studio](https://aistudio.google.com/))

### 2. Clone the Repository
```bash
git clone https://github.com/jeswanth007me/PlacementPrep-AI.git
cd PlacementPrep-AI
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

Run the complete test suite across all milestones (Milestones 1, 2, and 3):

```bash
python -m unittest discover -v
```

Expected output:
```
Ran 25 tests in ...
OK
```

### 7. Run the Conversational Interview Agent

#### Option A: Personalized Interview with Resume Analysis & Agentic RAG (Milestones 2 & 3)
```bash
python backend/agent.py --resume data/sample_resume.txt
```
- The agent analyzes the resume, extracts the structured `CandidateProfile`, and displays a profile summary.
- The interviewer greets the candidate by name and asks tailored questions probing their specific projects and tech stack.
- Between turns, the agent transparently evaluates candidate answers, adapts difficulty (Levels 1–5), queries the prep knowledge base when helpful, and prints a structured `Decision Log`.

#### Option B: Standard Generic Technical Interview (Milestones 1 & 3)
```bash
python backend/agent.py
```
- Starts a standard technical interview on Python with adaptive difficulty and confidence-gated RAG.
- Type `exit` or `quit` at any time to end the session.
