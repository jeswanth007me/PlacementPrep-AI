# PlaceMate AI

> **Practice. Prepare. Place.**

PlaceMate AI is an agentic AI-powered interview preparation platform designed to help candidates prepare for campus placements and technical interviews through adaptive, deeply personalized mock interviews grounded in their actual resumes, public GitHub projects, and real-time performance.

---

## The Core Concept: Agentic vs. Traditional Chatbots

Traditional interview bots operate as linear prompt responders:
```
[Question] ──> [User Answer] ──> [Static Next Question]
```

**PlaceMate AI** operates as a closed-loop agentic workflow:

```
  Resume + Public GitHub + Target Role
                  │
                  ▼
         [ 1. ANALYZE ]   Extract structured candidate context & project technical profiles
                  │
                  ▼
          [ 2. DECIDE ]    Formulate personalized questions probing real project implementations
                  │
                  ▼
           [ 3. ACT ]      Deliver technical questions via interactive interview session
                  │
                  ▼
        [ 4. EVALUATE ]    Assess answers objectively ('strong', 'acceptable', 'weak')
                  │
                  ▼
         [ 5. ADAPT ]     Deterministically update difficulty (Levels 1–5) & trigger RAG when needed
                  │
                  ▼
        [ 6. RECOMMEND ]   Synthesize multidimensional evaluation scores & tailored career roadmap
```

---

## Key Features (Currently Implemented)

### 1. Structured Resume Analysis
- Accepts plain-text candidate resume input and target job role.
- Extracts a validated Pydantic `CandidateProfile`: candidate name, target role, technical skills, project descriptions, technologies, education, and experience.
- Identifies strategic technical probe areas without assuming candidate deficiencies in advance.

### 2. Public GitHub Repository Analysis
- Analyzes public GitHub repositories for a given username using the GitHub REST API.
- Evaluates repository metadata, primary languages, topics, README content, and directory file trees.
- Extracts a structured `ProjectProfile` for top repositories, identifying architectural patterns, core technologies, and potential interview questions grounded in the candidate's actual public code.
- *Note*: GitHub OAuth is not implemented; analysis is performed via public repository endpoints with an optional personal access token for rate-limit headroom.

### 3. Agentic Interview Workflow (LangGraph)
- Multi-node state machine powered by LangGraph (`StateGraph`).
- Generates targeted, single-prompt questions directly tailored to the candidate's profile, projects, and target role.
- Maintains conversation history, current question index, difficulty state, evaluation feedback, and a structured decision log across turns.

### 4. Objective Answer Evaluation
- Pre-turn evaluation node (`evaluate_candidate_answer`) evaluates the candidate's technical response against the specific question asked.
- Uses structured output (`AnswerEvaluation`) with standardized criteria:
  - **`strong`**: Technically accurate, complete, addresses trade-offs and mechanisms.
  - **`acceptable`**: Fundamentally correct on basic concepts, but lacks depth or nuances.
  - **`weak`**: Contains significant technical inaccuracies, confusion, or misses the question.
- Returns concise constructive feedback explaining the rating.

### 5. Deterministic Adaptive Difficulty (Levels 1–5)
- Question difficulty is bounded on a scale of **1** (Foundational Concepts) to **5** (Deep-Dive & Architectural Systems).
- Difficulty transitions are calculated deterministically by application logic—the LLM is not permitted to assign arbitrary numbers:
  - `strong` answer $\rightarrow$ Difficulty $+ 1$ (capped at 5)
  - `acceptable` answer $\rightarrow$ Difficulty unchanged (re-probes or pivots)
  - `weak` answer $\rightarrow$ Difficulty $- 1$ (floored at 1)

### 6. Agentic Retrieval-Augmented Generation (RAG)
- The interview agent is equipped with a callable retrieval tool: `retrieve_prep_material(query, topic)`.
- **Autonomous Decision**: The agent decides *when* and *if* preparation reference material is needed based on the conversation turn and candidate performance; retrieval is never blindly forced on every turn.
- **Corpus & Search**: Searches a curated technical interview knowledge base using precomputed 3072-dimensional embeddings and vector cosine similarity.
- **Confidence Gate**: Enforces a strict confidence threshold (default: `0.70`). If similarity falls below threshold, the agent explicitly falls back to prior conversation history and candidate context.

### 7. Session Memory & Multi-Turn State
- State is managed via LangGraph's `InMemorySaver` checkpointer and indexed by an explicit `thread_id`.
- Preserves full turn history (`HumanMessage`, `AIMessage`, `ToolMessage`), accumulated decision log entries, current difficulty, and candidate context across HTTP requests.
- *Note*: Memory is currently session-based in RAM and is not backed by a persistent database.

### 8. Multidimensional Final Evaluation Report
- Upon interview completion (`POST /api/interview/finish`), the agent synthesizes the complete decision log and candidate profile into a structured `FinalResult`.
- Scores (0–100):
  - **Overall Score**
  - **Technical Knowledge**
  - **Problem Solving**
  - **Communication**
  - **Confidence**
- Extracts categorized **Strengths** and **Areas for Improvement (Weaknesses)** based on demonstrated answers.

### 9. Personalized Career & Study Recommendations
- Generates 2–3 recommended job roles with percentage match scores and actionable rationales based on demonstrated skills and interview performance.
- Provides 2–4 targeted technical study topics to guide candidate placement preparation.

---

## Architecture

```
                    ┌────────────────────────────────────────┐
                    │       PlaceMate AI Web Frontend        │
                    │      (React 19 + TypeScript + Vite)    │
                    └───────────────────┬────────────────────┘
                                        │ HTTP REST Requests
                                        ▼
                    ┌────────────────────────────────────────┐
                    │          FastAPI Application           │
                    │            (backend/api.py)            │
                    └───────┬────────────────────────┬───────┘
                            │                        │
       Resume / GitHub Data │                        │ Session Thread ID
                            ▼                        ▼
       ┌──────────────────────────────┐    ┌─────────────────────────────────┐
       │   Candidate Context Engine   │    │      Session Manager (In-RAM)   │
       │  • parser.py (Resume Parser) │    │  • InMemorySaver Checkpointer   │
       │  • github/analyzer.py        │    │  • thread_id Session Registry   │
       └──────────────┬───────────────┘    └────────────────┬────────────────┘
                      │                                     │
                      └──────────────────┬──────────────────┘
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    LangGraph Interview Agent Graph   │
                      │          (backend/agent.py)          │
                      │                                      │
                      │    START                             │
                      │      │                               │
                      │      ▼                               │
                      │  [ evaluate_node ]                   │
                      │      │  • Rates answer               │
                      │      │  • Computes difficulty (1-5)  │
                      │      ▼                               │
                      │  [ interviewer_node ] ◄────────┐     │
                      │      │                         │     │
                      │      ├─(tool call)──► [ tools ]│     │
                      │      │                (RAG)    │     │
                      │      ▼                         │     │
                      │  [ log_decision ] ─────────────┘     │
                      │      │                               │
                      │      ▼                               │
                      │     END                              │
                      └──────────────────┬───────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
┌─────────────────────────────────┐             ┌─────────────────────────────────┐
│     Agentic RAG Tool (rag.py)   │             │   Results Engine (results.py)   │
│  • Curated Interview Corpus     │             │  • Multidimensional 0-100 Scores│
│  • Precomputed Embeddings       │             │  • Strengths & Weaknesses       │
│  • Cosine Similarity (>= 0.70)  │             │  • Role Matches & Study Topics  │
└─────────────────────────────────┘             └─────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technologies Used | Details / Usage |
| :--- | :--- | :--- |
| **AI & Orchestration** | LangChain, LangGraph, Pydantic v2 | Graph workflow, state checkpointing, Pydantic structured output validation |
| **LLM Provider** | Google Gemini (`gemini-3.6-flash`) | Structured extraction, dynamic questioning, answer evaluation, final report |
| **Embeddings & RAG** | Gemini Embeddings (`gemini-embedding-001`), Cosine Similarity | Precomputed 3072-dimensional embeddings for curated interview prep corpus |
| **Backend API** | Python 3.11+, FastAPI, Uvicorn | RESTful endpoints, CORS middleware, session manager |
| **Frontend UI** | React 19, TypeScript, Vite, Tailwind CSS | Modern responsive dark-mode interface, audio visualization, live feedback |
| **Integration** | GitHub REST API (`httpx`) | Public profile and repository tree/README analysis |
| **Testing** | pytest, FastAPI TestClient, unittest | 58 automated unit & integration tests across 5 suites |

---

## REST API Reference

All routes are served by the FastAPI application (`backend/api.py`) on `http://localhost:8000`.

### Health Check
```http
GET /health
```
- **Response**: `{"status": "healthy", "service": "PlaceMate AI API"}`

### Candidate Resume Analysis
```http
POST /api/candidate/analyze-resume
```
- **Payload**:
  ```json
  {
    "resume_text": "Plain text of candidate resume...",
    "target_role": "Backend Engineer"
  }
  ```
- **Response**: Structured `CandidateProfile` JSON (name, skills, projects, experience, interview focus).

### GitHub Profile Analysis
```http
POST /api/github/analyze
```
- **Payload**:
  ```json
  {
    "username": "octocat",
    "target_role": "Backend Engineer"
  }
  ```
- **Response**: Structured `GitHubProfile` JSON (repositories, primary languages, architecture summaries).

### Start Interview Session
```http
POST /api/interview/start
```
- **Payload**:
  ```json
  {
    "candidate_profile": { ... },
    "subject": "Python",
    "thread_id": "optional-custom-uuid"
  }
  ```
- **Response**:
  ```json
  {
    "thread_id": "e2e_thread_1789658382156",
    "question": "Hello Alex! To begin...",
    "difficulty": 2,
    "evaluation_rating": "greeting",
    "decision_log": [],
    "messages": [ ... ]
  }
  ```

### Submit Candidate Answer
```http
POST /api/interview/answer
```
- **Payload**:
  ```json
  {
    "thread_id": "e2e_thread_1789658382156",
    "answer": "I used Redis caching with an LRU eviction policy..."
  }
  ```
- **Response**: Returns the next question (clean string), `evaluation_rating`, updated `difficulty`, and `decision_log` entry.

### Finish Interview Session
```http
POST /api/interview/finish
```
- **Payload**:
  ```json
  {
    "thread_id": "e2e_thread_1789658382156"
  }
  ```
- **Response**: Generates and returns the complete `FinalResult` report.

### Retrieve Session Result
```http
GET /api/interview/result?thread_id=e2e_thread_1789658382156
```
- **Response**: Current interview messages, decision logs, and cached final result if completed.

### Retrieve Career Recommendations
```http
GET /api/recommendations?thread_id=e2e_thread_1789658382156
```
- **Response**: Structured list of `recommended_roles` (with match scores and reasons) and `recommended_topics`.

---

## Repository Structure

```
placment prep -ai/
├── backend/                           # FastAPI backend and LangGraph agent package
│   ├── __init__.py
│   ├── agent.py                       # LangGraph StateGraph, interviewer nodes & tools
│   ├── api.py                         # FastAPI REST API application & endpoints
│   ├── evaluation.py                  # Answer evaluator & deterministic difficulty scaling
│   ├── memory.py                      # InMemorySaver checkpointer & session config helpers
│   ├── models.py                      # Pydantic schemas (CandidateProfile, FinalResult, etc.)
│   ├── parser.py                      # Resume parser using structured output
│   ├── prompts.py                     # Role-grounded system prompts & level guidelines
│   ├── rag.py                         # Confidence-gated vector retrieval over prep corpus
│   ├── results.py                     # Final report & recommendations generator
│   ├── session_manager.py             # In-memory session tracking by thread_id
│   └── github/
│       ├── __init__.py
│       ├── analyzer.py                # Repository analyzer extracting ProjectProfile
│       └── client.py                  # GitHub REST API client
├── data/
│   ├── prep_embeddings.json           # Precomputed embeddings for curated preparation corpus
│   └── sample_resume.txt              # Sample student resume for local testing & demos
├── tests/                             # Automated test suite (58 unit & integration tests)
│   ├── __init__.py
│   ├── test_agent.py                  # Milestone 1 agent workflow & prompt tests (13 tests)
│   ├── test_api.py                    # Milestone 4 REST API endpoints & session tests (22 tests)
│   ├── test_github.py                 # GitHub analyzer & client mocked tests (5 tests)
│   ├── test_rag.py                    # RAG retrieval & confidence threshold tests (11 tests)
│   └── test_resume_parser.py          # Structured resume parser tests (7 tests)
├── PlaceMate-AI-Frontend/             # React 19 + TypeScript + Vite web application
│   ├── src/
│   │   ├── context/AppContext.tsx     # Global interview state & session actions
│   │   ├── services/api.ts            # Frontend API client talking to FastAPI backend
│   │   ├── screens/                   # 16 specialized UI screens (Dashboard, LiveInterview, etc.)
│   │   ├── types.ts                   # Frontend TypeScript interfaces
│   │   └── App.tsx                    # Main layout and screen router
│   ├── package.json                   # Frontend dependencies and npm scripts
│   ├── vite.config.ts                 # Vite bundler configuration
│   └── server.ts                      # Express server for local development & production preview
├── .env.example                       # Backend environment template (safe to commit)
├── requirements.txt                   # Minimal Python backend dependencies
└── README.md                          # Project documentation
```

---

## Local Setup & Installation

### 1. Prerequisites
- **Python**: 3.10 or 3.11 (tested on Python 3.11)
- **Node.js**: v18 or higher (tested on Node.js v20+)
- **Google Gemini API Key**: Obtain from [Google AI Studio](https://aistudio.google.com/)

---

### 2. Backend Setup

1. **Navigate to the root directory**:
   ```bash
   cd "placment prep -ai"
   ```

2. **Create and activate a virtual environment**:
   - *Windows (PowerShell)*:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - *Linux / macOS*:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a local `.env` file in the project root based on `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Set the following variables:
   ```env
   LLM_PROVIDER=gemini
   GEMINI_MODEL=gemini-3.6-flash
   GEMINI_API_KEY=your_actual_gemini_api_key

   # Optional: GitHub personal access token for higher rate limits on public repos
   GITHUB_TOKEN=your_optional_github_token
   ```

   > **Security Note**: Never commit `.env` or expose API keys to GitHub. The `.env` file is gitignored.

5. **Start the backend server**:
   ```bash
   python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
   ```
   Verify at `http://localhost:8000/health`.

---

### 3. Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd PlaceMate-AI-Frontend
   ```

2. **Install Node dependencies**:
   ```bash
   npm install
   ```

3. **Configure local frontend environment**:
   Create `.env.local` in `PlaceMate-AI-Frontend/`:
   ```env
   VITE_API_BASE=http://localhost:8000
   ```

4. **Run the frontend development server**:
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000` in your web browser.

5. **Build for production**:
   ```bash
   npm run build
   ```

---

### 4. Running Backend Automated Tests

All tests in `tests/` use mocked LLM responses and mocked network calls, allowing the full test suite to execute fast and reliably without consuming live Gemini API quota:

```bash
pytest tests/
```

Expected result:
```text
tests/test_agent.py .............                                        [ 22%]
tests/test_api.py ......................                                 [ 60%]
tests/test_github.py .....                                               [ 68%]
tests/test_rag.py ...........                                            [ 87%]
tests/test_resume_parser.py .......                                      [100%]
============================== 58 passed in ~50s ===============================
```

---

## Verified End-to-End Flow

The PlaceMate AI pipeline has been verified end-to-end against live Google Gemini endpoints across all 12 stages:

```
[ Frontend Reachable (HTTP 200) ]
               ↓
[ Candidate Profile Initialized ]
               ↓
[ Resume Analysis (POST /api/candidate/analyze-resume) ] ──> Structured CandidateProfile
               ↓
[ Interview Start (POST /api/interview/start) ] ──────────> Personalized First Question
               ↓
[ Candidate Answer (POST /api/interview/answer) ]
               ↓
[ Answer Evaluation ] ────────────────────────────────────> Objective Rating ('weak'/'acceptable'/'strong')
               ↓
[ Difficulty Adaptation ] ────────────────────────────────> Bounded Step (Level 2 → 1)
               ↓
[ Agentic RAG Retrieval ] ────────────────────────────────> Confidence 0.81 (Query applied to next turn)
               ↓
[ Adaptive Follow-Up Question ] ──────────────────────────> Clean string question generated
               ↓
[ Interview Finish (POST /api/interview/finish) ] ────────> Multidimensional Scoring (45/100)
               ↓
[ Recommendations (GET /api/recommendations) ] ───────────> Matched Roles & Targeted Study Topics
```

---

## Responsible AI & Reliability Safeguards

1. **Grounded Schema Validation**:
   All outputs passed across components (Candidate Profile, Answer Evaluation, Decision Log, Final Report) are validated via strict Pydantic schemas. Unstructured or hallucinated outputs are rejected at the parsing boundary.
2. **Deterministic Difficulty Scaling**:
   The LLM cannot set difficulty arbitrarily. Transitions are bound to strict programmatic rules ($+1$, $0$, $-1$) clamped strictly within $[1, 5]$.
3. **Confidence-Gated RAG**:
   Retrieval requires a cosine similarity $\ge 0.70$. Sub-threshold matches trigger safe fallbacks to candidate context rather than injecting low-relevance or confusing facts into the interview.
4. **Transparent Decision Audit Logs**:
   Every turn records the exact difficulty before/after, evaluation rating, retrieval status, and rationales, giving candidates full transparency into why questions were asked.
5. **Human Placement Preparation Disclaimer**:
   PlaceMate AI is designed exclusively as an educational practice and preparation platform for candidates. It is not an automated hiring decision system and is not designed to replace human evaluation in formal employment processes.

---

## Current Architecture Limitations

To maintain technical transparency, the current prototype implementation has the following known boundaries:
- **Session Memory vs. Database**: Session memory is currently maintained in-RAM via LangGraph's `InMemorySaver` checkpointer and an in-memory session manager. Restarting the backend server clears active sessions.
- **Authentication**: User accounts, logins, and settings are currently stored in local frontend state. Production JWT/session authentication is not yet integrated into the backend.
- **GitHub Public Access Only**: The GitHub analyzer evaluates public repositories via the public REST API. GitHub OAuth and private repository access are not currently implemented.
- **Precomputed Embedding Corpus**: RAG retrieval utilizes precomputed Gemini embeddings and in-memory cosine similarity over a curated Python interview corpus rather than an external production vector database cluster.
- **Plain-Text Resume Ingestion**: The current resume parser accepts plain text. Uploading binary PDF or DOCX files currently relies on text extraction before reaching the backend parser.

---

## Future Scope

- [ ] **Persistent Database**: Integration with PostgreSQL/SQLite for long-term candidate profile storage, score trends, and past interview history.
- [ ] **Full User Authentication**: Secure candidate accounts with JWT authentication and session persistence.
- [ ] **GitHub OAuth Integration**: Direct OAuth authorization to allow candidate-consented access to private portfolios and code repositories.
- [ ] **Native Document Parsing**: Direct PDF/DOCX multi-page document parsing and OCR on the backend.
- [ ] **Voice-to-Voice Real-Time Interviewing**: Audio streaming using Gemini Live / WebRTC for hands-free spoken mock interviews.
- [ ] **Behavioral & HR Interview Modes**: STAR-method evaluation agents for behavioral, leadership, and cultural fit interviews.
- [ ] **Production Vector DB**: Migration of preparation knowledge bases to scalable vector infrastructure (pgvector / ChromaDB) supporting expanded subject catalogs beyond Python backend engineering.

---

## License

This project is developed for educational, placement preparation, and hackathon demonstration purposes.
