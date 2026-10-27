# ElderMind AI 🧠

ElderMind AI is an AI assistant application designed to aid cognitive support, memory search, facial recognition, and event timeline tracking.

## Setup & Running

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch Streamlit UI:
   ```bash
   streamlit run app/streamlit_app.py
   ```
🧠 MemorAi — AI-Powered Digital Memory Assistant
MemorAi is a local, multimodal memory assistance prototype designed to support older adults and people who benefit from help recalling everyday information. It combines computer vision, face recognition, speech transcription, medication-event monitoring, and retrieval-augmented generation (RAG) to turn observed events into searchable memories.
Project status: Version 1 prototype. MemorAi is an assistive research/development project, not a clinical device, guaranteed medication adherence monitor, or substitute for a caregiver.

✨ Key Features
Module	What it does
Object memory	Detects selected everyday objects through a camera, records observations and timestamps, and helps answer where an object was last observed.
People memory	Registers familiar people using face samples, recognizes previously registered faces, and records encounter events.
Conversation memory	Transcribes recorded speech, identifies important conversation content, and supports retrieval of saved conversations.
Medicine memory	Registers medicine reference images, recognizes medicines with visual/textual cues, tracks camera-observed interactions, and presents evidence-qualified medication states.
Ask MemorAi	Retrieves relevant memories and uses a local LLM to answer questions grounded in available records.
Voice assistant	Accepts spoken questions through the web interface and transcribes them for memory-based answering.
Unified timeline	Displays events from different memory modules in chronological order.
Local dashboard	Provides a NiceGUI interface for camera, people, medicine, conversations, memory search, settings, and timeline.


🎯 Motivation
Remembering where personal items were last seen, who visited, what was discussed, and whether a medication-related action was observed can be challenging. MemorAi explores how AI-powered observation and local memory retrieval can provide contextual reminders without pretending to know more than the available evidence supports.

Example questions:
- “Where was my phone last seen?”
- “When did I last see this person?”
- “What did we discuss earlier?”
- “Was a medication event recorded today?”
Answers depend on actual stored observations. MemorAi is designed to report uncertainty rather than invent missing memories.
🧰 Technology Stack
Area	Tools / Models
Core language	Python
Main user interface	NiceGUI, FastAPI
Alternative / earlier UI	Streamlit
Object detection	Ultralytics YOLO11s, OpenCV
Face detection and recognition	OpenCV YuNet and SFace (ONNX)
Medicine visual representation	DINOv2, OCR-based text cues, MediaPipe hand tracking
Speech recognition	Faster-Whisper
Semantic embeddings	BAAI/bge-small-en-v1.5, Sentence Transformers
Vector storage	ChromaDB
Structured storage	MySQL
Local question answering	Ollama with qwen3:4b
AI/ML ecosystem	PyTorch, Hugging Face Transformers, scikit-learn

How it works
1. Capture: The application receives camera frames or user-initiated audio input.
2. Interpret: Specialized models identify objects, faces, medications, speech, or contextual events.
3. Confirm: Confidence thresholds, multi-frame validation, and cooldowns help reduce repeated or unstable visual detections.
4. Store: Structured observations and metadata are stored in MySQL; selected memory text is embedded for semantic search in ChromaDB.
5. Retrieve: A user question is matched to relevant object, person, conversation, or medication memories.
6. Respond: A locally hosted LLM generates a concise answer constrained by retrieved evidence.
📁 Repository Structure
MemorAi/
├── app/
│   ├── nicegui_app.py       # Main NiceGUI application
│   ├── nicegui_ui/          # Pages, layout, and UI components
│   ├── services/            # Shared camera service
│   ├── streamlit_app.py     # Alternative/earlier Streamlit interface
│   └── ui/                  # Streamlit UI modules
├── conversation/            # Transcription-based conversation memory
├── database/                # MySQL schema, repositories, vector integrations
├── face/                    # Person registration, recognition, monitoring
├── llm/                     # Ollama question answering
├── medicine/                # Medicine registration and event monitoring
├── memory/                  # Domain-specific memory retrieval
├── models/                  # Local model resources (some downloaded separately)
├── rag/                     # Embeddings and semantic search
├── speech/                  # Audio input and transcription
├── tests/                   # Module and integration tests
├── timeline/                # Unified event timeline
├── utils/                   # Shared helpers and logging
├── vision/                  # Object detection, confirmation, location mapping
├── config.py                # Central application configuration
├── .env.example             # Local configuration template
└── requirements.txt         # Python dependencies
🚀 Installation and Setup
Prerequisites
- Python with a compatible virtual environment (check the pinned versions in requirements.txt)
- MySQL Server running locally or on a trusted host
- Ollama installed and running
- Working webcam and microphone for camera and voice features
- Sufficient disk space for computer vision, speech, and embedding models
Platform note: The prototype was developed for local execution. Package and accelerator compatibility may vary by Python, operating system, and CUDA/PyTorch versions. If installing all pinned dependencies fails, create a compatible environment and resolve version constraints before launching.

1. Clone the repository
git clone https://github.com/SakshiCC/MemorAi.git
cd MemorAi
2. Create and activate a virtual environment
Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate
3. Install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
If your environment does not already include NiceGUI, install it separately:
pip install nicegui
The repository's current requirements.txt includes Streamlit but does not explicitly pin NiceGUI, although the main application imports it.
4. Configure environment variables
Copy the template and fill in local settings:

Windows:
Copy-Item .env.example .env
Linux/macOS:
cp .env.example .env
At minimum, review these entries in .env:
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=Memor_ai

CAMERA_INDEX=0
YOLO_MODEL_PATH=models/yolo/yolo11s.pt

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b

CHROMA_PATH=chroma_data
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5

MEMORAI_STORAGE_SECRET=replace_with_a_long_random_secret
Never commit your .env, passwords, registered faces, private recordings, or user memory records. Review repository contents carefully before pushing sample data or evidence images.
5. Prepare MySQL
Create the database and a database user with appropriate local permissions. For example, in MySQL:
CREATE DATABASE IF NOT EXISTS Memor_ai
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
The NiceGUI startup invokes the repository's initialize_database() function to create required tables. The configured user must have the necessary permissions.
6. Prepare Ollama
ollama pull qwen3:4b
ollama serve
If Ollama is already running as a background service, you do not need a second ollama serve process.
7. Prepare model files
The configuration expects model resources including:
models/yolo/yolo11s.pt
models/face/face_detection_yunet_2023mar.onnx
models/face/face_recognition_sface_2021dec.onnx
models/medicine/hand_landmarker.task
Download or provide missing weights according to the applicable model licenses and the repository's model downloader scripts. DINOv2, embedding, and Whisper models may also download when first used.
8. Run MemorAi
Recommended — NiceGUI application:
python -m app.nicegui_app
Then open http://127.0.0.1:8080 in your browser.

Alternative — Streamlit interface:
streamlit run app/streamlit_app.py
The Streamlit interface is an earlier/alternative frontend and may not expose every NiceGUI feature.
🧭 Application Sections
Page	Purpose
Home	Quick overview and access to the assistant
Ask MemorAi	Question answering over stored memories
Voice	Ask memory questions using speech
Camera	View and manage live camera-assisted monitoring
People	Register and review familiar people
Medicines	Manage registered medicines and their observation history
Conversations	Capture and access important conversation memories
Timeline	Review memories and recorded events chronologically
Settings	Configure relevant application options


🧪 Testing
The tests/ directory contains focused tests for object monitoring, person registration and recognition, conversation privacy and retrieval, medicine processing, voice pipelines, RAG, and the unified timeline.

To run tests, first install a compatible pytest version, then execute specific tests as appropriate:
pip install pytest
python -m pytest tests/test_unified_rag.py
python -m pytest tests/test_unified_timeline.py
Some tests require a live camera, MySQL, Ollama, downloaded models, or locally configured data. Passing the tests has not been independently verified as part of this README update.
🔐 Privacy, Safety, and Limitations
- Local-first prototype: Ollama and MySQL can run locally. Some model setup steps may require downloads; local hosting does not automatically guarantee that all data is private or secure.
- Face and conversation consent: Only register or record people with appropriate notice, consent, and lawful authorization.
- Observational, not definitive: An item’s last observed position may differ from its current location; face identification may be imperfect.
- Medication evidence: A camera-confirmed interaction is not proof of swallowing or taking a prescribed dose. Do not use MemorAi as the sole basis for medication decisions.
- Model errors: Camera angles, lighting, occlusion, background noise, and model bias can degrade results.
- Security: The local development server binds to 127.0.0.1. Do not expose it to the public internet without authentication, encryption, access controls, and a security review.
- Data stewardship: Avoid uploading sensitive photos, recordings, transcripts, or real patient records to public GitHub repositories.
🛣️ Future Improvements
Potential enhancements include improved object-class coverage with a custom detector, more robust on-device inference, usability and accessibility evaluation with target users, stronger data lifecycle controls, and quantitative performance evaluation across all memory domains.
👩‍💻 Author
Sakshi Chabukswar
GitHub: @SakshiCC
Repository: SakshiCC/MemorAi
MemorAi aims to make observed everyday information easier to recall through evidence-grounded, multimodal AI.
