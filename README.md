# ElderMind AI 🧠

ElderMind AI is an AI assistant application designed to aid cognitive support, memory search, facial recognition, and event timeline tracking.

## Folder Structure

```text
ElderMind-AI/
│
├── app/
│   └── nicegui_app.py
│
├── database/
│   ├── db_manager.py
│   └── eldermind.db
│
├── vision/
│   ├── object_detection.py
│   └── location_mapper.py
│
├── timeline/
│   └── timeline_manager.py
│
├── rag/
│   ├── embeddings.py
│   └── memory_search.py
│
├── llm/
│   └── chat.py
│
├── speech/
│   └── whisper_module.py
│
├── face/
│   └── face_recognition.py
│
├── dataset/
│
├── models/
│
├── tests/
│
├── requirements.txt
│
├── config.py
│
└── README.md
```

## Setup & Running

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch Streamlit UI:
   ```bash
   streamlit run app/streamlit_app.py
   ```
