🧠 Virtual Grant Officer — Streamlined AI Assistant
A custom Python-based conversational agent for answering university research funding and grant-related queries. It uses NVIDIA’s large language models via the langchain_nvidia package and orchestrates stateful interactions with langgraph.

🔧 Quick Setup
🛠 Environment Setup
Clone this repo or copy the relevant files to your local machine.

Ensure you're using Python 3.9+.

Install required packages:

bash
Copy
Edit
pip install numpy==1.24.3 streamlit-extras langchain langgraph python-dotenv openai sentence-transformers
✅ No .env file or /data, /components folders are required for this app.

🔑 API Keys
The app will prompt you for your NVIDIA_API_KEY during runtime.

Ensure your API key provides access to:

Meta LLaMA-3.3-70B-Instruct

Nemotron-3-8B-Super-Instruct

🚀 Running the App
You can run the app in two ways:

1. Backend-Only Mode
Run the backend pipeline to test logic without a frontend:

bash
Copy
Edit
python app.py
2. Full UI Mode (Recommended)
Run the full interactive UI with Streamlit:

bash
Copy
Edit
streamlit run virtualgco_app_ui.py
🧩 Core Logic
State management is handled with langgraph's StateGraph.

Custom nodes include:

check_relevance_node: Filters out irrelevant questions.

build_prompt_node: Merges chat history with retrieved documents to build the LLM prompt.

generate_answer_node: Optionally refines responses for clarity and alignment.

Document Retrieval:

JSON documents are loaded dynamically using load_documents_from_json(path).

Semantic search is powered by embeddings via sentence-transformers.

LLM Integration:

Answers are generated through NVIDIA-hosted LLMs.

Prompts are tuned for research compliance, institutional clarity, and grant-specific terminology.

📂 Data Handling
Input documents should be in JSON format with a "Scraped Text" key.

No fixed /data folder needed — you specify the folder path at runtime.

📌 Assumptions
Frontend UI is handled via Streamlit (virtualgco_app_ui.py).

No API keys or secrets are stored in plaintext; credentials are securely input at runtime.

Error handling ensures graceful failure on API rate limits or model timeouts.
