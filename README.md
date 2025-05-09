# 🧠 Virtual Grant Officer — Streamlined AI Assistant

A custom Python-based conversational agent designed for answering university research funding and grant-related queries. It utilizes NVIDIA’s large language models through the `langchain_nvidia` package and orchestrates stateful interactions with `langgraph`.

---

## 🔧 Quick Setup

### 🛠 Environment Setup

1. **Clone or Copy Files:**  
   Clone this repository or copy the relevant files to your local machine.

2. **Ensure Python Version:**  
   Make sure you're using Python 3.9 or higher.

3. **Install Required Packages:**  
   Install the necessary dependencies by running the following command:

   ```bash
   pip install numpy==1.24.3 streamlit-extras langchain langgraph python-dotenv openai sentence-transformers

✅ Note: No .env file or /data, /components folders are required for this app.

🔑 API Keys
The app will prompt you for your NVIDIA_API_KEY during runtime.

Important Note: You need to input your own NVIDIA NIMS Key. If you don’t have one, you can request it by contacting me at av364@cornell.edu, or change the model call configuration as mentioned below to use the OpenAI API or a different key if preferred.

Model Configuration
Currently, the app is configured to use NVIDIA’s instruction-tuned model:

Model: nvidia/llama-3.3-nemotron-super-49b-v1

This model call can easily be changed by modifying the LLM configuration. You can swap it out for another model by calling OpenAI's API or by specifying a different NVIDIA model using the format:

python
Copy
Edit
model = "<desired_model_name>"
Replace "<desired_model_name>" with the model identifier you prefer.

🚀 Running the App
You can run the app in two modes:

1. Backend-Only Mode
Run the backend pipeline to test the logic without a frontend:

bash
Copy
Edit
python app.py
2. Full UI Mode (Recommended)
Run the full interactive UI using Streamlit:

bash
Copy
Edit
streamlit run virtualgco_app_ui.py
🧩 Core Logic
State management is handled by langgraph's StateGraph. The core logic involves the following custom nodes:

check_relevance_node: Filters out irrelevant questions.

build_prompt_node: Merges chat history with retrieved documents to build the LLM prompt.

generate_answer_node: Optionally refines responses for clarity and alignment.

Document Retrieval:
Documents to build the database are currently stored as separate JSON files in the New_Data.zip file.

The created vector database is stored in the raiss_site_updated directory, including the appropriate pickle and index files.

The older version of the database is stored in the raiss_site directory, which contains the previous scraper and only text (without table elements).

LLM Integration:
Answers are generated using NVIDIA-hosted LLMs.

Prompts are fine-tuned for research compliance, institutional clarity, and grant-specific terminology.

📂 Data Handling
RAISS_sitemap.csv contains the list of sitemap URLs.

scraped_data_raw.zip and scraped_data_json.zip are compilations of the JSON files used to build the database.

The streamlit-app folder contains the Streamlit configuration files and logo files required for the app.

blocked_sites.csv lists all SSO-blocked non-public-facing sites (sites that the scraper couldn't access), for reference later.

📌 Assumptions
The frontend UI is managed via Streamlit (virtualgco_app_ui.py).

No API keys or secrets are stored in plaintext. Credentials are securely input during runtime.

Error handling ensures graceful failure in case of API rate limits or model timeouts.

❗ Important Notes on API Key Setup
When you first run the Python script, it will ask you to set up an environment key for the NVIDIA API call. You will need to input your NVIDIA NIMS Key.

If you don't have an NVIDIA API key, contact me at av364@cornell.edu to obtain it, or modify the model calls to use another service (such as OpenAI).

After setting the API key, the app will launch in your web browser. When you do this, ensure the terminal window is open because the app will ask for the API key again via the terminal. You must enter the key here for the model calls to be made successfully. If the terminal isn't open, model calls will not work.

