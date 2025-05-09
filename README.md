# 🧠 Virtual Grant Officer — Streamlined AI Assistant

A custom Python-based conversational agent designed for answering university research funding and grant-related queries. It utilizes NVIDIA's large language models through the `langchain_nvidia` package and orchestrates stateful interactions with `langgraph`.

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
   ```

✅ **Note:** No .env file or /data, /components folders are required for this app.

## 🔑 API Keys

The app will prompt you for your NVIDIA_API_KEY during runtime.

**Important Note:** You need to input your own NVIDIA NIMS Key. If you don't have one, you can request it by contacting me at av364@cornell.edu, or change the model call configuration as mentioned below to use the OpenAI API or a different key if preferred.

## 📋 Running the App

1. Change directory to wherever the folder is downloaded:
   ```bash
   cd path/to/downloaded/folder
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run virtualgco_app_ui.py
   ```

3. **Critical:** When the app launches in your web browser, it will ask for the API key again via the terminal. Make sure the terminal window is open and enter your NVIDIA API key. If you don't enter the key in the terminal, model calls cannot be made.

4. Once the API key is entered, you can start interacting with the Virtual Grant Officer through the web interface.

## 🤖 Model Configuration

Currently, the app is configured to use NVIDIA's instruction-tuned model:

```python
model = "nvidia/llama-3.3-nemotron-super-49b-v1"
```

This model call can easily be changed by modifying the LLM configuration in the code. If you prefer to use OpenAI's API or a different NVIDIA model, you can update the model configuration and adjust the prompt format accordingly.

## 🧩 System Architecture

The application uses a graph-based architecture with the following key components:

### Core Nodes:
1. **check_relevance_node**: Filters out irrelevant questions by determining if the query is appropriate for a Virtual Grant Officer, returning a Yes/No output.

2. **build_prompt_node**: Merges chat history with retrieved documents to build the LLM prompt, combining the question with relevant context and sample format for the expected response.

3. **generate_answer_node**: Revises responses for clarity, conciseness, and relevance.

### Key Features:
- Few-shot prompting using validation Q&A examples
- Query relevance checking (guardrails & safety)
- Irrelevant query filtering before prompt building
- Contextual history leveraging past conversation messages when reference is unclear

### Document Retrieval:
- Documents to build the database are currently stored as separate JSON files in the `New_Data.zip` file.
- The created vector database is stored in the `raiss_site_updated` directory, including the appropriate pickle and index files.
- The older version of the database is stored in the `raiss_site` directory, which contains the previous scraper and only text (without table elements).

### LLM Integration:
- Answers are generated using NVIDIA-hosted LLMs.
- Prompts are fine-tuned for research compliance, institutional clarity, and grant-specific terminology.

## 📂 Repository Structure

The repository contains the following important files and directories:

- `virtualgco_app_ui.py`: Main application file to run with Streamlit
- `raiss_site_updated/`: Directory containing the current vector database with pickle and index files
- `raiss_site/`: Directory containing the older version of the vector database (text only, no table elements)
- `New_Data.zip`: ZIP file containing JSON files used to build the database
- `RAISS_sitemap.csv`: List of sitemap URLs
- `scraped_data_raw.zip` and `scraped_data_json.zip`: Compilations of JSON files used to build the database
- `streamlit-app/`: Directory containing Streamlit configuration files and logo files required for the app
- `blocked_sites.csv`: List of all SSO-blocked non-public-facing sites that the scraper couldn't access
- `webscraper.ipynb`: Jupyter notebook containing the web scraper code

## 📌 Technical Details

- The frontend UI is managed via Streamlit (`virtualgco_app_ui.py`).
- State management is handled by langgraph's StateGraph.
- Document retrieval uses FAISS vectorstore and HuggingFaceEmbeddings.
- No API keys or secrets are stored in plaintext. Credentials are securely input during runtime.
- Error handling ensures graceful failure in case of API rate limits or model timeouts.

## ❓ Support

If you need help with the NVIDIA API key or have other questions about this application, please contact av364@cornell.edu.
