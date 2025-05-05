#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:02:13 2025

@author: adityavinodh
"""
#install numpy==1.24.3
#install streamlit-extras
import getpass
import os
from langchain_nvidia import ChatNVIDIA
from langgraph.checkpoint.memory import MemorySaver



def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("NVIDIA_API_KEY")

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    


def load_documents_from_json(directory_path):
    """Load documents from a directory containing JSON files."""
    documents = []
    for file_path in Path(directory_path).glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = json.load(f)
                text = content.get("Scraped Text", "")
                # Decode any escape sequences like \u2013 (en dash) in the text
                text = text.encode('utf-8').decode('unicode_escape')
                metadata = {
                    "source": content.get("URL location", ""),
                    "last_modified": content.get("Last modification date", ""),
                    "change_frequency": content.get("Change frequency", ""),
                    "priority": content.get("Priority", "")
                }
                if text.strip():  # Only include documents with non-empty text
                    # Generate a unique ID for each document, e.g., using the file name
                    document_id = str(file_path.name)  # Using file name as the ID
                    documents.append(Document(page_content=text, metadata=metadata, id=document_id))
            except Exception as e:
                print(f"Failed to load {file_path.name}: {e}")
                continue  # Skip this file and move to the next one
    return documents

def load_and_search_docs(user_input, k=3):
    global vectorstore, embeddings

    # Use the retriever with the full search space and a specified `k`
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    docs = retriever.get_relevant_documents(user_input)

    for i, doc in enumerate(docs):
        metadata_str = doc.metadata if hasattr(doc, 'metadata') else "No metadata"

    return {f"doc_{i}": d.page_content for i, d in enumerate(docs)}



from langchain_core.messages import AIMessage

llm = ChatNVIDIA(base_url="https://integrate.api.nvidia.com/v1", model="meta/llama-3.3-70b-instruct")
def check_relevance_node(state):
    user_question = state["user_input"]

    prompt = f"""
You are an intelligent assistant helping university grant officers.

Your job is to check if this question is appropriate for a Virtual Grant Officer.

The Virtual Grant Officer *can* handle:
- Research grants
- Funding opportunities
- Proposal writing
- Deadlines
- Research compliance
- Sponsored programs at universities (e.g., Cornell)
- Follow-up questions
- Simple greetings (e.g., "thanks", "hello")
- Clarifications or continuation of earlier queries

The Virtual Grant Officer *cannot* handle:
- Personal or emotional support questions
- Counseling
- Irrelevant general knowledge
- Political, religious, or health-related questions
- Casual jokes, stories, or off-topic conversation

Here is the question:
"{user_question}"

Respond with one word only:
"YES" – if the question is relevant, appropriate, or harmless follow-up/comment.
"NO" – only if it is clearly inappropriate, personal, or completely unrelated.

If unsure, err on the side of "YES".
"""

    try:
        print(prompt)
        resp = llm.invoke(prompt)
        print(resp)
        answer = resp.content.strip().split("\n")[-1].strip().upper()

        if answer == "YES":
            state["is_relevant"] = True
        else:
            state["is_relevant"] = False
            state["answer"] = "⚠️ This assistant is focused on research funding and university grants. For unrelated or personal questions, please use a different service."

    except Exception as e:
        print("[Error] Relevance check failed:", e)
        state["is_relevant"] = False
        state["answer"] = "⚠️ Something went wrong evaluating your question. Please try again with a funding-related topic."

    return state


from langchain_core.messages import AIMessage

llm = ChatNVIDIA(base_url="https://integrate.api.nvidia.com/v1", model="nvidia/llama-3.3-nemotron-super-49b-v1")
def build_prompt_node(state):
    user_question = state["user_input"]
    relevant_docs = state.get("docs", {})
    history = state.get("message_history", [])

    if not relevant_docs:
        print("No relevant documents found. Skipping final prompt generation.")
        state["final_response"] = "Sorry, I couldn't find a relevant answer in the provided documents."
        return state

    # Format chat history into a string
    chat_history = ""
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            chat_history += f"User: {content}\n"
        elif role == "assistant":
            chat_history += f"AI: {content}\n"

    combined_context = "\n\n---\n\n".join(
        f"Document {i+1}:\n{content}" for i, content in enumerate(relevant_docs.values())
    )

    final_prompt = f"""
You are an intelligent assistant.

The following is a conversation between a user and you:

{chat_history}
User: {user_question}

Here are relevant pieces of context from various documents:
{combined_context}

Use the above content and the conversation so far to provide a complete and helpful answer to the user's latest question.
Be as specific and concise as possible. If the content is not enough, say so clearly, but first try to reference previous conversation history.


Here are some examples of how you should respond to questions. Respond in the same format and keep answers a short as this:

---

Q: *How do I know if I need a subaward?*  
A: If a portion of the programmatic work will be performed by another institution/entity, a subaward may be necessary. Subaward requests are submitted in Cornell’s Research Administration Support System (RASS) by the unit research administrator. The Subaward Team then manages the process. Only the Office of Sponsored Programs may sign subawards.

---

Q: *When should I determine if there will be a subaward?*  
A: This is usually determined at the proposal submission stage. To help decide whether the collaboration is a subaward or contractor, refer to the Subrecipient vs Contractor Checklist.  
**Link:** [https://researchservices.cornell.edu/sites/default/files/2022-11/Subrecipient%20vs%20Contractor%20Checklist.pdf](https://researchservices.cornell.edu/sites/default/files/2022-11/Subrecipient%20vs%20Contractor%20Checklist.pdf)

---

Q: *What is the difference between a vendor/procurement agreement and a subaward?*  
A: Vendor services require a procurement agreement through Cornell’s Procurement and Payment Services (email: procurement@cornell.edu). Subawards, on the other hand, involve substantive programmatic work and are managed by the Office of Sponsored Programs. Subawards flow down most terms from the parent award.

---

Q: *What if I determine I need a subaward after the grant has been awarded?*  
A: You may need prior sponsor approval and possibly rebudgeting. Your unit research administrator will collect documentation and submit a request via RASS. The Grant and Contract Officer (GCO) will seek sponsor approval.

---

Q: *What information do I need to provide for the subaward?*  
A: The sponsor needs to approve the collaborator’s scope of work and budget. Additional documents (e.g., biosketch, current/pending support) may also be required. Coordinate with your GCO.

---

Q: *I received an award, will my subaward be set up automatically?*  
A: No. The unit research administrator must initiate a new subaward request in RASS.  
**Link for guidance:** [https://guide.rass.cornell.edu/sponsored-projects/initiate-a-subaward](https://guide.rass.cornell.edu/sponsored-projects/initiate-a-subaward)

---

Q: *How is a subaward set up?*  
A: It is initiated in RASS by the unit research administrator. After review and execution by both parties, the Subaward Team distributes it in RASS and Sponsored Financial Services sets up a subaccount.

---

Q: *My grant received an extension or more funds. Will my collaborator's subaward be extended automatically?*  
A: No. A modification request must be submitted in RASS.  
**Link for guidance:** [https://guide.rass.cornell.edu/sponsored-projects/submit-change-requests/](https://guide.rass.cornell.edu/sponsored-projects/submit-change-requests/)
    """

    try:
        response = llm.invoke(final_prompt)
        final_answer = response.content.strip()
        state["final_response"] = final_answer
    except Exception as e:
        print("[Error] Failed to generate final response:", e)
        state["final_response"] = "An error occurred while generating the answer."
    
    print(state['final_response'])
    return state


def generate_answer_node(state):
    final_answer = state.get("final_response", None)
    user_question = state.get("user_input", "")
    relevant_docs = state.get("docs", {})
    

    if not final_answer:
        print("No final response available to return.")
        state["answer"] = "Sorry, I couldn't generate an answer at this time."
        return state

    # Prompt to revise the answer for clarity, conciseness, and relevance
    refinement_prompt = f"""
You are an expert assistant that ensures responses are high quality.

Here is the original user question:
"{user_question}"

Here is a draft answer:
"{final_answer}"

Here were the documents retrieved for reference:
"{relevant_docs}"


Please revise the answer to:
- Directly and clearly answer the user's question.
- Be as concise and informative as possible.
- Remove any irrelevant or inaccurate information.
- Ensure there are no broken or non-functional links (remove or fix them).

Output ONLY the final version of the answer without repeating the user's question. If no revisions are required and the output is ready as a professional virtual assistant's response, then leave as is. 
"""

    try:
        improved_answer = llm.invoke(refinement_prompt).content.strip()
        state["answer"] = improved_answer
    except Exception as e:
        print("[Error] Failed to refine the answer:", e)
        state["answer"] = "An error occurred while refining the answer."

    return state

from langgraph.graph import StateGraph

memory = MemorySaver()  # This is the correct in-memory checkpointer

builder = StateGraph(State)


builder.add_node("get_initial_docs", lambda state: {**state, "docs": load_and_search_docs(state["user_input"])})

builder.add_node("check_relevance", check_relevance_node)
builder.add_node("build_prompt", build_prompt_node)
builder.add_node("generate_answer", generate_answer_node)

builder.set_entry_point("get_initial_docs")
builder.add_edge("get_initial_docs", "check_relevance")
builder.add_edge("check_relevance", "build_prompt")
builder.add_edge("build_prompt", "generate_answer")

builder.set_finish_point("generate_answer")
graph = builder.compile(checkpointer=memory)


from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# import nltk
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.docstore.document import Document
# from nltk.tokenize import sent_tokenize

# # # Download punkt tokenizer once
# # nltk.download('punkt')

# # Define global variables for embeddings and model names
# vectorstore = None
# embeddings = None
# model_name="sentence-transformers/all-MiniLM-L6-v2"
# vectorDB_name = "raiss_site"

# def initialize_embeddings():
#     """Initialize the embedding model with remote code trust enabled."""
#     global embeddings
#     embeddings = HuggingFaceEmbeddings()

# def create_vector_database(chunked_documents):
#     """Create a Vector Database Base (VDB) from pre-processed document chunks."""
#     global vectorstore
#     try:
#         # Ensure each chunked document is properly structured
#         wrapped_documents = []
#         for doc in chunked_documents:
#             try:
#                 # Directly access page_content and metadata from Document
#                 if not isinstance(doc, Document):
#                     print(f"Skipping document with id {doc.id} as it is not a Document object.")
#                     continue  # Skip this document if it is not a Document object
                
#                 # Append valid Document objects
#                 wrapped_documents.append(doc)
#             except Exception as e:
#                 print(f"Error processing document with id {doc.id}: {str(e)}")
#                 continue  # Skip this document if an error occurs

#         # Create FAISS index and vector embeddings for chunks of data
#         vectorstore = FAISS.from_documents(wrapped_documents, embeddings)

#         # Save the vector database index locally
#         vectorstore.save_local(vectorDB_name)

#         return "Successfully created the vector database from chunked documents."
#     except Exception as e:
#         return f"Error creating vector database: {str(e)}"

# def semantic_chunk_document(text, metadata=None, max_chunk_words=100):
#     if metadata is None:
#         metadata = {}

#     sentences = sent_tokenize(text)
#     chunks = []
#     current_chunk = ""
#     current_word_count = 0

#     for sentence in sentences:
#         words = sentence.split()
#         if current_word_count + len(words) > max_chunk_words:
#             chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata))
#             current_chunk = sentence
#             current_word_count = len(words)
#         else:
#             current_chunk += " " + sentence
#             current_word_count += len(words)

#     if current_chunk:
#         chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata))

#     return chunks

# def semantic_chunk_all_documents(documents, max_chunk_words=100):
#     all_chunks = []
#     for doc in documents:
#         chunks = semantic_chunk_document(doc.page_content, metadata=doc.metadata, max_chunk_words=max_chunk_words)
#         all_chunks.extend(chunks)
#     return all_chunks


# # Step 1: Load documents from a local directory
# documents_directory = "/Users/adityavinodh/Documents/sp25mps/raiss_ai/scraped_data_json"
# raw_documents = load_documents_from_json(documents_directory)  

# # Step 2: Apply semantic chunking
# chunked_documents = semantic_chunk_all_documents(raw_documents, max_chunk_words=100)

# # Step 3: Initialize embeddings model
# initialize_embeddings()

# # Step 4: Create the vector database
# result_message = create_vector_database(chunked_documents)

# print(result_message)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st

# === Page Config ===
st.set_page_config(page_title="Virtual Grant Officer", layout="wide")

# === Custom Styles + Fonts ===
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Baloo+2&display=swap');        
        
        h1 {
            color: #0B2973 !important;
            font-size: 28px !important;
            font-family: 'Baloo 2', sans-serif !important;
            text-align: center;
        }
        p {
            color: #8A9DCB;
            font-family: 'Baloo 2', sans-serif;
            font-size: 14px !important;
            text-align: center;
        }

        .prompt-title {
            font-family: 'Baloo 2', sans-serif;
            font-size: 20px;
            font-weight: 400;
            color: #0B2973;
            margin-bottom: 10px;
            margin-top: 60px !important;
        }
        button {
            background-color: #ECF2FD;
            border: 1px solid #0B2973 !important;
            border-radius: 999px !important;
            padding: 0.5em 1em;
            font-family: 'Baloo 2', sans-serif !important;
            cursor: pointer;
            margin: 10px;
            width: auto;
            display: inline-block;
        }
        button p {
            color: #0B2973;
            font-size: 14px !important;
        }
        button:hover {
            background-color: #0B2973 !important;
        }
        button:hover p {
            color: white;
        }

        .custom-label {
            font-family: 'Baloo 2', sans-serif !important;
            font-size: 20px !important;
            font-weight: 400 !important;
            color: #0B2973 !important;
            margin-bottom: 0 !important;
            margin-top: 20px !important;
        }
        input[type="text"] {
            background: white;
            font-family: 'Baloo 2', sans-serif !important;
            font-size: 16px;
            color: #0B2973;
            margin-top: 0 !important;
        }
        
        [data-testid=stSidebar] h1 {
            text-align: left;
            color: #8A9DCB !important;
            font-size: 16px !important;
        }
        [data-testid=stSidebar] p {
            font-family: 'Baloo 2', sans-serif !important;
            font-size: 14px !important;
            text-align: left;
            color: white;
            margin-bottom: 10px;
        }
        [data-testid=stSidebar] button {
            background-color: #0B2971;
            padding: 0.5em 1em;
            color: white !important;
            font-size: 16px !important;
            font-family: 'Baloo 2', sans-serif !important;
            cursor: pointer;
            min-width: 280px;
            display: block;
            border-radius: 10px !important;
            margin-top: 220px;
        }
    </style>
""", unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.title("Chat History")
for msg in st.session_state.get("messages", []):
    if msg["role"] == "user":
        st.sidebar.write("🧑‍💻 " + msg["content"])
    else:
        st.sidebar.write("🤖 " + msg["content"][:80] + "...")

if st.sidebar.button("New Chat"):
    st.session_state.messages = []

# === Logo and Header ===
col1, col2, col3 = st.columns([4, 2, 4])
with col2:
    st.image("streamlit-app/logo.png", width=200)

st.markdown('<h1>Chat with Grant & Contract Officer</h1>', unsafe_allow_html=True)
st.markdown('<p>Our Chatbot transforms your content, tailoring<br>it to your specific needs and preferences,<br>making customization a breeze.</p>', unsafe_allow_html=True)

# === Prompt Suggestions ===
prompts = [
    "Make it more descriptive and vivid",
    "Condense it and make it concise",
    "Add some flair and creativity",
    "Simplify the language for clarity",
    "Inject a sense of urgency and excitement",
    "Personalize it for our target audience"
]

if "custom_input" not in st.session_state:
    st.session_state["custom_input"] = ""

def set_prompt(prompt_text):
    st.session_state["custom_input"] = prompt_text

st.markdown('<div class="prompt-title">Prompt Suggestions For You</div>', unsafe_allow_html=True)
cols = st.columns(3)
for i, prompt in enumerate(prompts):
    with cols[i % 3]:
        st.button(prompt, key=f"btn_{i}", on_click=set_prompt, args=(prompt,))

# Initialize embeddings
@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings()

embeddings = initialize_embeddings()

# Load FAISS vectorstore
@st.cache_resource
def load_vector_database():
    return FAISS.load_local(
        "/Users/adityavinodh/Documents/sp25mps/raiss_site_updated", #Replace with path to vectorstore
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vector_database()


# === Chatbot Backend Logic ===
def gco_chatbot(user_input):
    message_history = st.session_state.messages.copy()
    message_history.append({"role": "user", "content": user_input})

    state = {
        "user_input": user_input,
        "message_history": message_history
    }

    # REPLACE with your own logic:
    state = check_relevance_node(state)
    if not state.get("is_relevant", False):
        return state["answer"]

    state["docs"] = load_and_search_docs(user_input)
    state = build_prompt_node(state)
    state = generate_answer_node(state)

    docs = state.get("docs", {})
    docs_preview = "\n\n".join(
        [f"🔹 **{title}**: {content[:300]}{'...' if len(content) > 300 else ''}"
         for title, content in docs.items()]
    )

    answer = state.get("answer", "⚠️ No answer generated.")
    return f"{answer}" 

# === Chat Area ===
st.markdown('<div class="custom-label">Ask me anything</div>', unsafe_allow_html=True)
user_input = st.text_input("", value=st.session_state["custom_input"], key="custom_input")

if "messages" not in st.session_state:
    st.session_state.messages = []

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = gco_chatbot(user_input)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# === Feedback ===
st.markdown("---")
st.markdown("### Was this response helpful?")
col1, col2 = st.columns(2)
with col1:
    if st.button("👍 Yes"):
        st.success("Thank you for your feedback!")
with col2:
    if st.button("👎 No"):
        st.error("Sorry! Please try asking in a different way.")