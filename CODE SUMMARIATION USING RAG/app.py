import streamlit as st
import subprocess
import json
import os
import zipfile


#from PyPDF2 import PdfReader
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torch.nn.functional as F
import re
import faiss
from tqdm import tqdm
import pickle as pkl
from utils import get_embeddings


#for chatting with the code
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import faiss
import pickle as pkl
import gradio as gr
from utils import get_embeddings
from llm import LLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

path = 'embeddings'
chunk_size = 3
overlap = 0
split_char = '.'

folder_name = 'embeddings'
name = 'code_embeddings' #file name




tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
embeddings_model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
index = faiss.IndexFlatL2(embeddings_model.config.hidden_size)
embeddings_model.to(device)
embeddings_model.eval()
    


if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists.")

@torch.no_grad()
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_answer(query, k,text_info,chatbot):
    query = query
    query_embedding = get_embeddings([query], tokenizer, embeddings_model)
    scores, text_idx = index.search(query_embedding, k)
    text_idx = text_idx.flatten()
    info = '\n--------\n'.join(np.array(text_info)[text_idx])
    info = info.replace('passage: ', '').strip()
    ans = chatbot.get_response(query, info)
    return ans + '\n-------------Information: \n' + info + '\n---------\n'



@torch.no_grad()
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]






class Chuncker:
    def __init__(self, chunck_size=3, chunk_overlap=1, split_char='.'):
        self.chunck_size = chunck_size
        self.chunk_overlap = chunk_overlap
        self.split_char = split_char

    def get_chunks(self, text):
        all_words = text.split(self.split_char)
        chunks = []
        for i in range(0, len(all_words), self.chunck_size - self.chunk_overlap):
            chunks.append('passage: ' + ' '.join(all_words[i:i + self.chunck_size]))
        return chunks




# Set Streamlit theme to dark
st.markdown(
    """
    <style>
    .reportview-container {
        background: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextInput>div>div>input {
        color: #000000;
    }
    .st-bm {
        background-color: #2E2E2E !important;
        color: #FFFFFF !important;
    }
    .st-bq {
        border-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)
command = 'ollama run codellama'

def aggregate_js_files(folder_path):
    if not os.path.exists('aggregated_text'):
        os.makedirs('aggregated_text')
    
    with open(os.path.join('aggregated_text', 'aggregated_code.txt'), 'w') as aggregated_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.js','.ts','.json')):
                    aggregated_file.write(f"\n\n// File: {file}\n\n")
                    with open(os.path.join(root, file), 'r') as js_file:
                        js_content = js_file.read()
                        aggregated_file.write(js_content)

def execute_command_with_input(input_string):
    try:
        proc = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        proc.stdin.write(input_string + '\n')
        proc.stdin.flush()
        output, error = proc.communicate()
        if proc.returncode != 0:
            return f"Error: {error}"
        return output
    except Exception as e:
        return f"Error: {e}"
    
generate_graph = False
chat_button = False

def main():
    
    global generate_graph
    global chat_button
    global command
    extract_dir = 'extracted'
    st.title("Code Summary Generator")

    uploaded_zip = st.file_uploader("Upload Zip File Containing Angular Code", type="zip")
    if uploaded_zip is not None:
        with open(uploaded_zip.name, "wb") as f:
            f.write(uploaded_zip.getvalue())

        # Create a directory to extract the contents of the zip file
        extract_dir = os.path.join(os.getcwd(), "extracted")
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)

        # Extract the zip file into the "extracted" directory
        with zipfile.ZipFile(uploaded_zip.name, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    embeddings_model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    extracted_folder_path = extract_dir if os.path.exists(extract_dir) else None
    index = faiss.IndexFlatL2(embeddings_model.config.hidden_size)

    file_contents = ''
    if st.button("Pass to Model") and extracted_folder_path:
        aggregate_js_files(extracted_folder_path)

        with open(os.path.join('aggregated_text', 'aggregated_code.txt'), 'r') as aggregated_file:
            file_contents = aggregated_file.read()

        st.subheader("Uploaded Project Contents are as follows: ")
        st.text_area("", value=file_contents, height=200, key="uploaded_file_contents")
        
        chunker = Chuncker(chunck_size=chunk_size, chunk_overlap=overlap, split_char=split_char)

        index = faiss.IndexFlatL2(embeddings_model.config.hidden_size)
        #make index of the text file
        text_info = []


        text = open('aggregated_text/aggregated_code.txt', 'r').read()

        text_chunks = chunker.get_chunks(text)
        text_info.extend([x for x in text_chunks])
        
        
        # The following is probably slow, should batch this
        name = "code_embeddings"

        index_file_path = f'embeddings/{name}.index'
        text_info_file_path = f'embeddings/{name}_text_info.pkl'

        if not (os.path.exists(index_file_path) or os.path.exists(text_info_file_path)):

            text_embeddings = np.array([get_embeddings([x], tokenizer, embeddings_model) for x in tqdm(text_chunks)]).reshape(len(text_chunks), -1)
            index.add(text_embeddings)


            faiss.write_index(index, f'embeddings/{name}.index')

            pkl.dump(text_info, open(f'embeddings/{name}_text_info.pkl', 'wb'))

        generate_graph = True
        chat_button = True

        
        #===================================================================================
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    embeddings_model.to(device)
    embeddings_model.eval()
    pass

    chatbot = LLM()
    name = 'code_embeddings' #file name
    index = faiss.read_index(f'embeddings/{name}.index')
    text_info = pkl.load(open(f'embeddings/{name}_text_info.pkl', 'rb'))

            #+----------------------------------------------------------------------------------
        #CODE FOR GENERTATING GRAPH USING OLLAMA   
    #if generate_graph == True:
    if st.button("Generate graph"):
        st.subheader("GRAPH GENERATED: ")

        question = 'generate graph data output, Json data output for this angular code' 

        #response = get_answer(file_contents,10)
        response = chatbot.get_response(question, file_contents)

        st.text_area("", value=response, height=500, key="response_text_area")

        try:
            #graph_data = json.loads(response)
            with open("graph_data.txt", "w") as json_file:
                json.dump(graph_data, json_file)
        except Exception as e:
            st.error(f"Error saving graph data: {e}")

        #+----------------------------------------------------------------------------------
        #CODE FOR CHATING WITH PROJECT USING RAG       


        st.subheader("Query Chat")
        user_query = st.text_input("Ask a question:")
        if user_query:
            response = get_answer(user_query,7,text_info,chatbot)
            st.text_area("", value=response, height=700, key="query_response_text_area")

if __name__ == "__main__":
    main()

