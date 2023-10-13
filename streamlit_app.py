import os
from hashlib import blake2b
from tempfile import NamedTemporaryFile

import dotenv

dotenv.load_dotenv(override=True)

import streamlit as st
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from document_qa_engine import DocumentQAEngine

if 'rqa' not in st.session_state:
    st.session_state['rqa'] = None

if 'openai_key' not in st.session_state:
    st.session_state['openai_key'] = False

if 'doc_id' not in st.session_state:
    st.session_state['doc_id'] = None

if 'loaded_embeddings' not in st.session_state:
    st.session_state['loaded_embeddings'] = None

if 'hash' not in st.session_state:
    st.session_state['hash'] = None

if 'git_rev' not in st.session_state:
    st.session_state['git_rev'] = "unknown"
    if os.path.exists("revision.txt"):
        with open("revision.txt", 'r') as fr:
            from_file = fr.read()
            st.session_state['git_rev'] = from_file if len(from_file) > 0 else "unknown"

if "messages" not in st.session_state:
    st.session_state.messages = []


def new_file():
    st.session_state['loaded_embeddings'] = None
    st.session_state['doc_id'] = None


@st.cache_resource
def init_qa(openai_api_key):
    chat = PromptLayerChatOpenAI(model_name="gpt-3.5-turbo",
                                 temperature=0,
                                 return_pl_id=True,
                                 pl_tags=["streamlit", "chatgpt"],
                                 openai_api_key=openai_api_key)
    # chat = ChatOpenAI(model_name="gpt-3.5-turbo",
    #                   temperature=0)
    return DocumentQAEngine(chat, OpenAIEmbeddings(openai_api_key=openai_api_key), grobid_url=os.environ['GROBID_URL'])


def get_file_hash(fname):
    hash_md5 = blake2b()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def play_old_messages():
    if st.session_state['messages']:
        for message in st.session_state['messages']:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(message['content'])
            elif message['role'] == 'assistant':
                with st.chat_message("assistant"):
                    if mode == "LLM":
                        st.markdown(message['content'])
                    else:
                        st.write(message['content'])


has_openai_api_key = False
if not st.session_state['openai_key']:
    openai_api_key = st.sidebar.text_input('OpenAI API Key')
    if openai_api_key:
        st.session_state['openai_key'] = has_openai_api_key = True
        st.session_state['rqa'] = init_qa(openai_api_key)
else:
    has_openai_api_key = st.session_state['openai_key']

st.title("📝 Document insight Q&A")
st.subheader("Upload a PDF document, ask questions, get insights.")

upload_col, radio_col, context_col = st.columns([7, 2, 2])
with upload_col:
    uploaded_file = st.file_uploader("Upload an article", type=("pdf", "txt"), on_change=new_file,
                                     disabled=not has_openai_api_key,
                                     help="The file will be uploaded to Grobid, extracted the text and calculated "
                                          "embeddings of each paragraph which are then stored to a Db for be picked "
                                          "to answer specific questions. ")
with radio_col:
    mode = st.radio("Query mode", ("LLM", "Embeddings"), disabled=not uploaded_file, index=0,
                    help="LLM will respond the question, Embedding will show the "
                         "paragraphs relevant to the question in the paper.")
with context_col:
    context_size = st.slider("Context size", 3, 10, value=4,
                             help="Number of paragraphs to consider when answering a question",
                             disabled=not uploaded_file)

question = st.chat_input(
    "Ask something about the article",
    # placeholder="Can you give me a short summary?",
    disabled=not uploaded_file
)

with st.sidebar:
    st.header("Documentation")
    st.write("""To upload the PDF file, click on the designated button and select the file from your device.""")

    st.write(
        """After uploading, please wait for the PDF to be processed. You will see a spinner or loading indicator while the processing is in progress. Once the spinner stops, you can proceed to ask your questions.""")

    st.markdown("**Revision number**: [" + st.session_state[
        'git_rev'] + "](https://github.com/lfoppiano/grobid-magneto/commit/" + st.session_state['git_rev'] + ")")

    st.header("Query mode (Advanced use)")
    st.write(
        """By default, the mode is set to LLM (Language Model) which enables question/answering. You can directly ask questions related to the PDF content, and the system will provide relevant answers.""")

    st.write(
        """If you switch the mode to "Embedding," the system will return specific paragraphs from the document that are semantically similar to your query. This mode focuses on providing relevant excerpts rather than answering specific questions.""")

if uploaded_file and not st.session_state.loaded_embeddings:
    with st.spinner('Reading file, calling Grobid, and creating memory embeddings...'):
        binary = uploaded_file.getvalue()
        tmp_file = NamedTemporaryFile()
        tmp_file.write(bytearray(binary))
        # hash = get_file_hash(tmp_file.name)[:10]
        st.session_state['doc_id'] = hash = st.session_state['rqa'].create_memory_embeddings(tmp_file.name)
        st.session_state['loaded_embeddings'] = True

    # timestamp = datetime.utcnow()

if st.session_state.loaded_embeddings and question and len(question) > 0 and st.session_state.doc_id:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message['mode'] == "LLM":
                st.markdown(message["content"])
            elif message['mode'] == "Embeddings":
                st.write(message["content"])

    text_response = None
    if mode == "Embeddings":
        text_response = st.session_state['rqa'].query_storage(question, st.session_state.doc_id,
                                                              context_size=context_size)
    elif mode == "LLM":
        _, text_response = st.session_state['rqa'].query_document(question, st.session_state.doc_id,
                                                                  context_size=context_size)

    if not text_response:
        st.error("Something went wrong. Contact Luca Foppiano (Foppiano.Luca@nims.co.jp) to report the issue.")

    with st.chat_message("user"):
        st.markdown(question)
        st.session_state.messages.append({"role": "user", "mode": mode, "content": question})

    with st.chat_message("assistant"):
        if mode == "LLM":
            st.markdown(text_response)
        else:
            st.write(text_response)
        st.session_state.messages.append({"role": "assistant", "mode": mode, "content": text_response})

elif st.session_state.loaded_embeddings and st.session_state.doc_id:
    play_old_messages()
