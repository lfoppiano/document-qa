import os
import re
from hashlib import blake2b
from tempfile import NamedTemporaryFile

import dotenv
from grobid_quantities.quantities import QuantitiesAPI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from streamlit_pdf_viewer import pdf_viewer

from document_qa.ner_client_generic import NERClientGeneric

dotenv.load_dotenv(override=True)

import streamlit as st
from document_qa.document_qa_engine import DocumentQAEngine, DataStorage
from document_qa.grobid_processors import GrobidAggregationProcessor, decorate_text_with_annotations

OPENAI_MODELS = ['gpt-3.5-turbo',
                 "gpt-4",
                 "gpt-4-1106-preview"]

OPENAI_EMBEDDINGS = [
    'text-embedding-ada-002',
    'text-embedding-3-large',
    'openai-text-embedding-3-small'
]

OPEN_MODELS = {
    'mistral-7b-instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.2',
    # 'Phi-3-mini-128k-instruct': "microsoft/Phi-3-mini-128k-instruct",
    'Phi-3-mini-4k-instruct': "microsoft/Phi-3-mini-4k-instruct"
}

DEFAULT_OPEN_EMBEDDING_NAME = 'Default (all-MiniLM-L6-v2)'
OPEN_EMBEDDINGS = {
    DEFAULT_OPEN_EMBEDDING_NAME: 'all-MiniLM-L6-v2',
    'Salesforce/SFR-Embedding-Mistral': 'Salesforce/SFR-Embedding-Mistral'
}

DISABLE_MEMORY = ['zephyr-7b-beta']

if 'rqa' not in st.session_state:
    st.session_state['rqa'] = {}

if 'model' not in st.session_state:
    st.session_state['model'] = None

if 'api_keys' not in st.session_state:
    st.session_state['api_keys'] = {}

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

if 'ner_processing' not in st.session_state:
    st.session_state['ner_processing'] = False

if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = False

if 'memory' not in st.session_state:
    st.session_state['memory'] = None

if 'binary' not in st.session_state:
    st.session_state['binary'] = None

if 'annotations' not in st.session_state:
    st.session_state['annotations'] = None

if 'should_show_annotations' not in st.session_state:
    st.session_state['should_show_annotations'] = True

if 'pdf' not in st.session_state:
    st.session_state['pdf'] = None

if 'pdf_rendering' not in st.session_state:
    st.session_state['pdf_rendering'] = None

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = None

st.set_page_config(
    page_title="Scientific Document Insights Q/A",
    page_icon="📝",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/lfoppiano/document-qa',
        'Report a bug': "https://github.com/lfoppiano/document-qa/issues",
        'About': "Upload a scientific article in PDF, ask questions, get insights."
    }
)

css_modify_left_column = '''
<style>                 
    [data-testid="stHorizontalBlock"] > div:nth-child(1) {
        overflow: hidden;
        background-color: red;
        height: 70vh;
    }
</style>
'''
css_modify_right_column = '''
<style>                 
    [data-testid="stHorizontalBlock"]> div:first-child {
        background-color: red;
        position: fixed
        height: 70vh;
    }
</style>
'''
css_disable_scrolling_container = '''
<style>
    [data-testid="ScrollToBottomContainer"] {
        overflow: hidden;
    }
</style>
'''


# st.markdown(css_lock_column_fixed, unsafe_allow_html=True)
# st.markdown(css2, unsafe_allow_html=True)


def new_file():
    st.session_state['loaded_embeddings'] = None
    st.session_state['doc_id'] = None
    st.session_state['uploaded'] = True
    if st.session_state['memory']:
        st.session_state['memory'].clear()


def clear_memory():
    st.session_state['memory'].clear()


# @st.cache_resource
def init_qa(model, embeddings_name=None, api_key=None):
    ## For debug add: callbacks=[PromptLayerCallbackHandler(pl_tags=["langchain", "chatgpt", "document-qa"])])
    if model in OPENAI_MODELS:
        if embeddings_name is None:
            embeddings_name = 'text-embedding-ada-002'

        st.session_state['memory'] = ConversationBufferWindowMemory(k=4)
        if api_key:
            chat = ChatOpenAI(model_name=model,
                              temperature=0,
                              openai_api_key=api_key,
                              frequency_penalty=0.1)
            if embeddings_name not in OPENAI_EMBEDDINGS:
                st.error(f"The embeddings provided {embeddings_name} are not supported by this model {model}.")
                st.stop()
                return
            embeddings = OpenAIEmbeddings(model=embeddings_name, openai_api_key=api_key)

        else:
            chat = ChatOpenAI(model_name=model,
                              temperature=0,
                              frequency_penalty=0.1)
            embeddings = OpenAIEmbeddings(model=embeddings_name)

    elif model in OPEN_MODELS:
        if embeddings_name is None:
            embeddings_name = DEFAULT_OPEN_EMBEDDING_NAME

        chat = HuggingFaceEndpoint(
            repo_id=OPEN_MODELS[model],
            temperature=0.01,
            max_new_tokens=2048,
            model_kwargs={"max_length": 4096}
        )
        embeddings = HuggingFaceEmbeddings(
            model_name=OPEN_EMBEDDINGS[embeddings_name])
        st.session_state['memory'] = ConversationBufferWindowMemory(k=4) if model not in DISABLE_MEMORY else None
    else:
        st.error("The model was not loaded properly. Try reloading. ")
        st.stop()
        return

    storage = DataStorage(embeddings)
    return DocumentQAEngine(chat, storage, grobid_url=os.environ['GROBID_URL'], memory=st.session_state['memory'])


@st.cache_resource
def init_ner():
    quantities_client = QuantitiesAPI(os.environ['GROBID_QUANTITIES_URL'], check_server=True)

    materials_client = NERClientGeneric(ping=True)
    config_materials = {
        'grobid': {
            "server": os.environ['GROBID_MATERIALS_URL'],
            'sleep_time': 5,
            'timeout': 60,
            'url_mapping': {
                'processText_disable_linking': "/service/process/text?disableLinking=True",
                # 'processText_disable_linking': "/service/process/text"
            }
        }
    }

    materials_client.set_config(config_materials)

    gqa = GrobidAggregationProcessor(grobid_quantities_client=quantities_client,
                                     grobid_superconductors_client=materials_client)
    return gqa


gqa = init_ner()


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
                        st.markdown(message['content'], unsafe_allow_html=True)
                    else:
                        st.write(message['content'])


# is_api_key_provided = st.session_state['api_key']

with st.sidebar:
    st.session_state['model'] = model = st.selectbox(
        "Model:",
        options=OPENAI_MODELS + list(OPEN_MODELS.keys()),
        index=(OPENAI_MODELS + list(OPEN_MODELS.keys())).index(
            "mistral-7b-instruct-v0.2") if "DEFAULT_MODEL" not in os.environ or not os.environ["DEFAULT_MODEL"] else (
                OPENAI_MODELS + list(OPEN_MODELS.keys())).index(os.environ["DEFAULT_MODEL"]),
        placeholder="Select model",
        help="Select the LLM model:",
        disabled=st.session_state['doc_id'] is not None or st.session_state['uploaded']
    )
    embedding_choices = OPENAI_EMBEDDINGS if model in OPENAI_MODELS else OPEN_EMBEDDINGS

    st.session_state['embeddings'] = embedding_name = st.selectbox(
        "Embeddings:",
        options=embedding_choices,
        index=0,
        placeholder="Select embedding",
        help="Select the Embedding function:",
        disabled=st.session_state['doc_id'] is not None or st.session_state['uploaded']
    )

    if (model in OPEN_MODELS) and model not in st.session_state['api_keys']:
        if 'HUGGINGFACEHUB_API_TOKEN' not in os.environ:
            api_key = st.text_input('Huggingface API Key', type="password")

            st.markdown("Get it [here](https://huggingface.co/docs/hub/security-tokens)")
        else:
            api_key = os.environ['HUGGINGFACEHUB_API_TOKEN']

        if api_key:
            # st.session_state['api_key'] = is_api_key_provided = True
            if model not in st.session_state['rqa'] or model not in st.session_state['api_keys']:
                with st.spinner("Preparing environment"):
                    st.session_state['api_keys'][model] = api_key
                    # if 'HUGGINGFACEHUB_API_TOKEN' not in os.environ:
                    #     os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
                    st.session_state['rqa'][model] = init_qa(model, embedding_name)

    elif model in OPENAI_MODELS and model not in st.session_state['api_keys']:
        if 'OPENAI_API_KEY' not in os.environ:
            api_key = st.text_input('OpenAI API Key', type="password")
            st.markdown("Get it [here](https://platform.openai.com/account/api-keys)")
        else:
            api_key = os.environ['OPENAI_API_KEY']

        if api_key:
            if model not in st.session_state['rqa'] or model not in st.session_state['api_keys']:
                with st.spinner("Preparing environment"):
                    st.session_state['api_keys'][model] = api_key
                    if 'OPENAI_API_KEY' not in os.environ:
                        st.session_state['rqa'][model] = init_qa(model, st.session_state['embeddings'], api_key)
                    else:
                        st.session_state['rqa'][model] = init_qa(model, st.session_state['embeddings'])
    # else:
    #     is_api_key_provided = st.session_state['api_key']

    st.button(
        'Reset chat memory.',
        key="reset-memory-button",
        on_click=clear_memory,
        help="Clear the conversational memory. Currently implemented to retrain the 4 most recent messages.",
        disabled=model in st.session_state['rqa'] and st.session_state['rqa'][model].memory is None)

left_column, right_column = st.columns([1, 1])

with right_column:
    st.title("📝 Scientific Document Insights Q/A")
    st.subheader("Upload a scientific article in PDF, ask questions, get insights.")

    st.markdown(
        ":warning: [Usage disclaimer](https://github.com/lfoppiano/document-qa?tab=readme-ov-file#disclaimer-on-data-security-and-privacy-%EF%B8%8F) :warning: ")

    uploaded_file = st.file_uploader(
        "Upload an article",
        type=("pdf", "txt"),
        on_change=new_file,
        disabled=st.session_state['model'] is not None and st.session_state['model'] not in
                 st.session_state['api_keys'],
        help="The full-text is extracted using Grobid."
    )

question = st.chat_input(
    "Ask something about the article",
    # placeholder="Can you give me a short summary?",
    disabled=not uploaded_file
)

query_modes = {
    "llm": "LLM Q/A",
    "embeddings": "Embeddings",
    "question_coefficient": "Question coefficient"
}

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Query mode",
        ("llm", "embeddings", "question_coefficient"),
        disabled=not uploaded_file,
        index=0,
        horizontal=True,
        format_func=lambda x: query_modes[x],
        help="LLM will respond the question, Embedding will show the "
             "relevant paragraphs to the question in the paper. "
             "Question coefficient attempt to estimate how effective the question will be answered."
    )

    # Add a checkbox for showing annotations
    # st.session_state['show_annotations'] = st.checkbox("Show annotations", value=True)
    # st.session_state['should_show_annotations'] = st.checkbox("Show annotations", value=True)

    chunk_size = st.slider("Text chunks size", -1, 2000, value=-1,
                           help="Size of chunks in which split the document. -1: use paragraphs, > 0 paragraphs are aggregated.",
                           disabled=uploaded_file is not None)
    if chunk_size == -1:
        context_size = st.slider("Context size (paragraphs)", 3, 20, value=10,
                                 help="Number of paragraphs to consider when answering a question",
                                 disabled=not uploaded_file)
    else:
        context_size = st.slider("Context size (chunks)", 3, 10, value=4,
                                 help="Number of chunks to consider when answering a question",
                                 disabled=not uploaded_file)

    st.session_state['ner_processing'] = st.checkbox("Identify materials and properties.")
    st.markdown(
        'The LLM responses undergo post-processing to extract <span style="color:orange">physical quantities, measurements</span>, and <span style="color:green">materials</span> mentions.',
        unsafe_allow_html=True)

    st.session_state['pdf_rendering'] = st.radio(
        "PDF rendering mode",
        ("unwrap", "legacy_embed"),
        index=0,
        disabled=not uploaded_file,
        help="PDF rendering engine."
             "Note: The Legacy PDF viewer does not support annotations and might not work on Chrome.",
        format_func=lambda q: "Legacy PDF Viewer" if q == "legacy_embed" else "Streamlit PDF Viewer (Pdf.js)"
    )

    st.divider()

    st.header("Documentation")
    st.markdown("https://github.com/lfoppiano/document-qa")
    st.markdown(
        """Upload a scientific article as PDF document. Once the spinner stops, you can proceed to ask your questions.""")

    if st.session_state['git_rev'] != "unknown":
        st.markdown("**Revision number**: [" + st.session_state[
            'git_rev'] + "](https://github.com/lfoppiano/document-qa/commit/" + st.session_state['git_rev'] + ")")

    st.header("Query mode (Advanced use)")
    st.markdown(
        """By default, the mode is set to LLM (Language Model) which enables question/answering. 
        You can directly ask questions related to the document content, and the system will answer the question using content from the document.""")

    st.markdown(
        """If you switch the mode to "Embedding," the system will return specific chunks from the document 
        that are semantically related to your query. This mode helps to test why sometimes the answers are not 
        satisfying or incomplete. """)

if uploaded_file and not st.session_state.loaded_embeddings:
    if model not in st.session_state['api_keys']:
        st.error("Before uploading a document, you must enter the API key. ")
        st.stop()

    with right_column:
        with st.spinner('Reading file, calling Grobid, and creating memory embeddings...'):
            binary = uploaded_file.getvalue()
            tmp_file = NamedTemporaryFile()
            tmp_file.write(bytearray(binary))
            st.session_state['binary'] = binary

            st.session_state['doc_id'] = hash = st.session_state['rqa'][model].create_memory_embeddings(tmp_file.name,
                                                                                                        chunk_size=chunk_size,
                                                                                                        perc_overlap=0.1)
            st.session_state['loaded_embeddings'] = True
            st.session_state.messages = []

    # timestamp = datetime.utcnow()


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def generate_color_gradient(num_elements):
    # Define warm and cold colors in RGB format
    warm_color = (255, 165, 0)  # Orange
    cold_color = (0, 0, 255)  # Blue

    # Generate a linear gradient of colors
    color_gradient = [
        rgb_to_hex(tuple(int(warm * (1 - i / num_elements) + cold * (i / num_elements)) for warm, cold in
                         zip(warm_color, cold_color)))
        for i in range(num_elements)
    ]

    return color_gradient


with right_column:
    # css = '''
    #     <style>
    #         [data-testid="column"] {
    #             overflow: auto;
    #             height: 70vh;
    #         }
    #     </style>
    #     '''
    # st.markdown(css, unsafe_allow_html=True)

    # st.markdown(
    #     """
    #     <script>
    #     document.querySelectorAll('[data-testid="column"]').scrollIntoView({behavior: "smooth"});
    #     </script>
    #     """,
    #     unsafe_allow_html=True,
    # )

    if st.session_state.loaded_embeddings and question and len(question) > 0 and st.session_state.doc_id:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message['mode'] == "llm":
                    st.markdown(message["content"], unsafe_allow_html=True)
                elif message['mode'] == "embeddings":
                    st.write(message["content"])
                if message['mode'] == "question_coefficient":
                    st.markdown(message["content"], unsafe_allow_html=True)
        if model not in st.session_state['rqa']:
            st.error("The API Key for the " + model + " is  missing. Please add it before sending any query. `")
            st.stop()

        with st.chat_message("user"):
            st.markdown(question)
            st.session_state.messages.append({"role": "user", "mode": mode, "content": question})

        text_response = None
        if mode == "embeddings":
            with st.spinner("Fetching the relevant context..."):
                text_response, coordinates = st.session_state['rqa'][model].query_storage(
                    question,
                    st.session_state.doc_id,
                    context_size=context_size
                )
        elif mode == "llm":
            with st.spinner("Generating LLM response..."):
                _, text_response, coordinates = st.session_state['rqa'][model].query_document(
                    question,
                    st.session_state.doc_id,
                    context_size=context_size
                )

        elif mode == "question_coefficient":
            with st.spinner("Estimate question/context relevancy..."):
                text_response, coordinates = st.session_state['rqa'][model].analyse_query(
                    question,
                    st.session_state.doc_id,
                    context_size=context_size
                )

        annotations = [[GrobidAggregationProcessor.box_to_dict([cs for cs in c.split(",")]) for c in coord_doc]
                       for coord_doc in coordinates]
        gradients = generate_color_gradient(len(annotations))
        for i, color in enumerate(gradients):
            for annotation in annotations[i]:
                annotation['color'] = color
        st.session_state['annotations'] = [annotation for annotation_doc in annotations for annotation in
                                           annotation_doc]

        if not text_response:
            st.error("Something went wrong. Contact Luca Foppiano (Foppiano.Luca@nims.co.jp) to report the issue.")

        with st.chat_message("assistant"):
            if mode == "llm":
                if st.session_state['ner_processing']:
                    with st.spinner("Processing NER on LLM response..."):
                        entities = gqa.process_single_text(text_response)
                        decorated_text = decorate_text_with_annotations(text_response.strip(), entities)
                        decorated_text = decorated_text.replace('class="label material"', 'style="color:green"')
                        decorated_text = re.sub(r'class="label[^"]+"', 'style="color:orange"', decorated_text)
                        text_response = decorated_text
                st.markdown(text_response, unsafe_allow_html=True)
            else:
                st.write(text_response)
            st.session_state.messages.append({"role": "assistant", "mode": mode, "content": text_response})

    elif st.session_state.loaded_embeddings and st.session_state.doc_id:
        play_old_messages()

with left_column:
    if st.session_state['binary']:
        pdf_viewer(
            input=st.session_state['binary'],
            annotation_outline_size=1,
            annotations=st.session_state['annotations'],
            rendering=st.session_state['pdf_rendering'],
            render_text=True
        )
