import copy
import os
from pathlib import Path
from typing import Union, Any

from grobid_client.grobid_client import GrobidClient
from langchain.chains import create_extraction_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm

from document_qa.grobid_processors import GrobidProcessor


class DocumentQAEngine:
    llm = None
    qa_chain_type = None
    embedding_function = None
    embeddings_dict = {}
    embeddings_map_from_md5 = {}
    embeddings_map_to_md5 = {}

    def __init__(self, llm, embedding_function, qa_chain_type="stuff", embeddings_root_path=None, grobid_url=None):
        self.embedding_function = embedding_function
        self.llm = llm
        self.chain = load_qa_chain(llm, chain_type=qa_chain_type)

        if embeddings_root_path is not None:
            self.embeddings_root_path = embeddings_root_path
            if not os.path.exists(embeddings_root_path):
                os.makedirs(embeddings_root_path)
            else:
                self.load_embeddings(self.embeddings_root_path)

        if grobid_url:
            self.grobid_url = grobid_url
            grobid_client = GrobidClient(
                grobid_server=self.grobid_url,
                batch_size=1000,
                coordinates=["p"],
                sleep_time=5,
                timeout=60,
                check_server=True
            )
            self.grobid_processor = GrobidProcessor(grobid_client)

    def load_embeddings(self, embeddings_root_path: Union[str, Path]) -> None:
        """
        Load the embeddings assuming they are all persisted and stored in a single directory.
        The root path of the embeddings containing one data store for each document in each subdirectory
        """

        embeddings_directories = [f for f in os.scandir(embeddings_root_path) if f.is_dir()]

        if len(embeddings_directories) == 0:
            print("No available embeddings")
            return

        for embedding_document_dir in embeddings_directories:
            self.embeddings_dict[embedding_document_dir.name] = Chroma(persist_directory=embedding_document_dir.path,
                                                                       embedding_function=self.embedding_function)

            filename_list = list(Path(embedding_document_dir).glob('*.storage_filename'))
            if filename_list:
                filenam = filename_list[0].name.replace(".storage_filename", "")
                self.embeddings_map_from_md5[embedding_document_dir.name] = filenam
                self.embeddings_map_to_md5[filenam] = embedding_document_dir.name

        print("Embedding loaded: ", len(self.embeddings_dict.keys()))

    def get_loaded_embeddings_ids(self):
        return list(self.embeddings_dict.keys())

    def get_md5_from_filename(self, filename):
        return self.embeddings_map_to_md5[filename]

    def get_filename_from_md5(self, md5):
        return self.embeddings_map_from_md5[md5]

    def query_document(self, query: str, doc_id, output_parser=None, context_size=4, extraction_schema=None,
                       verbose=False) -> (
            Any, str):
        # self.load_embeddings(self.embeddings_root_path)

        if verbose:
            print(query)

        response = self._run_query(doc_id, query, context_size=context_size)
        response = response['output_text'] if 'output_text' in response else response

        if verbose:
            print(doc_id, "->", response)

        if output_parser:
            try:
                return self._parse_json(response, output_parser), response
            except Exception as oe:
                print("Failing to parse the response", oe)
                return None, response
        elif extraction_schema:
            try:
                chain = create_extraction_chain(extraction_schema, self.llm)
                parsed = chain.run(response)
                return parsed, response
            except Exception as oe:
                print("Failing to parse the response", oe)
                return None, response
        else:
            return None, response

    def query_storage(self, query: str, doc_id, context_size=4):
        documents = self._get_context(doc_id, query, context_size)

        context_as_text = [doc.page_content for doc in documents]
        return context_as_text

    def _parse_json(self, response, output_parser):
        system_message = "You are an useful assistant expert in materials science, physics, and chemistry " \
                         "that can process text and transform it to JSON."
        human_message = """Transform the text between three double quotes in JSON.\n\n\n\n
        {format_instructions}\n\nText: \"\"\"{text}\"\"\""""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_message)

        prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        results = self.llm(
            prompt_template.format_prompt(
                text=response,
                format_instructions=output_parser.get_format_instructions()
            ).to_messages()
        )
        parsed_output = output_parser.parse(results.content)

        return parsed_output

    def _run_query(self, doc_id, query, context_size=4):
        relevant_documents = self._get_context(doc_id, query, context_size)
        return self.chain.run(input_documents=relevant_documents, question=query)
        # return self.chain({"input_documents": relevant_documents, "question": prompt_chat_template}, return_only_outputs=True)

    def _get_context(self, doc_id, query, context_size=4):
        db = self.embeddings_dict[doc_id]
        retriever = db.as_retriever(search_kwargs={"k": context_size})
        relevant_documents = retriever.get_relevant_documents(query)
        return relevant_documents

    def get_all_context_by_document(self, doc_id):
        db = self.embeddings_dict[doc_id]
        docs = db.get()
        return docs['documents']

    def _get_context_multiquery(self, doc_id, query, context_size=4):
        db = self.embeddings_dict[doc_id].as_retriever(search_kwargs={"k": context_size})
        multi_query_retriever = MultiQueryRetriever.from_llm(retriever=db, llm=self.llm)
        relevant_documents = multi_query_retriever.get_relevant_documents(query)
        return relevant_documents

    def get_text_from_document(self, pdf_file_path, chunk_size=-1, perc_overlap=0.1, verbose=False):
        if verbose:
            print("File", pdf_file_path)
        filename = Path(pdf_file_path).stem
        structure = self.grobid_processor.process_structure(pdf_file_path)

        biblio = structure['biblio']
        biblio['filename'] = filename.replace(" ", "_")

        if verbose:
            print("Generating embeddings for:", hash, ", filename: ", filename)

        texts = []
        metadatas = []
        ids = []
        if chunk_size < 0:
            for passage in structure['passages']:
                biblio_copy = copy.copy(biblio)
                if len(str.strip(passage['text'])) > 0:
                    texts.append(passage['text'])

                    biblio_copy['type'] = passage['type']
                    biblio_copy['section'] = passage['section']
                    biblio_copy['subSection'] = passage['subSection']
                    metadatas.append(biblio_copy)

                    ids.append(passage['passage_id'])
        else:
            document_text = " ".join([passage['text'] for passage in structure['passages']])
            # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size,
                chunk_overlap=chunk_size * perc_overlap
            )
            texts = text_splitter.split_text(document_text)
            metadatas = [biblio for _ in range(len(texts))]
            ids = [id for id, t in enumerate(texts)]

        return texts, metadatas, ids

    def create_memory_embeddings(self, pdf_path, doc_id=None, chunk_size=500, perc_overlap=0.1):
        texts, metadata, ids = self.get_text_from_document(pdf_path, chunk_size=chunk_size, perc_overlap=perc_overlap)
        if doc_id:
            hash = doc_id
        else:
            hash = metadata[0]['hash']

        if hash not in self.embeddings_dict.keys():
            self.embeddings_dict[hash] = Chroma.from_texts(texts, embedding=self.embedding_function, metadatas=metadata, collection_name=hash)

        self.embeddings_root_path = None

        return hash

    def create_embeddings(self, pdfs_dir_path: Path):
        input_files = []
        for root, dirs, files in os.walk(pdfs_dir_path, followlinks=False):
            for file_ in files:
                if not (file_.lower().endswith(".pdf")):
                    continue
                input_files.append(os.path.join(root, file_))

        for input_file in tqdm(input_files, total=len(input_files), unit='document',
                               desc="Grobid + embeddings processing"):

            md5 = self.calculate_md5(input_file)
            data_path = os.path.join(self.embeddings_root_path, md5)

            if os.path.exists(data_path):
                print(data_path, "exists. Skipping it ")
                continue

            texts, metadata, ids = self.get_text_from_document(input_file, chunk_size=500, perc_overlap=0.1)
            filename = metadata[0]['filename']

            vector_db_document = Chroma.from_texts(texts,
                                                   metadatas=metadata,
                                                   embedding=self.embedding_function,
                                                   persist_directory=data_path)
            vector_db_document.persist()

            with open(os.path.join(data_path, filename + ".storage_filename"), 'w') as fo:
                fo.write("")

    @staticmethod
    def calculate_md5(input_file: Union[Path, str]):
        import hashlib
        md5_hash = hashlib.md5()
        with open(input_file, 'rb') as fi:
            md5_hash.update(fi.read())
        return md5_hash.hexdigest().upper()
