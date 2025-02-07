import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from PyPDF2 import PdfReader
import fitz
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import fitz
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

# File paths
file_path = os.path.dirname(os.path.abspath(__file__))

DB_dir = os.path.join(file_path, "DB")
TEST_CASE_dir = os.path.join(file_path, "TEST_CASE")


class RAGImplementation:
    _openai_embeddings = None

    def __init__(self,
                 openai_api_key: str,
                 db_file: str,
                 chunk_size: int,
                 chunk_overlap: int,
                 text_file_names: list,
                 llm) -> None:

        self.db_file = db_file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_file_names = text_file_names
        self.llm = llm

        # Initialize OpenAI embeddings only once for the class
        if RAGImplementation._openai_embeddings is None:
            RAGImplementation._openai_embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-ada-002"  # Efficient and widely used OpenAI embedding model
            )

    def load_text(self):
        all_documents = []  # List to store all documents from all files
        if not os.path.exists(TEST_CASE_dir):
            os.makedirs(TEST_CASE_dir)

        for file_name in self.text_file_names:
            text_file_path = os.path.join(TEST_CASE_dir, file_name)

            if os.path.exists(text_file_path):
                if text_file_path.endswith(".txt"):
                    # Load plain text files
                    with open(text_file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    all_documents.append(Document(page_content=text, metadata={"source": file_name}))

                elif text_file_path.endswith(".pdf"):
                    # Load PDF files
                    pdf_documents = self._load_pdf(text_file_path, file_name)
                    all_documents.extend(pdf_documents)  # Append PDFs to the documents list

                else:
                    raise ValueError(f"Unsupported file format: {file_name}")
            else:
                raise FileNotFoundError(f"File {file_name} not found in {TEST_CASE_dir}.")

        return all_documents  # Return combined list of all documents

    def _load_pdf(self, file_path, file_name):
        """Extract text from a PDF file and format it as LangChain Documents."""
        pdf_documents = []
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text("text")  # Ensuring text extraction method is correct
                # Create a Document object for each page
                pdf_documents.append(

                    Document(page_content=text, metadata={"source": file_name, "page_number": page_num + 1})
                )

        return pdf_documents

    def split_text(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        documents = self.load_text()
        pages = text_splitter.split_documents(documents=documents)
        for page in pages:
            page.metadata['source'] = page.metadata.get('source', 'Unknown')
        return pages

    def create_chroma_index(self):
        if not os.path.exists(DB_dir):
            os.makedirs(DB_dir)

        path_chroma = os.path.join(DB_dir, self.db_file)
        pages = self.split_text()

        vector_store = Chroma.from_documents(
            documents=pages,
            embedding=RAGImplementation._openai_embeddings,
            persist_directory=path_chroma
        )
        return vector_store, path_chroma

    def retrieve_with_chroma(self, query: str):
        vector_store, _ = self.create_chroma_index()
        db_retriever = vector_store.similarity_search(query=query, k=3)
        if db_retriever:

            result = db_retriever[0]
            print(f"Result Metadata: {result.metadata}")  # Debugging metadata
            print("File Source:", result.metadata.get("source", "Unknown"))
            return result.page_content
        else:
            return "No relevant results found."

    def retrieve_with_chroma(self, query: str):
        vector_store, _ = self.create_chroma_index()
        db_retriever = vector_store.similarity_search(query=query, k=3)
        if db_retriever:

            result = db_retriever[0]
            print(f"Result Metadata: {result.metadata}")  # Debugging metadata
            print("File Source:", result.metadata.get("source", "Unknown"))
            return result.page_content
        else:
            return "No relevant results found."

    def create_llm_chain(self, prompt: str):
        # Create a simple prompt template for your LLM chain
        prompt_template = """
        You are a helpful assistant. Here is some information:
        {information}
        Based on the above, answer the following query:
        {query}
        """

        # Create a prompt using the template and the input variables
        prompt = PromptTemplate(input_variables=["information", "query"], template=prompt_template)

        # Create the LLMChain object
        llm_chain = prompt | llm
        return llm_chain

    def process_with_llm_chain(self, query: str):
        # Retrieve relevant document content from Chroma
        relevant_content = self.retrieve_with_chroma(query)
        # Initialize the LLM chain with the prompt and query
        llm_chain = self.create_llm_chain(query)

        # Run the chain to generate the answer
        result = llm_chain.invoke({"information": relevant_content, "query": query})
        return result


llm = ChatOpenAI(model_name="gpt-3.5-turbo")

rag = RAGImplementation(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    db_file="chroma_index",
    chunk_size=1000,
    chunk_overlap=50,
    text_file_names=[
        file_name for file_name in os.listdir(TEST_CASE_dir)
        if file_name.endswith((".pdf", ".txt"))  # Only include .pdf and .txt files
    ],
    llm=llm
)

# Create the vector store (index the data)
vector_store, path_chroma = rag.create_chroma_index()

# Perform a search query
query = "what is machine learning"
result = rag.process_with_llm_chain(query)
print("Search Result:", result.content)
