from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from typing import Optional, List
from app.services.logger import setup_logger
from pydantic import BaseModel


logger = setup_logger(__name__)
class GenerateNotesOutput(BaseModel):
    title: str
    notes: str

class BulletPoints(BaseModel):
    title: str
    points: List[str]

class Paragraph(BaseModel):
    title: str
    content: str

class Table(BaseModel):
    title: str
    rows: List[List[str]]

class NoteGeneratorPipeline:
    def __init__(self, args=None , verbose=False):       
        self.args = args
        self.verbose = verbose
        parsers = {
            "bullet points": JsonOutputParser(pydantic_object=BulletPoints),
            "paragraph": JsonOutputParser(pydantic_object=Paragraph),
            "table": JsonOutputParser(pydantic_object=Table),
        }
        self.parsers = parsers[args.page_layout]  if self.args!= None else None
        self.model = GoogleGenerativeAI(model="gemini-1.5-pro")
        self.vectorstore_class = Chroma
        self.vectorstore = None
        self.retriever = None
          
    def compile_vectorstore(self, documents: List[str]):
        """Creates a vector store for document retrieval."""
        if self.verbose:
            logger.info("Creating vectorstore from documents...")
        self.vectorstore = self.vectorstore_class.from_documents(
            documents, 
            GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        )
        print("vector store",self.vectorstore)
        self.retriever = self.vectorstore.as_retriever()
        print("retriever",self.retriever)
        
        if self.verbose:
            logger.info("Vectorstore and retriever created successfully.")

    def generate_context(self, query):
        """Retrieves relevant context from the vector database."""
        return self.retriever.invoke(query)

    def compile_pipeline(self):
        """Creates prompt templates for different layouts."""
        layout_templates = {
            "bullet points": PromptTemplate(
                template=(
                    "Generate structured notes as bullet points focusing on: {focus}. "
                    "Use the following text: {context}. "
                    "Ensure the output is concise and well-formatted. "
                    "Respond in the {lang} language."
                ),
                input_variables=["focus", "context", "lang"],
                partial_variables={"format_instructions": self.parsers.get_format_instructions()},
            ),
            "paragraph": PromptTemplate(
                template=(
                    "Summarize the key points in a well-structured paragraph focusing on: {focus}. "
                    "Use the following text: {context}. "
                    "Ensure clarity, coherence, and completeness. "
                    "Respond in the {lang} language."
                ),
                input_variables=["focus", "context", "lang"],
                partial_variables={"format_instructions": self.parsers.get_format_instructions()},
            ),
            "table": PromptTemplate(
                template=(
                    "Generate a structured table summarizing the key information focusing on: {focus}. "
                    "Use the following text: {context}. "
                    "Ensure the table is well-organized, with clear headers and concise content. "
                    "Respond in the {lang} language."
                ),
                input_variables=["focus", "context", "lang"],
                partial_variables={"format_instructions": self.parsers.get_format_instructions()},
            ),
        }

        # Select the prompt template based on page_layout
        if self.args.page_layout not in layout_templates:
            raise ValueError("Invalid page_layout. Choose 'bullet points', 'paragraph', or 'table'.")

        return layout_templates[self.args.page_layout] | self.model

    def generate_notes(self,documents: Optional[List[Document]]):
        """Generates notes based on the selected layout."""
        # If a file is uploaded, process it   
        
        # If documents are available, store them and retrieve relevant context
        if documents:
            self.compile_vectorstore(documents)
            query = "Provide general context for the topic to create notes."
            context = self.generate_context(query)
            
        else:
            context = ""  # Use manually provided text if no file is uploaded

        # Compile the processing pipeline
        pipeline = self.compile_pipeline()

        # Prepare inputs for the AI model
        inputs = {
            "focus": self.args.focus,
            "context": context,
            "lang": self.args.lang
        }

        try:
            result = pipeline.invoke(inputs)
            feedback = GenerateNotesOutput(
                title=f'Generated Notes in {self.args.page_layout} format',
                notes=result
            )

            if self.verbose:
                logger.info("Notes successfully generated.")
            return feedback
        except Exception as e:
            logger.error(f"Error generating notes: {e}")
            raise ValueError("Failed to generate notes.")
