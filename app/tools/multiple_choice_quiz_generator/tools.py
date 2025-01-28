from typing import List, Dict
import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langsmith import traceable

from app.services.logger import setup_logger

relative_path = "tools/multiple_choice_quiz_generator"

logger = setup_logger(__name__)

def transform_json_dict(input_data: dict) -> dict:
    generated_questions = []

    # Validate and parse the input data to ensure it matches the QuizQuestion schema
    quiz_questions_list = QuizQuestionsList(**input_data)
    
    for question in quiz_questions_list.questions_list:
        transformed_question = {
                "question": question.question,
                "choices": {choice.key: choice.value for choice in question.choices},
                "answer": question.answer,
                "explanation": question.explanation
            }
        generated_questions.append(transformed_question)

    return generated_questions

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()

class QuizBuilderConfig:
    
    def __init__(
        self,
        model = None,
        embedding_model = None,
        vectorstore_class = None,
        max_questions: int = 10,
        min_questions: int = 1,
        max_attempts: int = 2,
        prompt_template_path: str = "prompt/multiple_choice_quiz_generator_prompt.txt",
        multi_query_prompt_path: str = "prompt/multi_query_prompt.txt",
        parser: JsonOutputParser = None,
        verbose: bool = False
    ):
        self.model = model or GoogleGenerativeAI(model="gemini-1.5-pro", max_output_tokens=None ) 
        self.embedding_model = embedding_model or GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        self.vectorstore_class = vectorstore_class or Chroma
        self.max_questions = max_questions
        self.min_questions = min_questions
        self.max_attempts = max_attempts
        self.prompt_template_path = prompt_template_path
        self.multi_query_prompt_path = multi_query_prompt_path
        self.verbose = verbose

        # load the prompt template
        self.prompt_template = read_text_file(self.prompt_template_path)
        self.multi_query_prompt_template = read_text_file(self.multi_query_prompt_path)

        self.parser = parser or JsonOutputParser(pydantic_object=QuizQuestionsList)

class QuizBuilder:
    def __init__(self, topic: str, lang: str ='en', config: QuizBuilderConfig = None, verbose: bool = False):
        self.topic = topic
        self.lang = lang
        self.config = config or QuizBuilderConfig(verbose=verbose)

        self.verbose = verbose
        self.runner = None
        self._model = self.config.model
        self._prompt_template = self.config.prompt_template
        self._parser = self.config.parser


        # Initialize components
        self.vectorstore_manager = VectorStoreManager(self.config)
        self.retriever_factory = RetrieverFactory(self.config)

        if topic is None: raise ValueError("Topic must be provided")
    
    def compile(self, documents: List[Document], num_questions: int):
        # Return the chain
        if self.verbose: logger.info(f"Compiling chain")

        number_documents = len(documents)

        if self.runner is None:
            vectorstore = self.vectorstore_manager.create_vectorstore(documents)
            retriever_k = number_documents
            retriever = self.retriever_factory.create_multiquery_retriever(
                    vectorstore, num_questions, retriever_k)

            self.runner = RunnableParallel(
                {
                "context": retriever, 
                "attribute_collection": RunnablePassthrough()
                }
            )

        prompt = PromptTemplate(
            template=self._prompt_template,
            input_variables=[{"attribute_collection"}],
            partial_variables={"format_instructions": self._parser.get_format_instructions(), 
                               "num_questions": num_questions
                               }
        )
        trace_metadata = {
            "number_documents": number_documents,
            "n_questions": num_questions,
            "topic": self.topic,
            "lang": self.lang
        }
        chain = (self.runner | prompt | self._model | self._parser).with_config(trace_metadata)
        
        if self.verbose: logger.info(f"Chain compilation complete")
        
        return chain

    def validate_response(self, response: Dict) -> bool:
        try:
            # Assuming the response is already a dictionary
            if isinstance(response, dict):
                if 'question' in response and 'choices' in response and 'answer' in response and 'explanation' in response:
                    choices = response['choices']
                    if isinstance(choices, dict):
                        for key, value in choices.items():
                            if not isinstance(key, str) or not isinstance(value, str):
                                return False
                        return True
            return False
        except TypeError as e:
            if self.verbose:
                logger.error(f"TypeError during response validation: {e}")
            return False

    def format_choices(self, choices: Dict[str, str]) -> List[Dict[str, str]]:
        return [{"key": k, "value": v} for k, v in choices.items()]
    
    def create_questions(self, documents: List[Document], num_questions: int = 5) -> List[Dict]:
        if self.verbose: logger.info(f"Creating {num_questions} questions")
     
        if num_questions > self.config.max_questions:
            return {"message": "error", "data": "Number of questions cannot exceed 10"}
        
        chain = self.compile(documents, num_questions)
        
        generated_questions = []
        attempts = 0
        max_attempts = self.config.max_attempts  # Allow for more attempts to generate questions

        while len(generated_questions) == 0 and attempts < max_attempts:
            if self.verbose:
                logger.info(f"Running pipeline. Attempt {attempts + 1} of {max_attempts}")

            try:
                # Run the pipeline with the provided input data
                response = chain.invoke(f"Topic: {self.topic}, Lang: {self.lang}")
        
                logger.info(f"Generated response: {response}")
                if response is None: next

                questions_list = transform_json_dict(response)
                for question in questions_list:
                    # Directly check if the response format is valid
                    if self.validate_response(question):
                        question["choices"] = self.format_choices(question["choices"])
                        generated_questions.append(question)
                        if self.verbose:
                            logger.info(f"Valid question added: {question}")
                    else:
                        if self.verbose:
                            logger.warning(f"Invalid response format. Attempt {attempts + 1} of {max_attempts}")
            except TypeError as e:
                if self.verbose:
                    logger.error(f"TypeError generating questions: {e}")
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error generating questions: {e}")
            attempts += 1

        number_generated_questions = len(generated_questions)
        logger.info(f"Total generated questions: {number_generated_questions}") if self.verbose else None
        
        # Log if fewer questions are generated
        if number_generated_questions < num_questions:
            if self.verbose: logger.warning(f"Only generated {number_generated_questions} out of {num_questions} requested questions")
        

        self.vectorstore_manager.cleanup()
        
        # Return the list of questions
        return generated_questions[:num_questions]

class RetrieverFactory:
    
    def __init__(self, config: QuizBuilderConfig):
        self.verbose = config.verbose

        self._model = config.model
        self._prompt_template = config.multi_query_prompt_template
    
    def create_multiquery_prompt(self, num_questions: int) -> PromptTemplate:
        try:
            return PromptTemplate(
                input_variables=["question"],
                partial_variables={
                    "num_questions": num_questions,
                },
                template=self._prompt_template
            )
        except Exception as e:
            logger.error(f"Failed to create multiquery prompt: {e}") if self.verbose else None
            raise Exception(f"Prompt creation failed: {str(e)}")

    def create_base_retriever(self, vectorstore, retriever_k: int):
        try:
            return vectorstore.as_retriever(
                search_kwargs={
                    "k": retriever_k,
                }
            )
        except Exception as e:
            logger.error(f"Failed to create base retriever: {e}")
            raise Exception(f"Base retriever creation failed: {str(e)}")

    def create_multiquery_chain(self, prompt: PromptTemplate):
        # Create the multiquery chain
        try:
            output_parser = QueryListOutputParser()
            return prompt | self._model | output_parser
        except Exception as e:
            logger.error(f"Failed to create multiquery chain: {e}")
            raise Exception(f"Chain creation failed: {str(e)}")

    def create_multiquery_retriever(
        self, 
        vectorstore,
        num_questions: int,
        retriever_k: int
    ) -> MultiQueryRetriever:
        if self.verbose:
            logger.info("Setting up MultiQueryRetriever")
            
        try:
            base_retriever = self.create_base_retriever(vectorstore, retriever_k)
            prompt = self.create_multiquery_prompt(num_questions)
            chain = self.create_multiquery_chain(prompt)
            
            retriever = MultiQueryRetriever(
                retriever=base_retriever,
                llm_chain=chain,
                parser_key="lines",
                verbose=self.verbose
            )
            
            if self.verbose:
                logger.info("MultiQueryRetriever created successfully")
                
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to create enhanced retriever: {e}")
            raise Exception(f"Multiquery retriever creation failed: {str(e)}")
class VectorStoreManager:
    
    def __init__(self, config: 'QuizBuilderConfig'):
        self.config = config
        self.verbose = config.verbose
        self._vectorstore_class = self.config.vectorstore_class
        self._embedding_model = self.config.embedding_model 
        
        self._vectorstore = None
    
    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = self.create_vectorstore()
        return self._vectorstore
    
    @traceable(run_type="embedding")
    def create_vectorstore(self, documents: List[Document]):

        logger.info(f"Creating vectorstore from {len(documents)} documents") if self.verbose else None
        self._vectorstore = self._vectorstore_class.from_documents(
            documents, 
            self._embedding_model
        )
        logger.info(f"Vectorstore created") if self.verbose else None
        return self._vectorstore
    
    def cleanup(self):
        if self.verbose: logger.info(f"Deleting vectorstore")
        if self._vectorstore:
            self._vectorstore.delete_collection()
            self._vectorstore = None
            self._vectorstore_class = None

class QueryListOutputParser(BaseOutputParser[List[str]]):
    # Output parser for a list of lines.
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines)) 
class QuestionChoice(BaseModel):
    key: str = Field(description="A unique identifier for the choice using letters A, B, C, or D.")
    value: str = Field(description="The text content of the choice")

class QuizQuestion(BaseModel):
    question: str = Field(description="The question text")
    choices: List[QuestionChoice] = Field(description="A list of choices for the question, each with a key and a value")
    answer: str = Field(description="The key of the correct answer from the choices list")
    explanation: str = Field(description="An explanation of why the answer is correct")

class QuizQuestionsList(BaseModel):
    questions_list: List[QuizQuestion] = Field(description="A list of questions for the quiz")

    model_config = {
        "json_schema_extra": {
            "examples": """ 
                "questions_list": [
                    {
                        "question": "What is the capital of France?",
                        "choices": [
                            {"key": "A", "value": "Berlin"},
                            {"key": "B", "value": "Madrid"},
                            {"key": "C", "value": "Paris"},
                            {"key": "D", "value": "Rome"},
                        ],
                        "answer": "C",
                        "explanation": "Paris is the capital of France."
                    },
                    {
                        "question": "What is the official language of France?",
                        "choices": [
                            {"key": "A", "value": "French"},
                            {"key": "B", "value": "English"},
                            {"key": "C", "value": "German"},
                            {"key": "D", "value": "Spanish"}
                        ],
                        "answer": "A",
                        "explanation": "The official language of France is French."
                    },
                ]§

          """
        }

      }
    

