import faiss
import numpy as np
import os
import json
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import tiktoken

def load_texts(path):
        with open(path, 'r') as file:
            return json.load(file)

class TradLLMResponder:
    def __init__(self,question,top_k, question_embedding):
        self.llm = ChatOpenAI(
            openai_api_key = os.getenv('OPENAI_API_KEY'),
            temperature= 0
        )
        self.index_path = "./processed-data/paragraph_embeddings.faiss"
        self.paragraph_index = faiss.read_index(self.index_path)

        self.plain_text_path = "./processed-data/paragraph_texts.json"
        self.plain_text = load_texts(self.plain_text_path)

        self.llm_embedder = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.question = question
        self.question_embedding = question_embedding
        self.top_k = top_k
        self.context = ""
        #Number of input tokens
        self.input_tokens = 0
        #Number of output tokens
        self.output_tokens = 0
        #Input + output tokens
        self.total_tokens = 0

        print()
        print("***************************************************************")
        print("Traditional lLm")

        print("questions: ")
        print(self.question)
        print("K is : ")
        print(self.top_k)
    
        
    def get_embedding(self):
        response = self.llm_embedder.embeddings.create(
            input=self.question,
            model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding)
        
    def search_faiss_index(self):
        #Query FAISS index
        D, I = self.paragraph_index.search(np.array([self.question_embedding]), int(self.top_k)) #Search for k nearest neighbors
        return I[0], D[0] #Indices and distances of the results

    def get_response_from_GPT(self, prompt):
        messages = [
            ("system", "You are a helpful assistant."),
            ("human", prompt),
        ]

        response = self.llm.invoke(messages).content
        print("Response:")
        print(response)
        return response
    
    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def execute_query_and_respond(self):
        indices, distances = self.search_faiss_index()
        result_texts = [self.plain_text[idx] for idx in indices]

        self.context = result_texts
        
        print("Paragraphs used: ")
        print(str(self.context))
        
        prompt_query = """
        Context:
        {Chunk_Text}
        Given the above context information, answer the question below. If there is no context information above, then answer with "I
        am unable to assist you.". Stick to facts. Response should be generated only from the given context. If the question is 
        absolutely not relevent to the context then respond as "I am unable to assist you". 

        Respond only with the answer.

        Question: 
        {Question}
        """

        #Fill context with nodes that match query
        prompt_query = prompt_query.replace("{Chunk_Text}",str(result_texts))
        #Fill question with user question
        prompt_query = prompt_query.replace("{Question}", self.question)

        print("Trad prompt query is: ")
        print(prompt_query)

        #Get input tokens
        self.input_tokens = self.num_tokens_from_string(prompt_query, "cl100k_base")

        final_output = self.get_response_from_GPT(prompt_query)
        
        #Get output tokens
        self.output_tokens = self.num_tokens_from_string(final_output, "cl100k_base")

        #Total input + output tokens
        self.total_tokens = self.input_tokens + self.output_tokens

        return final_output

        