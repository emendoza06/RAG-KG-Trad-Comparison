import faiss
import numpy as np
import os
import json
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def load_texts(path):
        with open(path, 'r') as file:
            return json.load(file)

class TradLLMResponder:
    def __init__(self,question,top_k):
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
        self.top_k = top_k

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
        
    def search_faiss_index(self, query_embedding):
        #Query FAISS index
        D, I = self.paragraph_index.search(np.array([query_embedding]), int(self.top_k)) #Search for k nearest neighbors
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
    
    def execute_query_and_respond(self):
        query_embedding = self.get_embedding()
        indices, distances = self.search_faiss_index(query_embedding)
        result_texts = [self.plain_text[idx] for idx in indices]

        print("Paragraphs used: ")
        print(str(result_texts))
        
        prompt_query = """
        Context:
        {Chunk_Text}
        Given the above context information, answer the question below. Stick to facts. Response should be generated only from the given context.
        If the question is not relavant to the context then repond as "I am unable to assist you". Do not respond in any other way. 

        Question: 
        {Question}
        """

        #Fill context with nodes that match query
        prompt_query = prompt_query.replace("{Chunk_Text}",str(result_texts))
        #Fill question with user question
        prompt_query = prompt_query.replace("{Question}", self.question)

        print("\n")
        print("Prompt query sent: ")
        print(prompt_query)

        return self.get_response_from_GPT(prompt_query)

        