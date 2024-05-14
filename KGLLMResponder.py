import os
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
load_dotenv()

class GraphLLMResponder:
    def __init__(self, question, top_k):
        self.llm = ChatOpenAI(
            openai_api_key = os.getenv('OPENAI_API_KEY'),
            temperature= 0
        )

        self.graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI_AURA'),
            #username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD_AURA')
        )
        self.question = question
        self.top_k = top_k

        print()
        print("***************************************************************")
        print("KG lLm")



    def get_cypher_query(self):
        #Prompt to get cypher statement from the question
        prompt_template = """ You are an expert Neo4j Developer tasked with translating user questions into 
        generalized Cypher queries to answer questions about Neo4j Graph Academy lessons. Identify key concepts 
        or terms from the question to form search conditions that searches these terms across different node levels 
        (Courses, Modules, Lessons, Paragraphs) and integrate these conditions into a single Cypher query 
        using appropriate logical operators within a WHERE and OR clause. Avoid syntax errors by properly structuring the query 
        to handle multiple search conditions seamlessly. Specify in the query output to return only the text attribute and return limit is {top_k}. Only 
        respond with the Cypher code starting exactly with the query syntax. The response should start directly with the 'MATCH' 
        keyword and should not include any labels, comments, introductory words, or newline. Do not respond in any other way.


            Schema: {schema}
            Question: {question}
            Cypher Code:

        """
        #Update the prompt template schema with the graph schema
        prompt_template =  prompt_template.replace("{schema}", str(self.graph.get_schema))
        #Update the prompt template question with the user's question
        prompt_template = prompt_template.replace("{question}", self.question)
        #Update the prompt template top_k with desired amount
        prompt_template = prompt_template.replace("{top_k}", self.top_k)

        return self.invoke_llm(prompt_template)
    
    def invoke_llm(self, prompt):
        messages = [
            ("system", "You are a helpful assistant."),
            ("human", prompt),
        ]
        return self.llm.invoke(messages).content
    
    def execute_query_and_respond(self):
        cypher_query = self.get_cypher_query()
        print("Cypher query " + cypher_query)
        neo_output = self.graph.query(cypher_query)
        print("Neo output " + str(neo_output))


        prompt_query = """
        Context:
        {Neo4J_Text}
        Given the above context information, answer the question below. Stick to facts. Response should be generated only from the given context.
        If the question is not relevant to the context then respond as "I am unable to assist you". Do not respond in any other way.

        Question: 
        {Question}
        """

        prompt_query = prompt_query.replace("{Neo4J_Text}", str(neo_output))
        prompt_query = prompt_query.replace("{Question}", self.question)
        
        response = self.invoke_llm(prompt_query)
        print("response is: ")
        print(response)
        return response
                
                
#kg_responder = GraphLLMResponder(user_input)