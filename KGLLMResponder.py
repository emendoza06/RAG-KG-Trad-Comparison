import os
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()
import tiktoken

class GraphLLMResponder:
    def __init__(self, question, top_k, question_embedding):
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
        self.question_embedding = question_embedding
        self.context = ""
        self.cypher = ""
        #Number of input tokens
        self.input_tokens = 0
        #Number of output tokens
        self.output_tokens = 0
        #Input + output tokens
        self.total_tokens = 0

        print()
        print("***************************************************************")
        print("KG lLm")

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def get_cypher_query(self):
        #Prompt to get cypher statement from the question
        prompt_template = """You are a sophisticated query generation system designed specifically to interpret user questions and strategically 
        formulate the most effective search queries to retrieve answers from the Neo4j Graph Academy lessons. Identify broader themes or topics 
        from the question and consider synonyms, variations, and related terms to form search conditions.
        For example, for 'Neo4j', include 'Neo4J', 'Neo4j database', 'Neo4j Aura'; for 'LLMs', include 'LLMs', 'Large Language Models';
        and for 'GenAI', include 'Generative AI', 'AI generation'. Use these variations to explore these themes across different node levels
        (Courses, Modules, Lessons, Paragraphs).
        Use the CONTAINS keyword, rather than regex, to match against each term. Terms searched for should be written in lowercase, however you should
        never use the LOWER() or toLower() function. Construct the query to allow keywords to appear in any order by listing them in separate CONTAINS 
        conditions combined with OR clauses. Ensure not all specified terms need to be present in the text, thus widening the search scope. 
        Integrate these conditions into a single Cypher query using appropriate logical operators within a WHERE and OR clause. Avoid syntax 
        errors by properly structuring the query to handle multiple search conditions seamlessly. Specify in the query output to return only 
        the p.text and p.embedding attributes. Only respond with the Cypher code starting exactly with the query syntax. The response should 
        start directly with the 'MATCH' keyword and should not include any labels, comments, introductory words, or newline. Do not respond in any other way.

            Schema: {schema}
            Question: {question}
        """


        #Update the prompt template schema with the graph schema
        prompt_template =  prompt_template.replace("{schema}", str(self.graph.get_schema))
        #Update the prompt template question with the user's question
        prompt_template = prompt_template.replace("{question}", self.question)
        #Update the prompt template top_k with desired amount
        prompt_template = prompt_template.replace("{top_k}", self.top_k)

        print("KG LLM Prompt template is: ")
        print(prompt_template)

        #Get input tokens
        self.input_tokens = self.num_tokens_from_string(prompt_template, "cl100k_base")
        print("\nInput tokens from cypher prompt")
        print(self.input_tokens)


        return self.invoke_llm(prompt_template)
    
    def invoke_llm(self, prompt):
        messages = [
            ("system", "You are a helpful assistant."),
            ("human", prompt),
        ]
        return self.llm.invoke(messages).content
    
    def execute_query_and_respond(self):
        #---------Use LLM to get cypher query----------
        cypher_query = self.get_cypher_query()
        self.cypher = cypher_query
        print("Cypher query " + cypher_query)
        #Get output tokens
        self.output_tokens = self.num_tokens_from_string(cypher_query, "cl100k_base")
        
        #---------Use LLM to get answer from graph----------
        try:
            #Use cypher to get relevant nodes
            full_neo_output = self.graph.query(cypher_query)
            #If I get a response from the query, then calculate most similar nodes based on node embeddings
            if full_neo_output:
                texts = []
                embeddings = []
                for record in full_neo_output:
                    texts.append(record['p.text'])
                    embeddings.append(record['p.embedding'])

                #Calculate cosine similarity
                embeddings = np.array(embeddings)
                similarities = cosine_similarity([self.question_embedding], embeddings)[0] #List of similarity scores against query and list of embeddings

                #Get top k results
                top_k_int = int(self.top_k)
                top_indices = np.argsort(similarities)[-top_k_int:][::-1]
                top_k_neo_output = [(texts[i], similarities[i]) for i in top_indices]

                self.context = top_k_neo_output
                
                print("Paragraphs used: ")
                print(str(self.context))
        except Exception as e:
            print(f"An error occured: {e}")
        finally:
            prompt_query = """
                Context:
                {Neo4J_Text}
                Given the above context information, answer the question below. If there is no context information above, then answer with "I
                am unable to assist you.". Stick to facts. Response should be generated only from the given context. If the question is 
                absolutely not relevent to the context then respond as "I am unable to assist you". 

                Respond only with the answer.

                Question: 
                {Question}
                """

            prompt_query = prompt_query.replace("{Neo4J_Text}", str(self.context))
            prompt_query = prompt_query.replace("{Question}", self.question)
            
            print("KG prompt template second pass is: ")
            print(prompt_query)

            #Additional input tokens from second call to llm
            #Get input tokens
            self.input_tokens += self.num_tokens_from_string(prompt_query, "cl100k_base")
            print("\nAdding input tokens from second pass")
            print(self.input_tokens)

            final_output = self.invoke_llm(prompt_query)
            print("response is: ")
            print(final_output)
            #Additional output tokens from second call to llm
            #Get output tokens
            self.output_tokens += self.num_tokens_from_string(final_output, "cl100k_base")
            print("\nAdding output tokens from second pass")
            print(self.output_tokens)

            #Total input + output tokens
            self.total_tokens = self.input_tokens + self.output_tokens
            
            return final_output
                
                
#kg_responder = GraphLLMResponder(user_input)