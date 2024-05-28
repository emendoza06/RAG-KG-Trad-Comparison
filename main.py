import streamlit as st
import time
from KGLLMResponder import GraphLLMResponder
from TradLLMResponder import TradLLMResponder
from openai import OpenAI
import os
import numpy as np

st.set_page_config(layout="wide")

def get_embedding(question):
    llm_embedder = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = llm_embedder.embeddings.create(
        input=question,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

def update_response():
    user_input = st.session_state.query
    top_k_trad = st.session_state.top_k_trad
    top_k_kg = st.session_state.top_k_kg
    print("\nUser entered: ")
    print(user_input)
    print("\nUser entered trad k: ")
    print(top_k_trad)
    print("\nUser entered kg k: ")
    print(top_k_kg)

    question_embedding = get_embedding(user_input)
    start_trad_process(user_input, top_k_trad, question_embedding)
    start_kg_process(user_input, top_k_kg, question_embedding)
    

def start_trad_process(user_input, top_k_trad, question_embedding):
    #Traditional response
    start_time_trad = time.time()
    trad_responder = TradLLMResponder(user_input, top_k_trad, question_embedding) #Create the trad llm responder mechanism
    trad_response = trad_responder.execute_query_and_respond() #Save the response
    st.session_state['response1'] = trad_response
    st.session_state['trad_context'] = trad_responder.context
    end_time_trad = time.time()

    elapsed_time_trad = end_time_trad - start_time_trad
    st.session_state['elapsed_time_trad'] = elapsed_time_trad
    st.session_state['input_tokens_trad'] = trad_responder.input_tokens
    st.session_state['output_tokens_trad'] = trad_responder.output_tokens
    st.session_state['total_tokens_trad'] = trad_responder.total_tokens

def start_kg_process(user_input, top_k_kg, question_embedding):
    #Knowledge graph response
    start_time_kg = time.time()
    kg_responder = GraphLLMResponder(user_input, top_k_kg, question_embedding) #Create the graph responder mechanism
    kg_response = kg_responder.execute_query_and_respond() #Save the response
    st.session_state['response2'] = kg_response #Directly update session state
    st.session_state['kg_context'] = kg_responder.context
    end_time_kg = time.time()
    elapsed_time_kg = end_time_kg - start_time_kg
    st.session_state['elapsed_time_kg'] = elapsed_time_kg
    st.session_state['input_tokens_kg'] = kg_responder.input_tokens
    st.session_state['output_tokens_kg'] = kg_responder.output_tokens
    st.session_state['total_tokens_kg'] = kg_responder.total_tokens
    st.session_state['kg_cypher'] = kg_responder.cypher

def main():
    # Custom CSS to include colors and dividing line
    st.markdown(
        """
        <style>
        .reportview-container .main .block-container {
            background: white;
        }
        .headers {
            font-size:23px !important;
        }
        .blue-box {
            background-color: #add8e6;
            color: white;
            padding: 8px;
            border-radius: 4px;
            width: 30%;
        }
        .red-box {
            background-color: #f08080;
            color: white;
            padding: 8px;
            border-radius: 4px;
            width: 25%;
        }
        .divider {
            height: 550px;
            width: 1px;
            background-color: #000000;
            display: inline-block;
        }
        div[data-testid="stForm"]{
            margin-top: 50px;
            margin-bottom: 50px;
            width: 50%;
            margin-left: 22%;
        }
        div[data-testid="stForm"] * {
            font-size: 18px; /* Apply font size to all elements within the form */
        }
        div[data-testid="stForm"] input, div[data-testid="stForm"] textarea, div[data-testid="stForm"] select, div[data-testid="stForm"] button {
            font-size: 18px; /* Ensure inputs, text areas, selects, and buttons have the specified font size */
        }
        div[data-testid="stForm"] label {
            font-size: 18px; /* Specific font size for labels if needed */
        }
        div.stButton > button:first-child {
            display: block;
            margin: 0 auto;
        }
        div[data-testid="stTextAreast.response_text_area2"] > label {
            display: none;
        }
        textarea{
            font-size: 20px !important
        }
        .cypher{
            font-size: 18px;
        }
        .metrics{
            margin-top: 30px;
        }
        div[data-testid="stExpander"] .stText {
            font-size: 20px; /* Increase font size */
        }
        div[data-testid="stExpander"] > div:first-child {
            font-size: 24px !important; /* Adjust the font size of the expander title */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Function to create expandable sections
    def create_section_expandable_context(title, content, key):
        with st.expander(title):
            st.session_state[key] = st.session_state[key]
            if isinstance(content, list):
                #Define a list of colors to loop through so each paragraph used can be a new color
                if key == "trad_context":
                    colors = ['#004AAD', '#49DCE1', '#004AAD', '#49DCE1', '#004AAD', '#49DCE1']
                else:
                    colors = ['#8B0F0F', '#FF3030', '#8B0F0F', '#FF3030', '#8B0F0F', '#FF3030']
                #If a list then separate each element with a newline
                #formatted_content = "<br>".join(map(str, content))

                #Each item gets its own css color
                #Knowledge graph gives context in dictionary form. Trad gives in list form. If item contains {'p.text': ...}
                #then we know it's in dictionary format. Check if 'item' is a dictionary and contains the key 'p.text'. If both conditions are true
                #extract item['p.text'] to just get the value
                formatted_content = ''.join(f"<div style='margin-bottom: 50px; color: {colors[i % len(colors)]};'>"
                                            f"{item['p.text'] if isinstance(item, dict) and 'p.text' in item else str(item)}"
                                            for i, item in enumerate(content))
                
                st.markdown(f"<div style='font-size: 20px;'>{formatted_content}</div>", unsafe_allow_html=True)
            else:
                #If not a list then print whole string
                st.markdown(f"<div style='font-size: 20px;'>{content}</div>", unsafe_allow_html=True)
        
    def create_time_section_1(content):
        # st.text(content)
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: blue'><strong>Time: </strong> {content}</div>", unsafe_allow_html=True)

    def create_time_section_2(content):
        # st.text(content)
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: red'><strong>Time: </strong> {content}</div>", unsafe_allow_html=True)
    

    # Page layout
    st.title('LLM Comparison Dashboard')

        #Use a form to take input and button interactions
    with st.form("query_form"):
        #Input for new queries
        user_input = st.text_input("Ask a question", key="query", autocomplete="off")
        #Top k columns
        col3, col4 = st.columns(2)
        with col3:
            #Input for top_k_trad
            top_k_trad = st.text_input("Top K for traditional", key="top_k_trad", autocomplete="off")
        with col4:
            #Input for top_k_kg
            top_k_kg = st.text_input("Top K for KG", key="top_k_kg", autocomplete="off")
        #Submit button for the form
        submit_button = st.form_submit_button("Run", on_click=update_response) #On click, call function to perform kg and trad search + response

    col1, divider, col2 = st.columns([4.5, 0.1, 5])



    # TRADITIONAL column
    with col1:

        #--------RESPONSE AREA------------
        st.markdown('<div class="blue-box"><h1 class="headers">LLM with RAG</h1></div>', unsafe_allow_html=True)
        #Use session state to hold the response
        if 'response1' not in st.session_state:
            st.session_state['response1'] = "Awaiting response..."
        #display response in text area
        st.text_area("Response-trad", st.session_state['response1'], height=200, key='response_text_area', label_visibility="hidden")


        #-------METRICS AREA---------
        st.markdown('<h4 class="metrics" style="margin-top: 80px">Metrics</h4>', unsafe_allow_html=True)
        #Use session state to hold time response
        if 'elapsed_time_trad' not in st.session_state:
            st.session_state['elapsed_time_trad'] = ""
        #Time
        create_time_section_1(st.session_state['elapsed_time_trad'])
        #Cost
        if 'input_tokens_trad' not in st.session_state:
            st.session_state['input_tokens_trad'] = ""
        if 'output_tokens_trad' not in st.session_state:
            st.session_state['output_tokens_trad'] = ""
        if 'total_tokens_trad' not in st.session_state:
            st.session_state['total_tokens_trad'] = ""
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: blue'><strong>Input tokens: </strong>{st.session_state['input_tokens_trad']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: blue'><strong>Output tokens: </strong>{st.session_state['output_tokens_trad']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: blue'><strong>Total tokens: </strong>{st.session_state['total_tokens_trad']}</div>", unsafe_allow_html=True)
        #Context
        if 'trad_context' not in st.session_state:
            st.session_state['trad_context'] = "Awaiting response..."
        create_section_expandable_context("Top K (Context used)", st.session_state['trad_context'], 'trad_context') #key at the end in order to distinguish between expandables
        #create_section("Cost:", "Details about cost efficiency...", key="2")
        

    # Divider
    with divider:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)



    # KNOWLEDGE GRAPH column
    with col2:
        st.markdown('<div class="red-box"><h1 class="headers">LLM with RAG using KG</h1></div>', unsafe_allow_html=True)
        #Use session state to hold the response
        if 'response2' not in st.session_state:
            st.session_state['response2'] = "Awaiting response..."
        #display response in text area
        st.text_area("Response-kg", st.session_state['response2'], height=200, key='response_text_area2', label_visibility="hidden")

        #Cypher
        if 'kg_cypher' not in st.session_state:
            st.markdown('<p class="cypher">Cypher: None set.</p>', unsafe_allow_html=True)
        else:
            cypher_query = st.session_state['kg_cypher']
            st.markdown(f'<p class="cypher">Cypher: {cypher_query} </p>', unsafe_allow_html=True)

        st.markdown('<h4 class="metrics">Metrics</h4>', unsafe_allow_html=True)
        #Use session state to hold time response
        if 'elapsed_time_kg' not in st.session_state:
            st.session_state['elapsed_time_kg'] = ""
        #display time
        create_time_section_2(st.session_state['elapsed_time_kg'])
        #Cost
        if 'input_tokens_kg' not in st.session_state:
            st.session_state['input_tokens_kg'] = ""
        if 'output_tokens_kg' not in st.session_state:
            st.session_state['output_tokens_kg'] = ""
        if 'total_tokens_kg' not in st.session_state:
            st.session_state['total_tokens_kg'] = ""
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: red'><strong>Input tokens: </strong>{st.session_state['input_tokens_kg']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: red'><strong>Output tokens: </strong>{st.session_state['output_tokens_kg']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: red'><strong>Total tokens: </strong>{st.session_state['total_tokens_kg']}</div>", unsafe_allow_html=True)
        #Display context used
        if 'kg_context' not in st.session_state:
            st.session_state['kg_context'] = "Awaiting response..."
        create_section_expandable_context("Top K (Context used)", st.session_state['kg_context'], 'kg_context')
        #create_section("Cost:", "Details about cost efficiency...", key="5")
        

if __name__== "__main__":
    main()