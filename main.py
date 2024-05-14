import streamlit as st
import time
from KGLLMResponder import GraphLLMResponder
from TradLLMResponder import TradLLMResponder

st.set_page_config(layout="wide")

def update_response():
    user_input = st.session_state.query
    top_k = st.session_state.top_k
    print("User entered: ")
    print(user_input)
    print("User entered k: ")
    print(top_k)

    #Traditional response
    start_time_trad = time.time()
    trad_responder = TradLLMResponder(user_input, top_k) #Create the trad llm responder mechanism
    trad_response = trad_responder.execute_query_and_respond() #Save the response
    st.session_state['response1'] = trad_response
    end_time_trad = time.time()
    elapsed_time_trad = end_time_trad - start_time_trad
    st.session_state['elapsed_time_trad'] = elapsed_time_trad

    #Knowledge graph response
    start_time_kg = time.time()
    kg_responder = GraphLLMResponder(user_input, top_k) #Create the graph responder mechanism
    kg_response = kg_responder.execute_query_and_respond() #Save the response
    st.session_state['response2'] = kg_response #Directly update session state
    end_time_kg = time.time()
    elapsed_time_kg = end_time_kg - start_time_kg
    st.session_state['elapsed_time_kg'] = elapsed_time_kg

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
            height: 500px;
            width: 1px;
            background-color: #000000;
            display: inline-block;
        }
        div[data-testid="stForm"]{
            margin-top: 100px;
            width: 50%;
            margin-left: auto;
            margin-right: auto;
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
        div[data-testid="stTextAreast.response_text_area2"] > label {
            display: none;
        }
        textarea{
            font-size: 20px !important
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
    def create_section_expandable(title, content):
        with st.expander(title):
            # st.text(content)
            st.markdown(f"<div style='font-size: 20px;'>{content}</div>", unsafe_allow_html=True)

    def create_time_section_1(content):
        # st.text(content)
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: blue'><strong>Time:</strong> {content}</div>", unsafe_allow_html=True)

    def create_time_section_2(content):
        # st.text(content)
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: red'><strong>Time:</strong> {content}</div>", unsafe_allow_html=True)
    

    # Page layout
    st.title('LLM Comparison Dashboard')

    col1, divider, col2 = st.columns([5, 0.1, 5])



    # TRADITIONAL column
    with col1:

        #Response area
        st.markdown('<div class="blue-box"><h1 class="headers">LLM Traditional Response</h1></div>', unsafe_allow_html=True)
        #Use session state to hold the response
        if 'response1' not in st.session_state:
            st.session_state['response1'] = "Awaiting response..."
        #display response in text area
        st.text_area("Response-trad", st.session_state['response1'], height=200, key='response_text_area', label_visibility="hidden")


        #Metrics area
        st.markdown('<h4 class="metrics">Metrics</h4>', unsafe_allow_html=True)
        #Use session state to hold time response
        if 'elapsed_time_trad' not in st.session_state:
            st.session_state['elapsed_time_trad'] = "Awaiting response..."
        #display time
        create_time_section_1(st.session_state['elapsed_time_trad'])

        create_section_expandable("Top K (Context used)", "May use more context")
        #create_section("Cost:", "Details about cost efficiency...", key="2")
        

    # Divider
    with divider:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)



    # KNOWLEDGE GRAPH column
    with col2:
        st.markdown('<div class="red-box"><h1 class="headers">LLM with KG Response</h1></div>', unsafe_allow_html=True)
        #Use session state to hold the response
        if 'response2' not in st.session_state:
            st.session_state['response2'] = "Awaiting response..."
        #display response in text area
        st.text_area("Response-kg", st.session_state['response2'], height=200, key='response_text_area2', label_visibility="hidden")

        st.markdown('<h4 class="metrics">Metrics</h4>', unsafe_allow_html=True)
        #Use session state to hold time response
        if 'elapsed_time_kg' not in st.session_state:
            st.session_state['elapsed_time_kg'] = "Awaiting response..."
        #display time
        create_time_section_2(st.session_state['elapsed_time_kg'])

        create_section_expandable("Top K (Context used)", "Should use less context")
        #create_section("Cost:", "Details about cost efficiency...", key="5")
        


    #Use a form to take input and button interactions
    with st.form("query_form"):
        #Input for new queries
        user_input = st.text_input("Ask a question", key="query", autocomplete="off")
        #Input for top_k
        top_k = st.text_input("Enter top k amount", key="top_k", autocomplete="off")
        #Submit button for the form
        submit_button = st.form_submit_button("Send", on_click=update_response) #On click, call function to perform kg and trad search + response


if __name__== "__main__":
    main()