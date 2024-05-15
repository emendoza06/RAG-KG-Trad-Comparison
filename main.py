import streamlit as st
import time
from KGLLMResponder import GraphLLMResponder
from TradLLMResponder import TradLLMResponder

st.set_page_config(layout="wide")

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

    #Traditional response
    start_time_trad = time.time()
    trad_responder = TradLLMResponder(user_input, top_k_trad) #Create the trad llm responder mechanism
    trad_response = trad_responder.execute_query_and_respond() #Save the response
    st.session_state['response1'] = trad_response
    st.session_state['trad_context'] = trad_responder.context
    end_time_trad = time.time()

    elapsed_time_trad = end_time_trad - start_time_trad
    st.session_state['elapsed_time_trad'] = elapsed_time_trad

    #Knowledge graph response
    start_time_kg = time.time()
    kg_responder = GraphLLMResponder(user_input, top_k_kg, trad_responder.query_embedding) #Create the graph responder mechanism
    kg_response = kg_responder.execute_query_and_respond() #Save the response
    st.session_state['response2'] = kg_response #Directly update session state
    st.session_state['kg_context'] = kg_responder.context
    end_time_kg = time.time()
    elapsed_time_kg = end_time_kg - start_time_kg
    st.session_state['elapsed_time_kg'] = elapsed_time_kg
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
    def create_section_expandable(title, content, key):
        with st.expander(title):
            #Make a button so when expandable is clicked, the context will update/refresh and populate 
            #if st.button("Refresh", key=f"refresh_{key}"):
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
                formatted_content = ''.join(f"<div style='margin-bottom: 50px; color: {colors[i % len(colors)]};'>"
                                            f"{item['p.text'] if isinstance(item, dict) and 'p.text' in item else str(item)}"
                                            for i, item in enumerate(content))
                #Knowledge graph gives context in dictionary form. Trad gives in list form. If item contains {'p.text': ...}
                #then we know it's in dictionary format. Check if 'item' is a dictionary and contains the key 'p.text'. If both conditions are true
                #extract item['p.text'] to just get the value
                st.markdown(f"<div style='font-size: 20px;'>{formatted_content}</div>", unsafe_allow_html=True)
            else:
                #If not a list then print whole string
                st.markdown(f"<div style='font-size: 20px;'>{content}</div>", unsafe_allow_html=True)

    def create_time_section_1(content):
        # st.text(content)
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: blue'><strong>Time:</strong> {content}</div>", unsafe_allow_html=True)

    def create_time_section_2(content):
        # st.text(content)
        st.markdown(f"<div style='font-size: 18px; font-style: italic; color: red'><strong>Time:</strong> {content}</div>", unsafe_allow_html=True)
    

    # Page layout
    st.title('LLM Comparison Dashboard')

    col1, divider, col2 = st.columns([4.5, 0.1, 5])



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
        st.markdown('<h4 class="metrics" style="margin-top: 80px">Metrics</h4>', unsafe_allow_html=True)
        #Use session state to hold time response
        if 'elapsed_time_trad' not in st.session_state:
            st.session_state['elapsed_time_trad'] = "Awaiting response..."
        #Time
        create_time_section_1(st.session_state['elapsed_time_trad'])

        #Context
        if 'trad_context' not in st.session_state:
            st.session_state['trad_context'] = "Awaiting response..."
        create_section_expandable("Top K (Context used)", st.session_state['trad_context'], 'trad_context') #key at the end in order to distinguish between expandables
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

        #Cypher
        if 'kg_cypher' not in st.session_state:
            st.markdown('<p class="cypher">Cypher: None set.</p>', unsafe_allow_html=True)
        else:
            cypher_query = st.session_state['kg_cypher']
            st.markdown(f'<p class="cypher">Cypher: {cypher_query} </p>', unsafe_allow_html=True)

        st.markdown('<h4 class="metrics">Metrics</h4>', unsafe_allow_html=True)
        #Use session state to hold time response
        if 'elapsed_time_kg' not in st.session_state:
            st.session_state['elapsed_time_kg'] = "Awaiting response..."
        #display time
        create_time_section_2(st.session_state['elapsed_time_kg'])

        #Display context used
        if 'kg_context' not in st.session_state:
            st.session_state['kg_context'] = "Awaiting response..."
        create_section_expandable("Top K (Context used)", st.session_state['kg_context'], 'kg_context')
        #create_section("Cost:", "Details about cost efficiency...", key="5")
        


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
        submit_button = st.form_submit_button("Send", on_click=update_response) #On click, call function to perform kg and trad search + response


if __name__== "__main__":
    main()