import streamlit as st
import pandas as pd
from memory import user_input 
# Define the maximum number of free queries
QUERY_LIMIT = 100

hide_github_icon = “”"

.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
“”"
st.markdown(hide_github_icon, unsafe_allow_html=True)

# Initialize session state for tracking the number of queries, conversation history, suggested questions, and authentication
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'suggested_question' not in st.session_state:
    st.session_state.suggested_question = ""

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def authenticate_user(email):
    # Load the Excel file
    df = pd.read_excel('user.xlsx')
    # Convert the input email to lowercase
    email = email.lower()
    # Convert the emails in the dataframe to lowercase
    df['Email'] = df['Email'].str.lower()
    # Check if the email matches any entry in the file
    user = df[df['Email'] == email]
    if not user.empty:
        return True
    return False

def create_ui():
    st.markdown("<h2 style='text-align: center; color: #0adbfc;'><u>MMM GPT</u></h2>", unsafe_allow_html=True)
    st.sidebar.image("Aryma Labs Logo.jpeg")
    st.sidebar.markdown("<h2 style='color: #08daff;'>Welcome to Aryma Labs</h2>", unsafe_allow_html=True)
    st.sidebar.write("Ask a question below and get instant insights.")

    if not st.session_state.authenticated:
        st.markdown("<h3 style='color: #4682B4;'>Login</h3>", unsafe_allow_html=True)
        with st.form(key='login_form'):
            email = st.text_input("Email")
            # password = st.text_input("Password", type="password")
            login_button = st.form_submit_button(label='Login')

            if login_button:
                if authenticate_user(email):
                    st.session_state.authenticated = True
                    st.experimental_rerun()
                else:
                    st.error("Invalid email or password. Please try again.")
        return

    if not st.session_state.suggested_question:
        # Suggested questions
        st.markdown("<h3 style='color: #4682B4;'>Popular Questions</h3>", unsafe_allow_html=True)

        cols = st.columns(5)  # Create 5 columns for the buttons

        suggested_questions = [
            "What is MMM ?",
            "What are Contribution Charts ?",
            "Ways to calibrate MMM ?",
            "Tell something about Adstock.",
            "What is RCT ?"
        ]

        for i, question in enumerate(suggested_questions):
            if cols[i % 5].button(question):
                st.session_state.suggested_question = question
                st.experimental_rerun()

    # Display the conversation history in reverse order to resemble a chat interface
    chat_container = st.container()

    with chat_container:
        for q, r in st.session_state.conversation_history:
            st.markdown(f"<p style='text-align: right; color: #484f4f;'><b> {q}</b> </p>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 8])
            with col1:
                st.image('download.png', width=30)
            with col2:
                st.write(r)

    # Get user input at the bottom
    st.markdown("---")
    instr = "Ask a question:"
    with st.form(key='input_form', clear_on_submit=True):
        col1, col2 = st.columns([8, 1])
        with col1:
            if st.session_state.suggested_question:
                question = st.text_input(instr, value=st.session_state.suggested_question, key="input_question", label_visibility='collapsed')
            else:
                question = st.text_input(instr, key="input_question", placeholder=instr, label_visibility='collapsed')
        with col2:
            submit_button = st.form_submit_button(label='Chat')

        if submit_button and question:
            if st.session_state.query_count >= QUERY_LIMIT:
                st.warning("You have reached the limit of free queries. Please consider our pricing options for further use.")
            else:
                with st.spinner("Generating response..."):
                    response, _ = user_input(question)
                    output_text = response.get('output_text', 'No response')  # Extract the 'output_text' from the response
                    st.session_state.conversation_history.append((question, output_text))
                    st.session_state.suggested_question = ""  # Reset the suggested question after submission
                    st.session_state.query_count += 1  # Increment the query count
                    st.experimental_rerun()

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #A9A9A9;'>Powered by: Aryma Labs</p>", unsafe_allow_html=True)

# Main function to run the app
def main():
    create_ui()

if __name__ == "__main__":
    main()
