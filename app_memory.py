import streamlit as st
from memory import user_input
import time

# Initialize the session state for conversation history and suggested questions if not already initialized
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'suggested_question' not in st.session_state:
    st.session_state.suggested_question = ""

def create_ui():
    st.markdown("<h1 style='text-align: center; color: #08daff;'><u>Aryma MMM GPT</u></h1>", unsafe_allow_html=True)
    st.sidebar.image("Aryma Labs Logo.jpeg", use_column_width=True)
    st.sidebar.markdown("<h2 style='color: #08daff;'>Welcome to Aryma Labs</h2>", unsafe_allow_html=True)
    st.sidebar.write("Ask a question below and get instant insights.")

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
            if cols[i % 5].checkbox(question) :
                st.session_state.suggested_question = question
                st.experimental_rerun()

    # Display the conversation history in reverse order to resemble a chat interface
    chat_container = st.container()

    with chat_container:
        for q, r in st.session_state.conversation_history:
            st.success("You:")
            st.write(q)
            st.success("MMM GPT:")
            st.write(r)

    # Get user input at the bottom
    st.markdown("---")
    instr = "Ask a question:"
    with st.form(key='input_form', clear_on_submit=True):
        col1 , col2 = st.columns([8,1])
        with col1 :
            if st.session_state.suggested_question:
                question = st.text_input(instr, value=st.session_state.suggested_question, key="input_question" ,label_visibility='collapsed')
            else:
                question = st.text_input(instr, key="input_question" ,placeholder=instr , label_visibility='collapsed')
        with col2 :
            submit_button = st.form_submit_button(label='Chat')

        if submit_button and question:
            with st.spinner("Generating response..."):
                response, _ = user_input(question)
                output_text = response.get('output_text', 'No response')  # Extract the 'output_text' from the response
                st.session_state.conversation_history.append((question, output_text))
                st.session_state.suggested_question = ""  # Reset the suggested question after submission
                st.experimental_rerun()

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #A9A9A9;'>Powered by: Aryma Labs</p>", unsafe_allow_html=True)

# Main function to run the app
def main():
    create_ui()

if __name__ == "__main__":
    main()
