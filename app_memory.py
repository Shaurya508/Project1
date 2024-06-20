import streamlit as st
from memory import user_input
import time

# Initialize the session state for conversation history if not already initialized
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def create_ui():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'><u>Aryma MMM GPT!</u></h1>", unsafe_allow_html=True)
    st.sidebar.image("Aryma Labs Logo.jpeg", use_column_width=True)
    st.sidebar.markdown("<h2 style='color: #4CAF50;'>Welcome to Aryma Labs</h2>", unsafe_allow_html=True)
    st.sidebar.write("Ask a question below and get instant insights.")

    # Add some instructions
    # st.markdown("<h3 style='color: #FF6347;'>Instructions</h3>", unsafe_allow_html=True)
    # st.markdown(
    #     """
    #     1. Enter your question in the text box below.
    #     2. Click on 'Submit' to get the response.
    #     3. View the answer generated based on the cool stuff from Aryma Labs.
    #     """
    # )

    # Display the conversation history in reverse order to resemble a chat interface
    # st.markdown("<h3 style='color: #4682B4;'>Chat History</h3>", unsafe_allow_html=True)
    chat_container = st.container()

    with chat_container:
        for q, r in st.session_state.conversation_history:
            # st.markdown(
            #     f"""
            #     <div style='text-align: left; margin: 10px; padding: 5px;'>
            #         <strong style='color: #FF4500;'>User:</strong> {q}
            #     </div>
            #     <div style='text-align: left; margin: 10px; padding: 5px;'>
            #         <strong style='color: #2E8B57;'>MMM GPT:</strong> {r}
            #     </div>
            #     """,
            #     unsafe_allow_html=True
            # )
            st.success("User:")
            st.write(q)
            st.success("MMM GPT :")
            st.write(r)

    # Get user input at the bottom
    st.markdown("---")
    
    with st.form(key='input_form', clear_on_submit=True):
        question = st.text_input("Ask a question:", key="input_question")
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button and question:
            with st.spinner("Generating response..."):
                # start_time = time.time()
                response , _ = user_input(question)
                # end_time = time.time()
                output_text = response.get('output_text', 'No response')  # Extract the 'output_text' from the response
                # output_text = response[0]
                # Append the question and response to the session state conversation history
                st.session_state.conversation_history.append((question, output_text))
                
            
                # Clear the text input after submitting the question
                st.rerun()
                


    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #A9A9A9;'>Powered by: Shaurya Mishra</p>", unsafe_allow_html=True)

# Main function to run the app
def main():
    create_ui()

if __name__ == "__main__":
    main()
