ut("Email")
            login_button = st.form_submit_button(label='Login')

            if login_button:
                if authenticate_user(email):
                    st.session_state.authenticated = True
                    st.experimental_rerun()
                else:
                    st.error("Invalid email or password. Please try again.")
        return

    st.sidebar.markdown("<h4 style='color: #08daff;'>Popular Questions</h3>", unsafe_allow_html=True)
    # cols = st.columns(5)  # Create 5 columns for the buttons

    suggested_questions = [
        "What is Market Mix modelling ?",
        "What are Contribution Charts  ?",
        "Provide code examples from Robyn. ",
        "How MMMs can be calibrated and validated ?",
        "Why Frequentist MMM is better than Bayesian MMM ?"
    ]

    for i, question in enumerate(suggested_questions):
        if st.sidebar.button(question, use_container_width = True):
            st.session_state.suggested_question = question
            st.session_state.generate_response = True
            break

    # Display the conversation history in reverse order to resemble a chat interface
    chat_container = st.container()

    with chat_container:
        if(st.session_state.conversation_history == []):
            col1, col2 = st.columns([1, 8])
            with col1:
                st.image('download.png', width=30)
            with col2:
                st.write("Hello , I am MMMGPT from Aryma Labs , How can I help you ?")
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
            st.session_state.generate_response = True

    if st.session_state.generate_response and question:
        if st.session_state.query_count >= QUERY_LIMIT:
            st.warning("You have reached the limit of free queries. Please consider our pricing options for further use.")
        else:
            with st.spinner("Generating response..."):
                response, docs = user_input(question)
                output_text = response.get('output_text', 'No response')  # Extract the 'output_text' from the response
                st.session_state.chat += str(output_text)
                st.session_state.conversation_history.append((question, output_text))
                st.session_state.suggested_question = ""  # Reset the suggested question after submission
                st.session_state.query_count += 1  # Increment the query count
                st.session_state.generate_response = False
                st.rerun()

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #A9A9A9;'>Powered by: Aryma Labs</p>", unsafe_allow_html=True)

# Main function to run the app
def main():
    create_ui()

if __name__ == "__main__":
    main()
