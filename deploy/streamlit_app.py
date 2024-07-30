import streamlit as st
from openai import OpenAI

# Header.
st.image("assets/banner.png")
st.title("AI Math Teacher")

# Sidebar.
st.sidebar.header("Instructions")
st.sidebar.write("Type your questions to your AI math teacher.")

options = ["No preference", "Concise answers", "Detailed answers"]

add_radio = st.sidebar.radio(
    "Select option",
    options
)

# Generate prompt from sidebar options.
prompt = ""
if add_radio == options[1]:
    prompt = "Give concise answers."
elif add_radio == options[2]:
    prompt = "Give detailed answers."

# Configure client to process requests via LM Studio.
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Generate system prompt.
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": "You are an AI math teacher. " + prompt})

# Display history of questions and answers (without system prompt).
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Interaction with user.
if message := st.chat_input("Please enter your question."):
    # Add new user message to history.
    st.session_state.messages.append({"role": "user", "content": message})

    # Display new user message.
    with st.chat_message("user"):
        st.markdown(message)

    # Wait for answer from LLM.
    with st.spinner(text="Thinking..."):
        response = client.chat.completions.create(
            model="AmandineFlachs/Mather-v1-gguf",
            messages=st.session_state.messages, temperature=0.7
        )

    # Add LLM answer to history.
    st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})

    # Display new LLM answer.
    with st.chat_message("assistant"):
        st.markdown(response.choices[0].message.content)
