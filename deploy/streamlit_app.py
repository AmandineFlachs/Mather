import streamlit as st
from openai import OpenAI

st.image("assets/banner.png")

st.sidebar.header("Instructions")
st.sidebar.write("Type your questions to your AI math teacher.")

options = ["No preference", "Concise answers", "Detailed answers"]

add_radio = st.sidebar.radio(
    "Select option",
    options
)

prompt = ""

if add_radio == options[1]:
    prompt = "Give concise answers."
elif add_radio == options[2]:
    prompt = "Give detailed answers."

st.title("AI Math Teacher")

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": "You are an AI math teacher. " + prompt})

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Please enter your question."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner(text="Thinking..."):
        response = client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
            messages=st.session_state.messages, temperature=0.7
        )

    st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})

    with st.chat_message("assistant"):
        st.markdown(response.choices[0].message.content)
