import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

from dotenv import load_dotenv
load_dotenv()

# Wikipedia Tool
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 250)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper_wiki)

# Arxiv Tool
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max = 250)
arxiv = ArxivQueryRun(api_wrapper = api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name = "Search")

st.title('Langchain - Chat with Search')
"""
In this example,we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for setting
st.sidebar.title('Settings')
api_key = st.sidebar.text_input("🔑 Enter your Groq API Key", type="password")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role' : 'assistance', 'content' : 'Hi, I am a chatbot who can search the web. How can I help you?'}
    ]
    
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"): st.markdown(prompt)
    llm = ChatGroq(groq_api_key=api_key, model = 'gemma2-9b-it', streaming = True)
    tools = [search, arxiv, wiki]
    
    
    search_agent = initialize_agent(tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error = True)

    # Run the agent and get the response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)