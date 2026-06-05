import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.title('Claims Intelligence Copilot')
st.caption('RAG-powered Q&A grounded in ICD codes, CPT codes, and claims policy')

# Initialize the copilot (cached so it only loads once per session)
if 'copilot' not in st.session_state:
    with st.spinner('Loading knowledge base...'):
        from copilot.rag_engine import ClaimsCopilot
        st.session_state.copilot = ClaimsCopilot()
        st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
        st.write(msg['content'])
        if msg['role'] == 'assistant' and 'sources' in msg:
            with st.expander('Sources used', expanded=False):
                for i, src in enumerate(msg['sources']):
                    st.caption(f'Source {i+1} (relevance: {src["score"]})')
                    st.text(src['text'])

# Chat input
if prompt := st.chat_input('Ask about ICD codes, CPT codes, or claim denials...'):
    with st.chat_message('user'):
        st.write(prompt)
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})

    with st.chat_message('assistant'):
        with st.spinner('Searching knowledge base...'):
            answer, sources = st.session_state.copilot.chat(prompt)
        st.write(answer)
        with st.expander('Sources used', expanded=False):
            for i, src in enumerate(sources):
                st.caption(f'Source {i+1} (relevance: {src["score"]})')
                st.text(src['text'])

    st.session_state.chat_history.append({
        'role': 'assistant', 'content': answer, 'sources': sources
    })

# Reset button in sidebar
if st.sidebar.button('Clear conversation'):
    st.session_state.copilot.reset()
    st.session_state.chat_history = []
    st.rerun()