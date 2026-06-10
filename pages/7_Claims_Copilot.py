import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.title('Claims Intelligence Copilot')
st.caption('Agentic AI with ICD lookup, CPT validation, and denial prediction')

# Initialize the agent
if 'agent' not in st.session_state:
    with st.spinner('Loading claims agent...'):
        from copilot.agent import ClaimsAgent
        st.session_state.agent = ClaimsAgent()
        st.session_state.agent_history = []

# Sidebar: reasoning trace
st.sidebar.header('Agent Reasoning Trace')
st.sidebar.caption('Every tool the agent called and why')

# Display chat history
for msg in st.session_state.agent_history:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# Chat input
if prompt := st.chat_input('Describe a claim or ask a question...'):
    with st.chat_message('user'):
        st.write(prompt)
    st.session_state.agent_history.append({'role': 'user', 'content': prompt})

    with st.chat_message('assistant'):
        with st.spinner('Agent is thinking...'):
            answer, trace = st.session_state.agent.run(prompt)
        st.write(answer)
    st.session_state.agent_history.append({'role': 'assistant', 'content': answer})

    # Show trace in sidebar
    if trace:
        st.sidebar.success(f'{len(trace)} tool(s) called')
        for i, entry in enumerate(trace):
            with st.sidebar.expander(f'Tool {i+1}: {entry["tool"]}'):
                st.json(entry['input'])
                st.caption('Result:')
                st.json(entry['output'])
    else:
        st.sidebar.info('No tools called — answered from knowledge')

# Reset button
if st.sidebar.button('Clear conversation'):
    st.session_state.agent.reset()
    st.session_state.agent_history = []
    st.rerun()