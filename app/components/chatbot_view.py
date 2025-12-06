# app/components/chatbot_view.py

def render_chatbot_view(chatbot, chat_history):
    """Render AI chatbot interface"""
    
    st.header("💬 AI Investment Advisor")
    
    if chatbot is None:
        st.error("""
        ⚠️ **AI Advisor Not Available**
        
        Please configure your Groq API key:
        1. Create a `.env` file in the project root
        2. Add: `GROQ_API_KEY=your_api_key_here`
        3. Get your key from: https://console.groq.com/
        """)
        return
    
    st.success("✅ AI Advisor is ready to help with your investment questions!")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for message in chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
    
    # Chat input
    user_input = st.chat_input("Ask about real estate investing...")
    
    if user_input:
        # Add user message
        chat_history.append({'role': 'user', 'content': user_input})
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = chatbot.chat(user_input)
        
        # Add assistant response
        chat_history.append({'role': 'assistant', 'content': response})
        
        st.rerun()
    
    # Sidebar with quick questions
    with st.sidebar:
        st.markdown("### 💡 Quick Questions")
        
        quick_questions = [
            "What's a good ROI for rental properties?",
            "How do I calculate rental yield?",
            "Urban vs suburban investing?",
            "What affects property appreciation?",
            "How to evaluate cash flow?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"q_{hash(question)}"):
                chat_history.append({'role': 'user', 'content': question})
                response = chatbot.chat(question)
                chat_history.append({'role': 'assistant', 'content': response})
                st.rerun()
        
        if st.button("🔄 Clear Chat"):
            chat_history.clear()
            chatbot.reset_conversation()
            st.rerun()