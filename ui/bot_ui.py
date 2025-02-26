import streamlit as st
from src.core.tools import OrderQuerySystem
from src.core.pgvector import PGVector
from src.core.embedding import EmbeddingClient
import json
from typing import Dict, List, AsyncGenerator
import asyncio
from contextlib import contextmanager


class ChatBot:
    def __init__(self):
        # Initialize resources
        self._initialize_resources()

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def _initialize_resources(self):
        """Initialize or get cached resources"""
        if "resources" not in st.session_state:
            st.session_state.resources = {
                "order_system": OrderQuerySystem(),
                "vector_db": PGVector(),
                "embedding_client": EmbeddingClient(),
            }

        self.order_system = st.session_state.resources["order_system"]
        self.vector_db = st.session_state.resources["vector_db"]
        self.embedding_client = st.session_state.resources["embedding_client"]

    def get_faq_response(self, query: str) -> Dict:
        """Get response from FAQ system"""
        try:
            # Get embedding for query
            query_embeddings = self.embedding_client.embed_documents(
                texts=[query], input_type="search_query"
            )

            if not query_embeddings:
                return {"found": False}

            # Search similar questions
            results = self.vector_db.similarity_search(
                query_embedding=query_embeddings[0], k=3, similarity_threshold=0.7
            )

            if results:
                return {
                    "found": True,
                    "answer": results[0]["answer"],
                    "similarity": results[0]["similarity"],
                }
        except Exception as e:
            print(f"FAQ error: {str(e)}")
        return {"found": False}

    async def get_streaming_response(
        self, user_input: str
    ) -> AsyncGenerator[str, None]:
        """Get streaming response from the order system"""
        try:
            # First check FAQ
            faq_response = self.get_faq_response(user_input)
            #
            if faq_response["found"] and faq_response["similarity"] > 0.7:
                yield faq_response["answer"]
                return

            # Use order system for streaming response
            async for chunk in self.order_system.process_query_stream(user_input):
                yield chunk

        except Exception as e:
            print(f"Response error: {str(e)}")
            yield "I encountered an error. Please try again."

    def display_chat(self):
        """Display chat interface"""
        # Chat header
        st.title("Gundam Store Assistant")
        st.write("Ask about your orders or any questions about our products!")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response with streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Create and run async response generator
                async def process_response():
                    nonlocal full_response
                    async for chunk in self.get_streaming_response(prompt):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                
                # Run the async function
                asyncio.run(process_response())
                
                # Update final response
                message_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

    def display_sidebar(self):
        """Display sidebar with additional information"""
        with st.sidebar:
            st.title("Help & Information")
            with st.expander("How to use", expanded=True):
                st.markdown(
                    """
                ### How to use:
                1. **Check Orders**: 
                   - Ask about your orders
                   - Provide your email when asked
                
                2. **Cancel Orders**:
                   - Request to cancel a pending order
                   - Provide your email and order ID
                   - Only pending orders can be cancelled
                
                3. **FAQ**:
                   - Ask general questions
                   - Get instant answers
                """
                )

            with st.expander("Example Questions", expanded=True):
                st.markdown(
                    """
                - "Show me my orders for example@email.com"
                - "Cancel my order #ORD-123"
                - "What's the status of my order?"
                - "What's your return policy?"
                """
                )

            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()


def main():
    try:
        # Set page config
        st.set_page_config(
            page_title="Gundam Store Assistant",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Initialize chatbot
        bot = ChatBot()

        # Display sidebar
        bot.display_sidebar()

        # Display chat interface
        bot.display_chat()

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        if st.button("Restart Application"):
            st.session_state.clear()
            st.rerun()


if __name__ == "__main__":
    main()
