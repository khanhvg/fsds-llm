from typing import List, Dict, Optional, Deque, Literal
from collections import deque
from langchain_aws import ChatBedrock
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
import psycopg
import json
from pydantic import BaseModel, EmailStr
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import asyncio


class OrderLookupInput(BaseModel):
    """Input for order lookup"""

    email: EmailStr


class OrderCancelInput(BaseModel):
    """Input for order cancellation"""

    email: EmailStr
    order_id: str


class ProductSearchInput(BaseModel):
    """Input for product search"""

    query: str


class IntentResponse(BaseModel):
    """Response model for intent classification"""

    intent: Literal["CHECK_ORDERS", "CANCEL_ORDER", "FAQ", "PRODUCT_SEARCH", "PURCHASE"]
    confidence: float
    email: Optional[str] = None
    order_id: Optional[str] = None
    product_query: Optional[str] = None


def get_db_connection():
    """Create database connection"""
    return psycopg.connect("postgresql://postgres:postgres@localhost:5432/postgres")


def get_products(query: str) -> List[Dict]:
    """Get products from the database"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                results = []
                for row in cur.fetchall():
                    results.append(
                        {
                            "product_id": row[0],
                            "product_name": row[1],
                            "product_description": row[2],
                            "product_price": row[3],
                            "product_category": row[4],
                            "product_series": row[5],
                            "product_scale": row[6],
                            "product_grade": row[7],
                            "product_release_date": row[8].isoformat() if row[8] else None,
                            "product_stock_quantity": row[9],
                            "product_image_url": row[10],
                        }
                    )
                return results
    except Exception as e:
        print(f"Database error: {str(e)}")
        return []


def get_customer_orders(email: str) -> List[Dict]:
    """Get orders for a given email"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        order_id,
                        order_detail,
                        total_price,
                        status,
                        created_at
                    FROM orders
                    WHERE customer_email = %s
                    ORDER BY created_at DESC
                    """,
                    (email,),
                )

                results = []
                for row in cur.fetchall():
                    order_detail = row[1]
                    if isinstance(order_detail, str):
                        order_detail = json.loads(order_detail)

                    results.append(
                        {
                            "order_id": row[0],
                            "order_detail": order_detail,
                            "total_price": str(float(row[2])),
                            "status": row[3],
                            "created_at": row[4].isoformat(),
                        }
                    )
                return results
    except Exception as e:
        print(f"Database error: {str(e)}")
        return []


def cancel_order(input_data: OrderCancelInput) -> Dict:
    """Cancel an order if it's in pending status"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # First check if order exists and is cancellable
                cur.execute(
                    """
                    SELECT status
                    FROM orders 
                    WHERE customer_email = %s AND order_id = %s
                    """,
                    (input_data.email, input_data.order_id),
                )

                result = cur.fetchone()
                if not result:
                    return {
                        "success": False,
                        "message": "Order not found. Please verify your order ID and email.",
                    }

                status = result[0]
                if status.lower() != "pending":
                    return {
                        "success": False,
                        "message": f"Cannot cancel order with status: {status}. Only pending orders can be cancelled.",
                    }

                # Update order status to cancelled
                cur.execute(
                    """
                    UPDATE orders 
                    SET status = 'cancelled',
                        updated_at = %s
                    WHERE customer_email = %s AND order_id = %s
                    RETURNING order_id
                    """,
                    (datetime.now(), input_data.email, input_data.order_id),
                )

                conn.commit()
                return {
                    "success": True,
                    "message": f"Order {input_data.order_id} has been cancelled successfully.",
                }

    except Exception as e:
        print(f"Database error: {str(e)}")
        return {
            "success": False,
            "message": "An error occurred while cancelling the order. Please try again.",
        }


class ChatMessage:
    """Class to represent a chat message"""

    def __init__(self, role: str, content: str, timestamp: datetime = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


class ConversationMemory:
    """Class to manage conversation history"""

    def __init__(self, max_messages: int = 5):
        self.messages: Deque[ChatMessage] = deque(maxlen=max_messages)
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        """Add a new message to the history"""
        message = ChatMessage(role, content)
        self.messages.append(message)

    def get_history(self, as_dict: bool = False) -> List:
        """Get conversation history"""
        if as_dict:
            return [msg.to_dict() for msg in self.messages]
        return list(self.messages)

    def get_context_string(self) -> str:
        """Get conversation history as a formatted string"""
        return json.dumps([msg.to_dict() for msg in self.messages])

    def clear(self):
        """Clear conversation history"""
        self.messages.clear()


class OrderQuerySystem:
    def __init__(self):
        self.llm = ChatBedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region="ap-northeast-1",
            temperature=0,
            streaming=True,
        )
        
        # Initialize memory
        self.memory = ConversationMemory(max_messages=10)
        
        # Create tools
        self._setup_tools()
        # Create chains
        self._setup_chains()

    def _setup_tools(self):
        """Initialize all tools"""
        self.tools = [
            StructuredTool.from_function(
                func=get_customer_orders,
                name="lookup_orders",
                description="Look up orders for a customer email",
            ),
            StructuredTool.from_function(
                func=cancel_order,
                name="cancel_order",
                description="Cancel a pending order for a customer",
            ),
            StructuredTool.from_function(
                func=get_products,
                name="search_products",
                description="Search for products in the database",
						)
        ]

    def _setup_chains(self):
        """Initialize all conversation chains"""
        self.chains = {
            "intent": self._create_intent_chain(),
            "email": self._create_email_chain(),
            "order_id": self._create_order_id_chain(),
            "response": self._create_response_chain(),
            "conversation": self._create_conversation_chain(),
            "response_formatter": self._create_response_formatter_chain(),
            "chat": self._create_chat_chain(),
            "product_search": self._create_product_search_chain(),
            "purchase_info_extractor": self._create_purchase_info_extractor_chain(),
            "purchase_summarizer": self._create_purchase_summarizer_chain()
        }

    def _create_product_search_chain(self):
        """Create chain for extracting product search parameters"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a product search assistant for a Gundam store.
            Extract search parameters from the user's query.
            
            Possible parameters to extract:
            - series: The Gundam series (e.g., "Mobile Suit Gundam", "Gundam Wing", "Gundam SEED")
            - grade: The model grade (e.g., "High Grade", "Master Grade", "Perfect Grade", "Real Grade")
            - scale: The model scale (e.g., "1/144", "1/100", "1/60")
            - price_range: Any price constraints mentioned (e.g., "under $20", "less than $50", "between $30 and $60")
            - other_keywords: Any other relevant search terms
            
            Respond in JSON format with these fields:
            {{
                "series": "<series name if mentioned, null if not>",
                "grade": "<grade if mentioned, null if not>",
                "scale": "<scale if mentioned, null if not>",
                "price_range": {{"min": <minimum price if mentioned, null if not>, "max": <maximum price if mentioned, null if not>}},
                "other_keywords": ["<keyword1>", "<keyword2>", ...],
                "sql_query": "<SQL WHERE clause based on the extracted parameters>"
            }}
            
            For the SQL query, construct a WHERE clause that can be used in a SQL query against a products table.
            The table has columns: name, description, price, category, series, scale, grade, release_date, stock_quantity
            
            Pay special attention to price constraints:
            - "under $20" should be translated to "price < 20"
            - "less than $50" should be translated to "price < 50"
            - "between $30 and $60" should be translated to "price >= 30 AND price <= 60"
            
            Examples:
            - "Show me Gundam Wing models" -> {{"series": "Mobile Suit Gundam Wing", "grade": null, "scale": null, "price_range": {{"min": null, "max": null}}, "other_keywords": [], "sql_query": "series ILIKE '%Gundam Wing%'"}}
            - "I'm looking for Perfect Grade kits" -> {{"series": null, "grade": "Perfect Grade", "scale": null, "price_range": {{"min": null, "max": null}}, "other_keywords": [], "sql_query": "grade ILIKE '%Perfect Grade%'"}}
            - "Do you have 1/100 scale Gundam SEED models?" -> {{"series": "Mobile Suit Gundam SEED", "scale": "1/100", "grade": null, "price_range": {{"min": null, "max": null}}, "other_keywords": [], "sql_query": "series ILIKE '%Gundam SEED%' AND scale = '1/100'"}}
            - "Show me Gundam models under $20" -> {{"series": null, "grade": null, "scale": null, "price_range": {{"min": null, "max": 20}}, "other_keywords": [], "sql_query": "price < 20"}}
            - "I want to see High Grade kits between $15 and $30" -> {{"series": null, "grade": "High Grade", "scale": null, "price_range": {{"min": 15, "max": 30}}, "other_keywords": [], "sql_query": "grade ILIKE '%High Grade%' AND price >= 15 AND price <= 30"}}
            """),
            ("human", "{input}")
        ]) | self.llm | StrOutputParser()

    def _create_intent_chain(self):
        """Create chain for intent classification"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for a Gundam store assistant.
            Analyze the user's message and determine their intent.
            
            Consider the conversation history for context:
            {history}
            
            Classify the intent into one of these categories:
            1. CHECK_ORDERS: User wants to view their order history or status
            2. CANCEL_ORDER: User wants to cancel an order
            3. FAQ: General questions about store policies, shipping, etc.
            4. PRODUCT_SEARCH: User wants to search for Gundam products by series, grade, scale, etc.
            5. PURCHASE: User wants to buy/purchase a Gundam product
            
            Also extract relevant entities:
            - email: Email address if present
            - order_id: Order ID if present
            - product_query: Any product-related search terms (series name, grade, scale, etc.)
            
            Respond in JSON format with these exact fields:
            {{
                "intent": "CHECK_ORDERS" | "CANCEL_ORDER" | "FAQ" | "PRODUCT_SEARCH" | "PURCHASE",
                "confidence": <float between 0 and 1>,
                "email": "<email if found, null if not>",
                "order_id": "<order id if found, null if not>",
                "product_query": "<product search terms if found, null if not>"
            }}
            
            Examples:
            - "I want to cancel order #123" -> {{"intent": "CANCEL_ORDER", "confidence": 0.95, "email": null, "order_id": "123", "product_query": null}}
            - "Show my orders for test@email.com" -> {{"intent": "CHECK_ORDERS", "confidence": 0.9, "email": "test@email.com", "order_id": null, "product_query": null}}
            - "What's your return policy?" -> {{"intent": "FAQ", "confidence": 0.8, "email": null, "order_id": null, "product_query": null}}
            - "Show me Gundam Wing models" -> {{"intent": "PRODUCT_SEARCH", "confidence": 0.9, "email": null, "order_id": null, "product_query": "Gundam Wing"}}
            - "I'd like to buy a Perfect Grade Gundam" -> {{"intent": "PURCHASE", "confidence": 0.95, "email": null, "order_id": null, "product_query": "Perfect Grade Gundam"}}
            - "I want to purchase RX-78-2" -> {{"intent": "PURCHASE", "confidence": 0.95, "email": null, "order_id": null, "product_query": "RX-78-2"}}
            """),
            ("human", "{input}")
        ]) | self.llm | StrOutputParser()

    def _create_email_chain(self):
        """Create chain for email extraction"""
        return (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Extract an email address from the message if present.
            Consider the conversation history for context:
            {history}
            
            If you find an email, respond with just the email address.
            If no email is found, respond with 'None'.
            Following the language of the message, respond in the same language.""",
                    ),
                    ("human", "{input}"),
                ]
            )
            | self.llm
        )

    def _create_order_id_chain(self):
        """Create chain for order ID extraction"""
        return (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Extract the order ID from the message if present.
            Consider the conversation history for context:
            {history}
            
            Look for patterns like:
            - Order numbers (e.g., ORD-123, #123)
            - Direct mentions of order IDs
            - References to specific orders
            
            If found, return just the order ID.
            If not found, return 'None'.
            Following the language of the message, respond in the same language.""",
                    ),
                    ("human", "{input}"),
                ]
            )
            | self.llm
        )

    def _create_response_chain(self):
        """Create chain for formatting order lookup responses"""
        return (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a helpful Gundam store assistant.
            Format the order information in a clear, organized manner.
            Include:
            - Order ID
            - Status
            - Total price
            - Key items
            - Order date
            
            Consider the conversation history for context:
            {history}
            
            Be concise but friendly.
            Following the language of the message, respond in the same language.""",
                    ),
                    ("human", "Here are the orders: {orders}"),
                ]
            )
            | self.llm
        )

    def _create_conversation_chain(self):
        """Create chain for general conversation"""
        return (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a helpful Gundam store assistant.
            Consider the conversation history for context:
            {history}
            
            - If asking for email, be polite and clear
            - If user seems confused, explain how you can help
            - Keep responses friendly but professional
            - Focus on order-related queries
            Following the language of the message, respond in the same language.""",
                    ),
                    ("human", "{input}"),
                ]
            )
            | self.llm
        )

    def _create_chain_with_streaming(self, prompt):
        """Create a chain that supports streaming"""
        return (
            {"message": RunnablePassthrough(), "history": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _create_response_formatter_chain(self):
        """Create chain for formatting responses with streaming"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful Gundam store assistant.
            Format the given message in a natural, conversational way.
            Keep the same meaning but make it sound more friendly and natural.
            
            Consider the conversation history for context:
            {history}
            
            Important:
            - Maintain the same language as the user's query
            - Keep the key information intact
            - Be polite and professional
            - Use appropriate cultural context
            """,
                ),
                ("human", "Please format this message: {message}"),
            ]
        )
        return self._create_chain_with_streaming(prompt)

    def _create_chat_chain(self):
        """Create chain for general chatting"""
        return (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a friendly Gundam store assistant who loves to chat about Gundam.
            Consider the conversation history for context:
            {history}
            
            Guidelines for chatting:
            - Be friendly and enthusiastic about Gundam
            - Share interesting facts about Gundam models when relevant
            - Stay in character as a Gundam store assistant
            - If the conversation moves towards orders, guide them to the order-related features
            - Keep responses concise but engaging
            - Match the user's language and tone
            - You can discuss:
              * Different Gundam series and models
              * Basic Gundam lore and history
              * Gundam building tips
              * Popular Gundam models
              * General Gundam-related topics
            
            Remember:
            - Keep the conversation Gundam-focused
            - Be helpful but maintain professional boundaries
            - If unsure, guide back to store-related topics
            - Following the language of the message, respond in the same language.
            """,
                    ),
                    ("human", "{input}"),
                ]
            )
            | self.llm
        )

    def _create_purchase_info_extractor_chain(self):
        """Create chain for extracting purchase information"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an information extraction assistant for a Gundam store.
            Extract purchase information from the conversation.
            
            Extract the following fields if present:
            - product_name: The name of the Gundam product the customer wants to buy
            - customer_name: The customer's full name
            - phone: The customer's phone number
            - email: The customer's email address
            - address: The customer's shipping address
            - quantity: The number of units the customer wants to buy (default to 1 if not specified)
            
            Respond in JSON format with these fields:
            {{
                "product_name": "<product name if mentioned, null if not>",
                "customer_name": "<customer name if mentioned, null if not>",
                "phone": "<phone number if mentioned, null if not>",
                "email": "<email if mentioned, null if not>",
                "address": "<address if mentioned, null if not>",
                "quantity": <quantity if mentioned, 1 if not specified, null if unclear>
            }}
            """),
            ("human", "{conversation}")
        ]) | self.llm | StrOutputParser()

    def _create_purchase_summarizer_chain(self):
        """Create chain for summarizing purchase information"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a customer service representative for a Gundam store.
            Create a friendly, professional purchase confirmation summary.
            
            The summary should:
            1. Thank the customer for their order
            2. Summarize all the order details
            3. Provide next steps (payment link, estimated shipping time)
            4. Ask if they need any other assistance
            """),
            ("human", "{summary_prompt}")
        ]) | self.llm

    async def _format_response_stream(self, message: str, history: str):
        """Format response using LLM with streaming"""
        try:
            async for chunk in self.chains["response_formatter"].astream(
                {"message": message, "history": history}
            ):
                yield chunk
        except Exception as e:
            print(f"Error formatting response: {str(e)}")
            yield message

    def _format_response(self, message: str, history: str) -> str:
        """Format response using LLM"""
        try:
            response = self.chains["response_formatter"].invoke(
                {"message": message, "history": history}
            )
            return response
        except Exception as e:
            print(f"Error formatting response: {str(e)}")
            return message

    async def process_query_stream(self, user_input: str):
        """Process user query with streaming response"""
        try:
            # Add user message to memory
            self.memory.add_message("user", user_input)
            history = self.memory.get_context_string()
            
            # Determine intent
            intent_result = json.loads(self.chains["intent"].invoke({
                "input": user_input,
                "history": history
            }))
            
            # Route to appropriate handler based on intent
            if intent_result["intent"] == "CANCEL_ORDER":
                response = await self._handle_cancellation(user_input, history, intent_result)
            elif intent_result["intent"] == "CHECK_ORDERS":
                response = self._handle_order_lookup(user_input, history, intent_result)
            elif intent_result["intent"] == "PRODUCT_SEARCH":
                response = self._handle_product_search(user_input, history, intent_result)
            elif intent_result["intent"] == "PURCHASE":
                response = self._handle_purchase(user_input, history, intent_result)
            elif intent_result["intent"] == "FAQ":
                response = "I understand you have a general question. I'm currently configured to help with order-related queries and product searches. Feel free to ask about orders or search for Gundam products!"
            else:
                response = "I'm here to help with order management and product searches. How can I assist you today?"

            # Stream formatted response
            async for chunk in self._format_response_stream(response, history):
                yield chunk
                
            # Save complete response to memory
            self.memory.add_message("assistant", response)

        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            error_msg = "I encountered an error processing your request. Please try again."
            async for chunk in self._format_response_stream(error_msg, history):
                yield chunk

    async def _handle_cancellation(
        self, user_input: str, history: str, intent_result: Dict
    ) -> str:
        """Handle order cancellation flow"""
        email = intent_result.get("email")
        order_id = intent_result.get("order_id")

        # If we don't have email, ask for it
        if not email:
            email_result = (
                self.chains["email"]
                .invoke({"input": user_input, "history": history})
                .content.strip()
            )
            if email_result.lower() != "none":
                email = email_result

        # If we don't have order ID, ask for it
        if not order_id:
            order_id_result = (
                self.chains["order_id"]
                .invoke({"input": user_input, "history": history})
                .content.strip()
            )
            if order_id_result.lower() != "none":
                order_id = order_id_result

        # If we have both pieces of information, proceed with cancellation
        if email and order_id:
            try:
                # Create the input data dictionary
                input_data = {"email": email, "order_id": order_id}
                result = cancel_order(OrderCancelInput(**input_data))
                return result["message"]
            except Exception as e:
                print(f"Cancellation error: {str(e)}")
                return "I encountered an error while trying to cancel the order. Please try again."
        else:
            missing = []
            if not email:
                missing.append("email address")
            if not order_id:
                missing.append("order ID")
            return f"To cancel your order, I'll need your {' and '.join(missing)}. Could you please provide {'them' if len(missing) > 1 else 'it'}?"

    def _handle_order_lookup(
        self, user_input: str, history: str, intent_result: Dict
    ) -> str:
        """Handle order lookup flow"""
        email = intent_result.get("email")
        
        if not email:
            email_result = (
                self.chains["email"]
                .invoke({"input": user_input, "history": history})
                .content.strip()
            )
            if email_result.lower() != "none":
                email = email_result

        if not email:
            return "I'll help you check your orders. Could you please provide your email address?"

        try:
            orders = self.tools[0].invoke(email)
            if not orders:
                return f"I couldn't find any orders associated with {email}. Please verify your email address."
            
            base_response = (
                self.chains["response"]
                .invoke({"orders": json.dumps(orders), "history": history})
                .content
            )
            return base_response
        except Exception as e:
            print(f"Order lookup error: {str(e)}")
            return (
                "I encountered an error while looking up your orders. Please try again."
            )
        
    def _handle_product_search(self, user_input: str, history: str, intent_result: Dict) -> str:
        """Handle product search queries"""
        product_query = intent_result.get("product_query")
        
        if not product_query:
            return "I'd be happy to help you find Gundam products. Could you please tell me what you're looking for? You can specify series, grade, scale, or any other details."
        
        try:
            # Extract search parameters
            search_params = json.loads(self.chains["product_search"].invoke({
                "input": user_input,
                "history": history
            }))
            
            # Build SQL query
            base_query = """
                SELECT 
                    product_id, 
                    name, 
                    description, 
                    price, 
                    category, 
                    series, 
                    scale, 
                    grade, 
                    release_date, 
                    stock_quantity, 
                    image_url 
                FROM products
            """
            
            where_clause = search_params.get("sql_query", "")
            if where_clause:
                query = f"{base_query} WHERE {where_clause} ORDER BY release_date DESC LIMIT 15"
            else:
                # Fallback to basic search if no specific parameters
                query = f"{base_query} WHERE name ILIKE '%{product_query}%' OR description ILIKE '%{product_query}%' ORDER BY release_date DESC LIMIT 15"
            
            print(f"Executing query: {query}")  # Debug log
            
            # Execute query
            products = get_products(query)
            
            if not products:
                # Try a more relaxed search if no results
                price_range = search_params.get("price_range", {})
                if price_range and (price_range.get("min") is not None or price_range.get("max") is not None):
                    # If we had price constraints but no results, mention that specifically
                    return f"I couldn't find any products matching your price requirements. Would you like to see products in a different price range?"
                return f"I couldn't find any products matching '{product_query}'. Would you like to try a different search?"
            
            # Format response
            response = f"I found {len(products)} products matching your search:\n\n"
            
            for i, product in enumerate(products, 1):
                response += f"{i}. {product['product_name']}\n"
                response += f"   Price: ${float(product['product_price']):.2f}\n"
                response += f"   Grade: {product['product_grade']}\n"
                response += f"   Scale: {product['product_scale']}\n"
                response += f"   Series: {product['product_series']}\n"
                response += f"   Stock: {product['product_stock_quantity']} units\n\n"
            
            # Add price range information if that was part of the search
            price_range = search_params.get("price_range", {})
            if price_range and (price_range.get("min") is not None or price_range.get("max") is not None):
                min_price = price_range.get("min")
                max_price = price_range.get("max")
                if min_price is not None and max_price is not None:
                    response += f"These products are priced between ${min_price} and ${max_price}.\n\n"
                elif max_price is not None:
                    response += f"These products are priced under ${max_price}.\n\n"
                elif min_price is not None:
                    response += f"These products are priced above ${min_price}.\n\n"
            
            response += "Would you like more details about any of these products or would you like to refine your search?"
            
            return response
            
        except Exception as e:
            print(f"Product search error: {str(e)}")
            return "I encountered an error while searching for products. Please try again with a different query."
        
    def _handle_purchase(self, user_input: str, history: str, intent_result: Dict) -> str:
        """Handle purchase request flow"""
        # Check if we have all required information in memory
        purchase_info = self._extract_purchase_info(history + "\n" + user_input)
        
        # Determine what information is missing
        missing_fields = []
        if not purchase_info.get('product_name'):
            missing_fields.append('product_name')
        if not purchase_info.get('customer_name'):
            missing_fields.append('customer_name')
        if not purchase_info.get('phone'):
            missing_fields.append('phone')
        if not purchase_info.get('email'):
            missing_fields.append('email')
        if not purchase_info.get('address'):
            missing_fields.append('address')
        if not purchase_info.get('quantity') or purchase_info.get('quantity') < 1:
            missing_fields.append('quantity')
        
        # If we're missing information, ask for it
        if missing_fields:
            if 'product_name' in missing_fields:
                return "I'd be happy to help you with your purchase! Which Gundam model would you like to buy?"
            elif 'customer_name' in missing_fields:
                return f"Great choice! Could you please provide your full name for the order?"
            elif 'phone' in missing_fields:
                return f"Thanks! Could you please provide your phone number?"
            elif 'email' in missing_fields:
                return f"Could you please provide your email address for order confirmation?"
            elif 'address' in missing_fields:
                return f"Could you please provide your shipping address?"
            elif 'quantity' in missing_fields:
                return f"How many units of {purchase_info.get('product_name')} would you like to purchase?"
        
        # Look up the product price from database
        product_price = self._get_product_price(purchase_info.get('product_name'))
        quantity = purchase_info.get('quantity', 1)
        total_price = product_price * quantity
        
        # Add price information to purchase info
        purchase_info['unit_price'] = product_price
        purchase_info['total_price'] = total_price
        
        # If we have all the required information, create an order summary
        return self._create_purchase_summary(purchase_info)

    def _extract_purchase_info(self, conversation_text: str) -> Dict:
        """Extract purchase information from conversation"""
        # The chain returns a string that contains JSON
        result = self.chains["purchase_info_extractor"].invoke({
            "conversation": conversation_text
        })
        
        # Parse the JSON string into a dictionary
        try:
            if isinstance(result, str):
                # Check if there's explanatory text before the JSON
                json_start = result.find('{')
                json_end = result.rfind('}')
                if json_start >= 0 and json_end >= 0:
                    # Extract just the JSON part
                    json_str = result[json_start:json_end+1]
                    return json.loads(json_str)
                return json.loads(result)
            return result  # In case it's already a dictionary
        except json.JSONDecodeError as e:
            print(f"Error parsing purchase info JSON: {str(e)}")
            print(f"Raw result: {result}")
            # Return an empty dictionary as fallback
            return {}

    def _get_product_price(self, product_name: str) -> float:
        """Get the price of a product from database"""
        try:
            if not product_name:
                print("Warning: Empty product name provided for price lookup")
                return 0.0
            
            # Clean product name for search query
            clean_name = product_name.replace("'", "''")  # Escape single quotes for SQL
            
            # Try different search strategies
            search_queries = [
                # Strategy 1: Try exact match
                f"SELECT price, name FROM products WHERE name = '{clean_name}' LIMIT 1",
                
                # Strategy 2: Try ILIKE with model number/code match
                f"SELECT price, name FROM products WHERE name ILIKE '%{clean_name}%' LIMIT 1",
                
                # Strategy 3: Break down the product name and search for key components
                # This helps with cases where the model number is correctly identified but other details differ
                f"""
                    SELECT price, name FROM products 
                    WHERE {' OR '.join([f"name ILIKE '%{word}%'" for word in clean_name.split() if len(word) > 2])}
                    ORDER BY 
                        CASE 
                            WHEN name ILIKE '%{clean_name}%' THEN 1 
                            ELSE 2 
                        END,
                        release_date DESC
                    LIMIT 1
                """
            ]
            
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    for query in search_queries:
                        cur.execute(query)
                        result = cur.fetchone()
                        if result:
                            price, name = float(result[0]), result[1]
                            print(f"Found product: {name} with price: ${price:.2f}")
                            return price
                
                    # If all searches fail, try one last general search with any keyword
                    print(f"Couldn't find exact match for '{product_name}', trying general search")
                    keywords = [word for word in clean_name.split() if len(word) > 3]
                    if keywords:
                        final_query = f"""
                            SELECT price, name FROM products 
                            WHERE {' OR '.join([f"name ILIKE '%{word}%'" for word in keywords])}
                            ORDER BY release_date DESC
                            LIMIT 1
                        """
                        cur.execute(final_query)
                        result = cur.fetchone()
                        if result:
                            price, name = float(result[0]), result[1]
                            print(f"Found similar product: {name} with price: ${price:.2f}")
                            return price
                
                    # If still not found, search by grade/type
                    grade_types = ["perfect grade", "master grade", "high grade", "real grade", "sd", "super deformed"]
                    for grade in grade_types:
                        if grade.lower() in clean_name.lower():
                            grade_query = f"SELECT AVG(price) FROM products WHERE grade ILIKE '%{grade}%'"
                            cur.execute(grade_query)
                            avg_result = cur.fetchone()
                            if avg_result and avg_result[0]:
                                avg_price = float(avg_result[0])
                                print(f"Using average price for {grade}: ${avg_price:.2f}")
                                return avg_price
                
                    # Last resort: use default price based on price ranges in the database
                    cur.execute("SELECT AVG(price) FROM products")
                    avg_result = cur.fetchone()
                    if avg_result and avg_result[0]:
                        default_price = float(avg_result[0])
                        print(f"Using database average price: ${default_price:.2f}")
                        return default_price
                
                    # Absolute last resort: hardcoded default
                    print(f"No product found for '{product_name}', using default price")
                    return 30.0
                
        except Exception as e:
            print(f"Error getting product price: {str(e)}")
            # Default price if lookup fails
            return 30.0

    def _create_purchase_summary(self, purchase_info: Dict) -> str:
        """Create a summary of the purchase information"""
        # Format prices with two decimal places
        unit_price = purchase_info.get('unit_price', 0.0)
        quantity = purchase_info.get('quantity', 1)
        total_price = purchase_info.get('total_price', unit_price * quantity)
        
        # Format as currency
        unit_price_str = f"${unit_price:.2f}"
        total_price_str = f"${total_price:.2f}"
        
        summary_prompt = f"""
        Please generate a purchase confirmation summary using the following information:
        
        Product: {purchase_info.get('product_name', 'Not provided')}
        Unit Price: {unit_price_str}
        Quantity: {quantity}
        Total Price: {total_price_str}
        Customer Name: {purchase_info.get('customer_name', 'Not provided')}
        Email: {purchase_info.get('email', 'Not provided')}
        Phone: {purchase_info.get('phone', 'Not provided')}
        Shipping Address: {purchase_info.get('address', 'Not provided')}
        
        The summary should:
        1. Thank the customer for their order
        2. Summarize all the order details including quantity and total price
        3. Provide next steps (payment link, estimated shipping time)
        4. Ask if they need any other assistance
        """
        
        result = self.chains["purchase_summarizer"].invoke({
            "summary_prompt": summary_prompt
        })
        
        # Return the content if it's an AIMessage or just the string if not
        return result.content if hasattr(result, 'content') else result
        
        
