# Gundam Store Assistant

An AI-powered chatbot for managing Gundam store orders, built with LangChain, PGVector, and Streamlit.

## Features

- 🔍 Order lookup by email
- ❌ Order cancellation for pending orders
- 💬 Natural language processing
- 🔄 Real-time streaming responses
- 📝 FAQ system with vector search
- 💾 Conversation memory

## Prerequisites

- Python 3.12+
- Docker and Docker Compose
- PostgreSQL
- AWS Account with Bedrock access

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd fsds-llm
```

2. **Install uv for python manage package**
Install uv following the instruction from [uv](https://docs.astral.sh/uv/getting-started/installation/)

3. **Install dependencies**
Run `uv sync` it will automatically install all the dependencies and create a virtual environment
```bash
uv sync
```

4. **Set up PostgreSQL**
Move to the `src/vectordb` directory and run `docker-compose up -d` to start the PostgreSQL and PGVector database
Then move to utils directory, in faq folder, run `uv run enrich_faq.py` with faq.json path as argument (if you want to modify the faq.json, you can do it in the file or base on the schema in the file)
Then run `uv run add_document_to_pgvector.py` to add the enriched faq to the database
Then move to database folder, run `uv run orders_insert.py` to create sample orders in the database

5. **AWS Configure**
If you not have AWS ClI install, following this link [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
Then run `aws configure` and input your AWS credentials
Remember to set the region to `ap-northeast-1` (Tokyo)

6. **Run the application**
Run `streamlit run main.py` to start the application

7. **Access the application**
Open your browser and navigate to:
```
http://localhost:8501/
```

## Usage

1. **Check Orders**
   - Ask "Show me my orders"
   - Provide your email when asked
   - View order details

2. **Cancel Orders**
   - Request "Cancel my order #ORD-123"
   - Provide email if not already given
   - Only pending orders can be cancelled

3. **FAQ**
   - Ask general questions about faq
   - Get instant answers from the knowledge base

## Project Structure

```
fsds-llm/
├── src/
│   ├── core/
│   │   ├── bedrock_client.py         # Client for bedrock runtime
│   │   ├── tools.py          # Tool implementations
│   │   ├── embedding.py      # Embedding utilities
│   │   └── pgvector.py       # Vector database client
│   ├── utils/
│   │   ├── database/         # Database utilities
│   │   └── faq/             # FAQ management
│   └── vectordb/
│       ├── docker-compose.yml
│       └── init.sql
├── ui/
│   └── bot_ui.py            # Streamlit interface
└── requirements.txt
```

## Troubleshooting

1. **Database Connection Issues**
   - Ensure Docker containers are running: `docker ps`
   - Check PostgreSQL logs: `docker logs vectordb-postgres-1`
   - Verify database credentials in connection strings

2. **AWS Bedrock Access**
   - Confirm AWS credentials are set correctly
   - Ensure you have access to Claude model in Bedrock
   - Check region settings match your Bedrock endpoint

3. **Application Errors**
   - Check console logs for error messages
   - Verify all dependencies are installed
   - Ensure Python version compatibility

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
