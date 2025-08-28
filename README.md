# Constitution AI Agent - Backend
## 🚀 Demo
https://github.com/user-attachments/assets/c6fa7743-81d6-4abc-b8f5-af32d2e73d6a

## 🏗️ Architecture
<img width="100%" alt="System Architecture" src="https://github.com/user-attachments/assets/95ab9cb8-8703-4370-b8b4-03ec17f60f2a" />

## 🚀 Overview

The Constitution AI Agent Backend is a powerful Node.js/Express API that provides intelligent constitutional document processing and querying capabilities. It uses advanced AI techniques including vector embeddings, semantic search, and intelligent chunking to deliver precise legal information retrieval.

**🌐 Live API:** [https://constitution-ai-agent-backend.onrender.com](https://constitution-ai-agent-backend.onrender.com)

## ✨ Key Features

### 🧠 AI-Powered Processing
- **Multiple Chunking Strategies**: AI-based, logical regex-based, and LangChain recursive splitting
- **Vector Embeddings**: Cloudflare BGE-large-en-v1.5 model for semantic understanding
- **Query Expansion**: AI-enhanced query processing for better search results
- **Real-time Progress Tracking**: Server-sent events for processing status

### 🔍 Advanced Search
- **Semantic Vector Search**: Cosine similarity matching for relevant content
- **Contextual AI Responses**: LLaMA 3.1 8B model for generating legal analysis
- **Similarity Scoring**: Precise relevance matching with configurable thresholds
- **Source Attribution**: Full traceability to constitutional sections

### 🏗️ Technical Architecture
- **Database**: LibSQL (Turso) with vector storage
- **AI Provider**: Cloudflare Workers AI
- **Text Processing**: LangChain integration
- **API Design**: RESTful with streaming support

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Runtime** | Node.js 18+ | Server execution environment |
| **Framework** | Express.js | Web application framework |
| **Database** | LibSQL (Turso) | Vector embeddings & data storage |
| **AI Models** | Cloudflare Workers AI | Embeddings & text generation |
| **Text Splitting** | LangChain | Document processing |
| **CORS** | Express CORS | Cross-origin resource sharing |

## 📋 Prerequisites

- Node.js 18 or higher
- Turso database account
- Cloudflare Workers AI account
- npm or yarn package manager

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SavaliyaJay/Constitution-Ai-Agent-backend.git
cd Constitution-Ai-Agent-backend
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Environment Configuration
Create a `.env` file in the root directory:

```env
# Turso Database Configuration
LIBSQL_DB_URL=your_turso_database_url
LIBSQL_DB_AUTH=your_turso_auth_token

# Cloudflare Workers AI Configuration
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token

# Server Configuration
PORT=5000
NODE_ENV=production
```

### 4. Database Setup
The application automatically initializes database tables on startup:
- `constitution_chunks` - Stores processed document chunks and embeddings
- `user_queries` - Maintains query history and responses

### 5. Start the Server
```bash
# Development
npm run dev

# Production
npm start
```

## 🔌 API Endpoints

### Document Processing
```http
POST /process
```
Process and store constitution text with intelligent chunking.

**Request Body:**
```json
{
  "constitutionText": "full constitution text...",
  "strategy": "ai" // Options: "ai", "logical", "langchain"
}
```

**Response:** Server-sent events with real-time progress updates

### Query Constitution
```http
POST /query
```
Query constitutional provisions with AI-powered search.

**Request Body:**
```json
{
  "query": "What are the fundamental rights?"
}
```

**Response:**
```json
{
  "success": true,
  "query": "original query",
  "expandedQueryForSearch": "expanded query for better search",
  "response": "AI-generated legal analysis",
  "relevantSections": [
    {
      "text": "constitutional text...",
      "similarity": 0.8945,
      "article_section": "Article 14: Equality Before Law",
      "chunk_type": "article"
    }
  ]
}
```

### Query History
```http
GET /history?limit=20
```
Retrieve recent queries and responses.

### System Statistics
```http
GET /stats
```
Get system metrics and database statistics.

### Health Check
```http
GET /health
```
Server health and status information.

### Clear Data
```http
DELETE /clear
```
Remove all stored data (for testing purposes).

## 🧠 AI Processing Strategies

### 1. AI-Based Chunking (Default)
- Uses Cloudflare's LLaMA 3.1 8B model
- Intelligent section identification
- Context-aware splitting
- JSON-structured output

### 2. Logical Regex-Based Chunking
- Constitutional structure recognition
- Hierarchical processing (Parts → Chapters → Articles)
- Pattern-based section detection
- Metadata extraction

### 3. LangChain Recursive Splitting
- Fallback method for unstructured text
- Configurable chunk size and overlap
- Consistent processing approach

## 🔄 Processing Workflow

1. **Document Upload** → Text extraction from PDF
2. **Strategy Selection** → Choose chunking approach
3. **Intelligent Chunking** → Break document into logical sections
4. **Vector Embedding** → Generate embeddings for each chunk
5. **Database Storage** → Store chunks with metadata
6. **Query Processing** → AI-enhanced search and response

## 📊 Performance Features

- **Batch Processing**: Efficient embedding generation
- **Progress Tracking**: Real-time processing updates
- **Error Handling**: Comprehensive retry logic
- **Rate Limiting**: API protection and optimization
- **Memory Management**: Optimized for large documents

## 🚀 Deployment

### Render Deployment
The application is deployed on Render with automatic builds from GitHub.

**Environment Variables Required:**
- `LIBSQL_DB_URL`
- `LIBSQL_DB_AUTH`
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`

### Local Development
```bash
npm run dev
```
Server runs on `http://localhost:5000`

## 📁 Project Structure

```
constitution-ai-backend/
├── src/
│   ├── server.js           # Main application file
│   ├── routes/            # API route handlers
│   ├── services/          # Business logic
│   ├── utils/            # Helper functions
│   └── models/           # Data models
├── .env                  # Environment variables
├── package.json         # Dependencies
└── README.md           # This file
```

## 🔧 Configuration Options

### Chunking Parameters
```javascript
// Smart logical chunking
chunkSize: 500,
chunkOverlap: 50,
similarityThreshold: 0.3

// AI model parameters
temperature: 0.2,
max_tokens: 500
```

### Database Indexes
- `idx_chunk_type` - Chunk type filtering
- `idx_article_section` - Section-based queries

## 📈 Monitoring & Logs

The application provides comprehensive logging:
- Processing time measurements
- Progress tracking
- Error handling and reporting
- Performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the API documentation
- Review the error logs

---

**Built with ❤️ for Constitutional Law Research**
