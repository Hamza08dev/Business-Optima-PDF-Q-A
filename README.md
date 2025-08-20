# Business Optima - AI-Powered PDF Analysis

A sophisticated web application that allows users to upload large PDF documents (up to 200 pages), ask questions about them, and generate comprehensive summaries with PDF download capability. Built with Flask, optimized for memory efficiency, and ready for Railway deployment.

## 🚀 Features

- **PDF Processing**: Handle documents up to 200 pages with intelligent chunking
- **AI-Powered Chat**: Ask questions about your documents using semantic search
- **Enhanced Summarization**: Generate 3-4 page comprehensive summaries using advanced MapReduce approach
- **PDF Download**: Download summaries as professionally formatted PDF files
- **Memory Optimized**: Designed for systems with 16GB RAM
- **Modern UI**: Responsive, drag-and-drop interface with real-time statistics
- **Real-time Processing**: No pre-processing required

## 🛠️ Technology Stack

- **Backend**: Flask (Python 3.13)
- **PDF Processing**: PyPDF2 + ReportLab (for PDF generation)
- **Embeddings**: All-MiniLM-L6-v2 (sentence-transformers)
- **Vector Database**: FAISS (CPU-optimized)
- **Frontend**: Modern HTML5, CSS3, JavaScript
- **Deployment**: Railway-ready

## 📋 Requirements

- Python 3.13
- 16GB RAM (minimum)
- Modern web browser

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd business_optima
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

### Railway Deployment

1. **Install Railway CLI** (optional)
   ```bash
   npm install -g @railway/cli
   ```

2. **Deploy to Railway**
   ```bash
   # Login to Railway
   railway login
   
   # Initialize project (if not already done)
   railway init
   
   # Deploy
   railway up
   ```

3. **Set environment variables** (in Railway dashboard)
   - `PORT`: Railway will set this automatically
   - `HOST`: Railway will set this automatically

## 📁 Project Structure

```
business_optima/
├── app.py                 # Main Flask application with enhanced features
├── requirements.txt       # Python dependencies including ReportLab
├── Procfile             # Railway deployment configuration
├── runtime.txt          # Python version specification
├── railway.json         # Railway configuration
├── templates/
│   └── index.html      # Enhanced frontend template
└── README.md           # This file
```

## 🔧 Configuration

### Memory Optimization Settings

The application is configured for optimal performance on 16GB RAM systems:

- **Chunk Size**: 1000 characters (optimal for 200-page documents)
- **Chunk Overlap**: 200 characters (maintains context)
- **Batch Size**: 10 (processes chunks in memory-efficient batches)
- **Summary Length**: 4000 characters (target for 3-4 page summary)

### Enhanced Summarization Features

- **Multi-factor Scoring**: Length, keyword density, importance indicators
- **Positional Boosting**: First and last sentences get priority
- **Comprehensive Coverage**: 3-4 pages instead of 2 pages
- **Smart Selection**: Top 3 sentences per chunk, 30 total sentences

### Embedding Model

- **Model**: All-MiniLM-L6-v2
- **Vector Dimension**: 384
- **Accuracy**: High semantic understanding
- **Memory Usage**: Optimized for CPU-only systems

## 📊 How It Works

### 1. PDF Processing
- Upload PDF document
- Extract text with memory-optimized batching
- Create overlapping chunks for context preservation
- Generate embeddings for semantic search

### 2. Chat Functionality
- User asks questions about the document
- System finds most relevant chunks using FAISS similarity search
- Returns contextual responses based on document content

### 3. Enhanced Summarization (MapReduce)
- **Map Phase**: Generate enhanced summaries for each chunk using multi-factor scoring
- **Reduce Phase**: Combine and refine chunk summaries with intelligent selection
- **Final Output**: 3-4 page comprehensive summary with PDF download

### 4. PDF Generation
- **Professional Formatting**: Clean, readable PDF layout
- **Statistics Display**: Summary length and target information
- **Easy Download**: One-click PDF generation and download

## 🚀 Railway Deployment Features

- **Health Check**: `/health` endpoint for monitoring
- **Auto-restart**: Configurable restart policies
- **Environment Variables**: Automatic PORT and HOST detection
- **Build Optimization**: NIXPACKS builder for efficient builds
- **Timeout Handling**: 5-minute health check timeout

## 📈 Performance Considerations

### Memory Management
- Garbage collection after each batch
- Efficient numpy operations
- FAISS CPU-optimized indexing
- Batch processing for large documents

### Processing Times
- **200-page PDF**: ~5-10 minutes (depending on content complexity)
- **Embedding Generation**: ~2-5 minutes
- **Enhanced Summary**: ~2-4 minutes (improved algorithm)
- **PDF Generation**: <1 second
- **Chat Responses**: <1 second

## 🔍 API Endpoints

- `GET /` - Main application interface
- `POST /upload` - PDF upload and processing
- `POST /chat` - Chat with document
- `POST /summarize` - Generate enhanced document summary
- `POST /download-summary` - Download summary as PDF
- `GET /status` - Processing status
- `GET /health` - Health check (Railway)

## 🆕 New Features

### Enhanced Summarization
- **Longer Summaries**: 3-4 pages instead of 2 pages
- **Better Content Selection**: Multi-factor scoring algorithm
- **Improved Coverage**: More comprehensive document analysis
- **Real-time Statistics**: Summary length and target information

### PDF Download
- **Professional Formatting**: Clean, readable PDF layout
- **Instant Generation**: No waiting time for PDF creation
- **Easy Access**: One-click download button
- **Copy to Clipboard**: Alternative text copying option

## 🛡️ Error Handling

- Comprehensive error logging
- User-friendly error messages
- Graceful degradation
- Memory cleanup on failures
- PDF generation error handling

## 🔧 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Ensure system has at least 16GB RAM
   - Close other applications during processing
   - Check available memory with `free -h` (Linux) or Task Manager (Windows)

2. **PDF Processing Issues**
   - Ensure PDF is not password-protected
   - Check PDF is not corrupted
   - Verify PDF contains extractable text

3. **Railway Deployment Issues**
   - Check build logs in Railway dashboard
   - Verify environment variables are set
   - Ensure health check endpoint is accessible

4. **PDF Download Issues**
   - Check if summary was generated successfully
   - Ensure browser supports blob downloads
   - Verify ReportLab installation

### Performance Tips

- Use SSD storage for faster I/O
- Close unnecessary browser tabs
- Process documents during low-usage periods
- Monitor system resources during processing

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review Railway deployment logs
- Open an issue in the repository

---

**Built with ❤️ for efficient PDF analysis, enhanced summarization, and Railway deployment**
