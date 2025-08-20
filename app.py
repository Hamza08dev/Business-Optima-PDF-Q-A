import os
import json
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import tiktoken
from typing import List, Dict, Tuple
import gc
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import io

app = Flask(__name__)
CORS(app)

# Configuration
CHUNK_SIZE = 1000  # Optimal chunk size for 200-page documents
CHUNK_OVERLAP = 200  # Overlap to maintain context
MAX_SUMMARY_LENGTH = 4000  # Target for 3-4 page summary (increased from 2000)
BATCH_SIZE = 10  # Process chunks in batches for memory efficiency

# Global variables for document processing
pdf_text = ""
chunks = []
embeddings = None
faiss_index = None
chunk_embeddings = None

# Initialize the embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded successfully!")

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file with memory optimization."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # Process pages in batches to manage memory
        for i in range(0, len(pdf_reader.pages), BATCH_SIZE):
            batch_pages = pdf_reader.pages[i:i + BATCH_SIZE]
            for page in batch_pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Force garbage collection after each batch
            gc.collect()
        
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Create overlapping chunks from text with optimal sizing for 200-page documents."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        
        # Ensure we don't get stuck in an infinite loop
        if start >= len(text):
            break
    
    return chunks

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for text chunks with memory optimization."""
    try:
        # Process in smaller batches to manage memory
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_embeddings = embedding_model.encode(batch, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
            gc.collect()  # Force garbage collection
        
        return np.vstack(all_embeddings)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return np.array([])

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create FAISS index for similarity search."""
    try:
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Use IndexFlatIP for inner product (cosine similarity)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        
        return index
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None

def search_similar_chunks(query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
    """Search for most similar chunks to the query."""
    global faiss_index, chunk_embeddings, chunks
    
    if faiss_index is None or chunk_embeddings is None:
        return []
    
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(chunks):
                results.append((int(idx), float(score), chunks[idx]))
        
        return results
    except Exception as e:
        print(f"Error searching chunks: {e}")
        return []

def calculate_sentence_importance(sentence: str) -> float:
    """Calculate importance score for a sentence based on multiple factors."""
    if not sentence.strip():
        return 0.0
    
    # Factor 1: Length (longer sentences often contain more information)
    length_score = min(len(sentence.split()) / 20.0, 1.0)
    
    # Factor 2: Keyword density (sentences with more unique words)
    words = sentence.lower().split()
    unique_words = len(set(words))
    keyword_score = min(unique_words / len(words), 1.0) if words else 0
    
    # Factor 3: Presence of important indicators
    important_indicators = ['key', 'important', 'significant', 'major', 'primary', 'essential', 'critical', 'crucial']
    indicator_score = sum(1 for indicator in important_indicators if indicator in sentence.lower()) / len(important_indicators)
    
    # Factor 4: Sentence position (first and last sentences often important)
    # This will be handled in the chunk processing
    
    # Combine scores with weights
    final_score = (length_score * 0.3 + keyword_score * 0.3 + indicator_score * 0.4)
    return final_score

def generate_summary_mapreduce() -> str:
    """Generate comprehensive summary using enhanced MapReduce approach for 200-page documents."""
    global chunks
    
    if not chunks:
        return "No document loaded for summarization."
    
    try:
        print("Starting enhanced MapReduce summarization...")
        
        # Phase 1: Map - Generate enhanced summaries for each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            # Enhanced extractive summarization
            sentences = chunk.split('. ')
            if len(sentences) > 2:
                # Score sentences based on multiple factors
                sentence_scores = []
                for j, sent in enumerate(sentences):
                    if len(sent.split()) > 3:  # Filter out very short sentences
                        base_score = calculate_sentence_importance(sent)
                        
                        # Boost first and last sentences in chunk
                        if j == 0 or j == len(sentences) - 1:
                            base_score *= 1.2
                        
                        sentence_scores.append((sent, base_score))
                
                # Select top 3 sentences per chunk (increased from 2)
                sentence_scores.sort(key=lambda x: x[1], reverse=True)
                selected_sentences = sentence_scores[:3]
                chunk_summary = '. '.join([sent for sent, _ in selected_sentences]) + '.'
            else:
                chunk_summary = chunk
            
            chunk_summaries.append(chunk_summary)
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks")
        
        # Phase 2: Reduce - Combine chunk summaries into final summary
        combined_text = ' '.join(chunk_summaries)
        
        # Further enhance the summary
        sentences = combined_text.split('. ')
        if len(sentences) > 30:  # Target ~30 sentences for 3-4 pages
            # Enhanced sentence selection with better scoring
            sentence_scores = []
            for sent in sentences:
                if len(sent.split()) > 5:
                    score = calculate_sentence_importance(sent)
                    sentence_scores.append((sent, score))
            
            # Select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            final_sentences = sentence_scores[:30]
            final_summary = '. '.join([sent for sent, _ in final_sentences]) + '.'
        else:
            final_summary = combined_text
        
        # Ensure summary doesn't exceed target length
        if len(final_summary) > MAX_SUMMARY_LENGTH:
            final_summary = final_summary[:MAX_SUMMARY_LENGTH] + "..."
        
        print("Enhanced MapReduce summarization completed!")
        return final_summary
        
    except Exception as e:
        print(f"Error in enhanced MapReduce summarization: {e}")
        return f"Error generating summary: {str(e)}"

def create_summary_pdf(summary: str, filename: str = "document_summary.pdf") -> bytes:
    """Create a PDF file from the summary text."""
    try:
        # Create a buffer to store the PDF
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor='#2c3e50'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leading=16
        )
        
        # Add title
        title = Paragraph("Document Summary", title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Add summary text
        summary_paragraphs = summary.split('. ')
        for para in summary_paragraphs:
            if para.strip():
                # Clean up the paragraph
                clean_para = para.strip()
                if not clean_para.endswith('.'):
                    clean_para += '.'
                
                story.append(Paragraph(clean_para, body_style))
                story.append(Spacer(1, 8))
        
        # Build the PDF
        doc.build(story)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
        
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return None

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload and processing."""
    global pdf_text, chunks, embeddings, faiss_index, chunk_embeddings
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'File must be a PDF'}), 400
        
        print(f"Processing PDF: {file.filename}")
        
        # Clear previous data
        pdf_text = ""
        chunks = []
        embeddings = None
        faiss_index = None
        chunk_embeddings = None
        gc.collect()
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(file)
        
        if not pdf_text:
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        print(f"Extracted {len(pdf_text)} characters from PDF")
        
        # Create chunks
        print("Creating text chunks...")
        chunks = create_chunks(pdf_text)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        chunk_embeddings = generate_embeddings(chunks)
        
        if chunk_embeddings.size == 0:
            return jsonify({'error': 'Failed to generate embeddings'}), 400
        
        print(f"Generated embeddings with shape: {chunk_embeddings.shape}")
        
        # Create FAISS index
        print("Creating FAISS index...")
        faiss_index = create_faiss_index(chunk_embeddings)
        
        if faiss_index is None:
            return jsonify({'error': 'Failed to create search index'}), 400
        
        print("FAISS index created successfully!")
        
        # Clean up memory
        gc.collect()
        
        return jsonify({
            'message': 'PDF processed successfully',
            'chunks': len(chunks),
            'text_length': len(pdf_text)
        })
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries about the PDF."""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({'error': 'No message provided'}), 400
        
        if not chunks or faiss_index is None:
            return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
        
        print(f"Processing chat query: {query}")
        
        # Search for relevant chunks
        similar_chunks = search_similar_chunks(query, top_k=3)
        
        if not similar_chunks:
            return jsonify({'response': 'I couldn\'t find relevant information in the document for your question.'})
        
        # Generate response based on relevant chunks
        context = "\n\n".join([chunk for _, _, chunk in similar_chunks])
        
        # Simple response generation (you can enhance this with more sophisticated LLM integration)
        response = f"Based on the document, here's what I found:\n\n{context[:1000]}..."
        
        if len(context) > 1000:
            response += "\n\n(Response truncated for readability)"
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({'error': f'Error processing chat: {str(e)}'}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate a summary of the uploaded PDF."""
    try:
        if not chunks:
            return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
        
        print("Starting enhanced summarization...")
        start_time = time.time()
        
        summary = generate_summary_mapreduce()
        
        end_time = time.time()
        print(f"Enhanced summarization completed in {end_time - start_time:.2f} seconds")
        
        return jsonify({
            'summary': summary,
            'processing_time': f"{end_time - start_time:.2f} seconds",
            'summary_length': len(summary),
            'target_length': MAX_SUMMARY_LENGTH
        })
        
    except Exception as e:
        print(f"Error in summarization: {e}")
        return jsonify({'error': f'Error generating summary: {str(e)}'}), 500

@app.route('/download-summary', methods=['POST'])
def download_summary():
    """Download the summary as a PDF file."""
    try:
        if not chunks:
            return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
        
        # Generate summary
        summary = generate_summary_mapreduce()
        
        # Create PDF
        pdf_content = create_summary_pdf(summary)
        
        if pdf_content is None:
            return jsonify({'error': 'Failed to create PDF'}), 500
        
        # Create a temporary file-like object
        buffer = io.BytesIO(pdf_content)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name="document_summary.pdf",
            mimetype="application/pdf"
        )
        
    except Exception as e:
        print(f"Error in PDF download: {e}")
        return jsonify({'error': f'Error creating PDF: {str(e)}'}), 500

@app.route('/status')
def status():
    """Get the current processing status."""
    return jsonify({
        'document_loaded': len(chunks) > 0,
        'chunks_count': len(chunks),
        'embeddings_ready': embeddings is not None,
        'faiss_index_ready': faiss_index is not None,
        'text_length': len(pdf_text) if pdf_text else 0
    })

@app.route('/health')
def health():
    """Health check endpoint for Railway."""
    return jsonify({'status': 'healthy', 'message': 'Business Optima PDF Analysis API is running'})

if __name__ == '__main__':
    print("Starting Business Optima PDF Analysis Application...")
    print("Memory optimization enabled for 16GB RAM systems")
    print("Optimized for 200-page PDF documents")
    print("Enhanced summarization with PDF download capability")
    
    # Railway deployment configuration
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    app.run(host=host, port=port)
