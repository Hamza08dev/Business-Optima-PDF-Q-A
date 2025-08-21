import os
import json
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
import PyPDF2
import numpy as np
# from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding
import faiss
from typing import List, Dict, Tuple
import gc
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import io

app = Flask(__name__)
CORS(app)

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_SUMMARY_LENGTH = 4000
BATCH_SIZE = 8  # smaller to fit 512Mi

# Global variables for document processing
pdf_text = ""
chunks: List[str] = []
faiss_index = None
chunk_embeddings = None
_embedding_model: TextEmbedding | None = None


def get_embedding_model() -> TextEmbedding:
	global _embedding_model
	if _embedding_model is None:
		# Use a small, strong model; default is "BAAI/bge-small-en-v1.5"
		_embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=512)
	return _embedding_model


def extract_text_from_pdf(pdf_file) -> str:
	try:
		pdf_reader = PyPDF2.PdfReader(pdf_file)
		text = ""
		for i in range(0, len(pdf_reader.pages), BATCH_SIZE):
			batch_pages = pdf_reader.pages[i:i + BATCH_SIZE]
			for page in batch_pages:
				page_text = page.extract_text()
				if page_text:
					text += page_text + "\n"
			gc.collect()
		return text.strip()
	except Exception as e:
		print(f"Error extracting PDF text: {e}")
		return ""


def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
	if len(text) <= chunk_size:
		return [text]
	result: List[str] = []
	start = 0
	while start < len(text):
		end = start + chunk_size
		if end < len(text):
			for i in range(end, max(start + chunk_size - 100, start), -1):
				if text[i] in '.!?':
					end = i + 1
					break
		chunk = text[start:end].strip()
		if chunk:
			result.append(chunk)
		start = end - overlap
		if start >= len(text):
			break
	return result


def generate_embeddings(texts: List[str]) -> np.ndarray:
	try:
		model = get_embedding_model()
		all_vecs: List[np.ndarray] = []
		for i in range(0, len(texts), BATCH_SIZE):
			batch = texts[i:i + BATCH_SIZE]
			# fastembed returns generator; collect as list then to array
			emb_list = list(model.embed(batch))
			emb_arr = np.array(emb_list, dtype=np.float32)
			all_vecs.append(emb_arr)
			gc.collect()
		if not all_vecs:
			return np.zeros((0, 384), dtype=np.float32)
		return np.vstack(all_vecs)
	except Exception as e:
		print(f"Error generating embeddings: {e}")
		return np.zeros((0, 384), dtype=np.float32)


def create_faiss_index(embeddings: np.ndarray):
	try:
		if embeddings.size == 0:
			return None
		faiss.normalize_L2(embeddings)
		index = faiss.IndexFlatIP(embeddings.shape[1])
		index.add(embeddings.astype('float32'))
		return index
	except Exception as e:
		print(f"Error creating FAISS index: {e}")
		return None


def search_similar_chunks(query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
	global faiss_index, chunks
	if faiss_index is None:
		return []
	try:
		model = get_embedding_model()
		q_vec = np.array(list(model.embed([query])), dtype=np.float32)
		faiss.normalize_L2(q_vec)
		scores, indices = faiss_index.search(q_vec, top_k)
		results: List[Tuple[int, float, str]] = []
		for score, idx in zip(scores[0], indices[0]):
			if 0 <= idx < len(chunks):
				results.append((int(idx), float(score), chunks[idx]))
		return results
	except Exception as e:
		print(f"Error searching chunks: {e}")
		return []


def calculate_sentence_importance(sentence: str) -> float:
	if not sentence.strip():
		return 0.0
	length_score = min(len(sentence.split()) / 20.0, 1.0)
	words = sentence.lower().split()
	unique_words = len(set(words))
	keyword_score = min(unique_words / len(words), 1.0) if words else 0
	important_indicators = ['key', 'important', 'significant', 'major', 'primary', 'essential', 'critical', 'crucial']
	indicator_score = sum(1 for indicator in important_indicators if indicator in sentence.lower()) / len(important_indicators)
	return (length_score * 0.3 + keyword_score * 0.3 + indicator_score * 0.4)


def generate_summary_mapreduce() -> str:
	global chunks
	if not chunks:
		return "No document loaded for summarization."
	try:
		chunk_summaries: List[str] = []
		for i, chunk in enumerate(chunks):
			sentences = chunk.split('. ')
			if len(sentences) > 2:
				scored: List[Tuple[str, float]] = []
				for j, sent in enumerate(sentences):
					if len(sent.split()) > 3:
						score = calculate_sentence_importance(sent)
						if j == 0 or j == len(sentences) - 1:
							score *= 1.2
						scored.append((sent, score))
				scored.sort(key=lambda x: x[1], reverse=True)
				selected = scored[:3]
				chunk_summary = '. '.join([s for s, _ in selected]) + '.'
			else:
				chunk_summary = chunk
			chunk_summaries.append(chunk_summary)
			if (i + 1) % 20 == 0:
				gc.collect()
		combined_text = ' '.join(chunk_summaries)
		sentences = combined_text.split('. ')
		if len(sentences) > 30:
			scored: List[Tuple[str, float]] = []
			for sent in sentences:
				if len(sent.split()) > 5:
					scored.append((sent, calculate_sentence_importance(sent)))
			scored.sort(key=lambda x: x[1], reverse=True)
			final_sentences = [s for s, _ in scored[:30]]
			final_summary = '. '.join(final_sentences) + '.'
		else:
			final_summary = combined_text
		if len(final_summary) > MAX_SUMMARY_LENGTH:
			final_summary = final_summary[:MAX_SUMMARY_LENGTH] + "..."
		return final_summary
	except Exception as e:
		print(f"Error in enhanced MapReduce summarization: {e}")
		return f"Error generating summary: {str(e)}"


def create_summary_pdf(summary: str, filename: str = "document_summary.pdf") -> bytes:
	try:
		buffer = io.BytesIO()
		doc = SimpleDocTemplate(buffer, pagesize=letter)
		story = []
		styles = getSampleStyleSheet()
		title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30, alignment=TA_CENTER)
		body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontSize=11, spaceAfter=12, alignment=TA_JUSTIFY, leading=16)
		story.append(Paragraph("Document Summary", title_style))
		story.append(Spacer(1, 20))
		for para in summary.split('. '):
			p = para.strip()
			if not p:
				continue
			if not p.endswith('.'):
				p += '.'
			story.append(Paragraph(p, body_style))
			story.append(Spacer(1, 8))
		doc.build(story)
		pdf_content = buffer.getvalue()
		buffer.close()
		return pdf_content
	except Exception as e:
		print(f"Error creating PDF: {e}")
		return None


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
	global pdf_text, chunks, faiss_index, chunk_embeddings
	try:
		if 'file' not in request.files:
			return jsonify({'error': 'No file provided'}), 400
		file = request.files['file']
		if file.filename == '':
			return jsonify({'error': 'No file selected'}), 400
		if not file.filename.lower().endswith('.pdf'):
			return jsonify({'error': 'File must be a PDF'}), 400
		pdf_text = extract_text_from_pdf(file)
		if not pdf_text:
			return jsonify({'error': 'Could not extract text from PDF'}), 400
		chunks = create_chunks(pdf_text)
		chunk_embeddings = generate_embeddings(chunks)
		if chunk_embeddings.size == 0:
			return jsonify({'error': 'Failed to generate embeddings'}), 400
		faiss_index = create_faiss_index(chunk_embeddings)
		if faiss_index is None:
			return jsonify({'error': 'Failed to create search index'}), 400
		gc.collect()
		return jsonify({'message': 'PDF processed successfully','chunks': len(chunks),'text_length': len(pdf_text)})
	except Exception as e:
		print(f"Error processing PDF: {e}")
		return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500


@app.route('/chat', methods=['POST'])
def chat():
	try:
		data = request.get_json()
		query = data.get('message', '').strip()
		if not query:
			return jsonify({'error': 'No message provided'}), 400
		if not chunks or faiss_index is None:
			return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
		similar_chunks = search_similar_chunks(query, top_k=3)
		if not similar_chunks:
			return jsonify({'response': "I couldn't find relevant information in the document for your question."})
		context = "\n\n".join([chunk for _, _, chunk in similar_chunks])
		response = f"Based on the document, here's what I found:\n\n{context[:1000]}..."
		if len(context) > 1000:
			response += "\n\n(Response truncated for readability)"
		return jsonify({'response': response})
	except Exception as e:
		print(f"Error in chat: {e}")
		return jsonify({'error': f'Error processing chat: {str(e)}'}), 500


@app.route('/summarize', methods=['POST'])
def summarize():
	try:
		if not chunks:
			return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
		start_time = time.time()
		summary = generate_summary_mapreduce()
		end_time = time.time()
		return jsonify({'summary': summary,'processing_time': f"{end_time - start_time:.2f} seconds",'summary_length': len(summary),'target_length': MAX_SUMMARY_LENGTH})
	except Exception as e:
		print(f"Error in summarization: {e}")
		return jsonify({'error': f'Error generating summary: {str(e)}'}), 500


@app.route('/download-summary', methods=['POST'])
def download_summary():
	try:
		if not chunks:
			return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
		summary = generate_summary_mapreduce()
		pdf_content = create_summary_pdf(summary)
		if pdf_content is None:
			return jsonify({'error': 'Failed to create PDF'}), 500
		buffer = io.BytesIO(pdf_content)
		buffer.seek(0)
		return send_file(buffer, as_attachment=True, download_name="document_summary.pdf", mimetype="application/pdf")
	except Exception as e:
		print(f"Error in PDF download: {e}")
		return jsonify({'error': f'Error creating PDF: {str(e)}'}), 500


@app.route('/status')
def status():
	return jsonify({'document_loaded': len(chunks) > 0,'chunks_count': len(chunks),'faiss_index_ready': faiss_index is not None,'text_length': len(pdf_text) if pdf_text else 0})


@app.route('/health')
def health():
	return jsonify({'status': 'healthy', 'message': 'Business Optima PDF Analysis API is running'})


if __name__ == '__main__':
	print("Starting Business Optima PDF Analysis Application (FastEmbed)...")
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
