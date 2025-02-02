from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
import os

app = Flask(__name__)

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    if request.method == 'POST':
        text = request.form.get('text')
        file = request.files.get('file')
        
        if text:
            summary = summarize_text(text)
        elif file:
            file_text = file.read().decode('utf-8')
            summary = summarize_text(file_text)
    
    return render_template('index.html', summary=summary)

def summarize_text(text, max_length=130, min_length=30):
    # Generate summary using the BART model
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == '__main__':
    app.run(debug=True)