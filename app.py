# app.py

from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from QAchatbot import answer_question  # Importing the answer_question function

app = Flask(__name__)

# Store questions and answers
qa_history = []

# ... (rest of your existing code)

@app.route('/get_answer', methods=['POST'])
def get_answer():
    url = request.form['url']
    question = request.form['question']
    context = fetch_wikipedia_content(url)
    answer = answer_question(question, context)
    
    # Store the question and answer
    qa_history.append((question, answer))

    return render_template('index.html', qa_history=qa_history)

# ... (rest of your existing code)
