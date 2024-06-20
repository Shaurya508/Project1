from flask import Flask, request, jsonify, render_template
from memory import user_input, main as load_db

app = Flask(__name__)

# Initialize and load the database
load_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get("question")
    response, docs = user_input(user_question)
    return jsonify({"response": response["output_text"]})

if __name__ == "__main__":
    app.run(debug=True)
