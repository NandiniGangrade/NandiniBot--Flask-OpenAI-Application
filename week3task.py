from flask import Flask, render_template, request
from langchain.llms import OpenAI
import tiktoken
from langchain_community.llms import OpenAI


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def count_tokens(string: str) -> int:
    # Load the encoding for gpt-3.5-turbo (which uses cl100k_base)
    encoding_name = "p50k_base"
    encoding = tiktoken.get_encoding(encoding_name) 
    # Encode the input string and count the tokens
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_response(input_text, openai_api_key):
    try:
        llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
        response = llm(input_text)
        num_tokens = count_tokens(input_text)
        return f"Input contains {num_tokens} tokens.", response
    except Exception as e:
        return "An error occurred:", str(e)

@app.route('/process', methods=['POST'])
def process():
    openai_api_key = request.form['openai_api_key']
    input_text = request.form['input_text']
    if not openai_api_key.startswith('sk-'):
        return "Please enter your OpenAI API key!"
    else:
        info, response = generate_response(input_text, openai_api_key)
        return f"{info}\n{response}"

if __name__ == '__main__':
    app.run(debug=True)
