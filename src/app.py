from flask import Flask

app = Flask(__name__)

# Import routes
from routes import *

@app.route('/')
def home():
    return "Welcome to the LLM FCL Prototype!"

if __name__ == '__main__':
    app.run(debug=True)