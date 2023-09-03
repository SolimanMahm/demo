from flask import Flask, request, render_template
from transformers import pipeline

def AnsweringQuestions(Question,Context) :
  model_name = "deepset/roberta-base-squad2"
  nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
  QA_input = {'question': Question,'context': Context}
  return nlp(QA_input)['answer']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predic', methods = ['POST'])
def predict():
    int_features  = [x for x in request.form.values()]
    return render_template('index.html',prediction_text=AnsweringQuestions(int_features[1],int_features[0]))

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
