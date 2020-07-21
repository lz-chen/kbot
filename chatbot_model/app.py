from flask import Flask, render_template, request
from gpt_model.src.conditional_model import ConditionalModel

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    model_result = cond_model.generate(sentences=[userText])
    reply = '\n'.join(list(model_result.values())[0].split()[:-1]).strip()
    if reply.startswith(userText.strip()):
        reply.strip(userText)
    return reply


if __name__ == "__main__":
    #TODO next: use argparse for model path, change default path to the original one
    cond_model = ConditionalModel(model_name='plato',
                                  seed=155,
                                  length=100)
    app.run(host='0.0.0.0', port=8000)
