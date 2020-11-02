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
    reply = ' '.join(list(model_result.values())[0].split()[:-1]).strip()
    print(reply)
    print(userText.strip())
    if reply.strip().startswith(userText.strip()):
        reply = reply.strip().strip(userText)
    return reply


if __name__ == "__main__":
    #TODO next: use argparse for model path, change default path to the original one
    cond_model = ConditionalModel(model_name='plato',
                                  seed=155,
                                  length=50)
    # app.run(host='0.0.0.0', port=8000)
    app.run(host='127.0.0.1', port=8000)
