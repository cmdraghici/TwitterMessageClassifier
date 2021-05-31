from flask import Flask

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
	return "Hello World"


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()