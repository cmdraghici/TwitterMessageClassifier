from flask import Flask
from flask import render_template, request

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
	return render_template('master.html')

@app.route('/go')
def go():
	query = request.args.get('query', '')
	return render_template('go.html', query=query)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()