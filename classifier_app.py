import sqlite3, os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
from contextlib import closing
from werkzeug import secure_filename
import pdb

#config settings
ALLOWED_EXTENSIONS = set(['csv', 'png', 'txt', 'jpg'])
#ALLOWED_EXTENSIONS = set(['csv'])
UPLOAD_FOLDER = './data/'
DATABASE = "./flaskr.db"
DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'garrett'
PASSWORD = 'default'

#create our little application

app = Flask(__name__)

#Loads the above configuration settings based on the config params set above
app.config.from_object(__name__)
#app.config.from_envvar('FLASKR_SETTINGS', silent=True)

def connect_db():
	return sqlite3.connect(app.config['DATABASE'])

def init_db():
	with closing(connect_db()) as db:
	    with app.open_resource('schema.sql', mode='r') as f:
	    	db.cursor().executescript(f.read())
	    db.commit()

#pdb.set_trace()

@app.before_request
def before_request():
	g.db = connect_db()


@app.teardown_request
def teardown_request(exception):
	db = getattr(g, 'db', None)
	if db is not None:
		db.close()

@app.route('/')
def show_entries():
	cur = g.db.execute('select title, text from entries order by id desc')
	entries = [dict(title=row[0], text = row[1]) for row in cur.fetchall()]
	return render_template('show_entries.html', entries = entries)

@app.route('/add', methods = ['POST'])
def add_entry():
	if not session.get('logged_in'):
		abort(401)
	g.db.execute('insert into entries (title, text) values (? , ?)', 
		[request.form['title'], request.form['text']])
	g.db.commit()
	flash('New entry was successfully posted')
	return redirect(url_for('show_entries'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != app.config['USERNAME']:
            error = 'Invalid username'
        elif request.form['password'] != app.config['PASSWORD']:
            error = 'Invalid password'
        else:
            session['logged_in'] = True
            flash('You were logged in')
            return redirect(url_for('show_entries'))
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
	session.pop('logged_in', None)
	flash('You were logged out')
	return redirect(url_for('show_entries'))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            print 'file is allowed'
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    flash("You're file could not be uploaded")
    return redirect(url_for('show_entries'))

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	print send_from_directory(app.config['UPLOAD_FOLDER'], filename)
	flash('Data was successfully uploaded.  Please wait while the model is trained')
	return redirect(url_for('show_entries'))

if __name__ == '__main__':
	app.run()
