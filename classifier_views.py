import sqlite3, os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, make_response
from contextlib import closing
from werkzeug import secure_filename
from classifier_model import Classifier_Model 
import numpy as np
import pdb

#config settings
ALLOWED_EXTENSIONS = set(['csv', 'tsv', 'png', 'txt', 'jpg'])
#ALLOWED_EXTENSIONS = set(['csv'])
UPLOAD_TRAIN_FOLDER = './data/train/'
UPLOAD_TEST_FOLDER = './data/test/'
DATABASE = "./flaskr.db"
DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'garrett'
PASSWORD = 'default'

#create the application
app = Flask(__name__)

#Loads the above configuration settings based on the config params set above
app.config.from_object(__name__)

@app.route('/', methods = ["GET", "POST"])
def upload_training():
    if request.method == "GET":
        return render_template('upload_training.html')
    elif request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            print 'file is allowed'
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_TRAIN_FOLDER'], filename))
            return redirect(url_for('select_training'))
            #return redirect(url_for('send_file',
                                    #filename=filename, whereNext=1))
        flash("You're file could not be uploaded")
        return redirect(url_for('upload_training'))


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
            return redirect(url_for('upload_training'))
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
	session.pop('logged_in', None)
	flash('You were logged out')
	return redirect(url_for('login'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


from flask import send_from_directory

@app.route('/uploads/<filename>/<whereNext>/')
def send_file(filename, whereNext):
    whereNext = int(whereNext)
    send_from_directory(app.config['UPLOAD_TRAIN_FOLDER'], filename)
    if whereNext == 1:
        return redirect(url_for('select_training'))
    elif whereNext == 2: #if the data has been uploaded
        return render_template("after_training.html", results = [0,0])


@app.route('/select_page/', methods = ["GET"])
def select_training():
    flash('Data was successfully uploaded. Select your dataset and choose a model')
    dataFiles = os.listdir(app.config["UPLOAD_TRAIN_FOLDER"]) 
    modelList = ["Logistic Regression", "Naive Bayes", "Neural Network"]
    return render_template('train_page.html', data = dataFiles, models = modelList)



@app.route('/train_model/', methods = ["GET", "POST"])
def train_model():
    if request.method == "GET":
        train = .7
        lamb = 1
        print request.args
        filename = request.args['dataFile']
        modelType = request.args['model']
        fileExt = filename.rsplit('.', 1)[1].lower()
        delimiter = ","
        global cm
        cm = Classifier_Model()
        if fileExt == "tsv":
            delimiter = "\t"
        dataMat, labels = cm.getData(filename, delimiter)
        res = cm.perform_CV(dataMat, labels, regValues= [1,5])
        return render_template("after_training.html", results = res)
    elif request.method == "POST":
        return redirect(url_for('after_training'), code=307)
        

@app.route('/after_training/', methods = ["POST"])
def after_training():
    #the post request indicates a test dataset is being uploaded
    if request.method == "POST":
        file = request.files['file']
        if file and allowed_file(file.filename):
            print 'file is allowed'
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_TEST_FOLDER'], filename))
            return redirect(url_for('test_model', test_fn=filename ))
        flash("You're file could not be uploaded")
        print "youre file could not be uploaded" * 100
        return redirect(url_for('train_model'))


@app.route('/inference/<test_fn>', methods= ["GET","POST"])
def test_model(test_fn):
    fileExt = test_fn.rsplit('.', 1)[1].lower()
    delimiter = ","
    if fileExt == "tsv":
        delimiter = "\t"
    testData = cm.getData(test_fn, myDelimiter = delimiter, test=True)
    testValues = cm.predict(testData)
    testValues = cm.convertBack(testValues)
    respText = ""
    for val in testValues:
        respText += val + "\n"
    response = make_response(respText)
    # This is the key: Set the right header for the response
    # to be downloaded, instead of just printed on the browser
    response.headers["Content-Disposition"] = "attachment; filename=test_results.csv"
    return response
    #return render_template('after_testing.html', results = testValues)



if __name__ == '__main__':
	app.run()
