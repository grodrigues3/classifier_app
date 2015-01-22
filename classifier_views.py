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
DEBUG = False #change this to True for development  
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
        if 'logged_in' in session and session['logged_in']:
            return render_template('upload_training.html')
        else:
            return redirect(url_for('login'))
    elif request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            print 'file is allowed'
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_TRAIN_FOLDER'], filename))
            return redirect(url_for('select_training'))
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

#@app.route('/uploads/<filename>/')
#def send_file(filename):
#    send_from_directory(app.config['UPLOAD_TRAIN_FOLDER'], filename)
#    return redirect(url_for('select_training'))
    


@app.route('/select_page/', methods = ["GET"])
def select_training():
    flash('Data was successfully uploaded. Select your dataset and choose a model')
    dataFiles = os.listdir(app.config["UPLOAD_TRAIN_FOLDER"]) 
    return render_template('select_page.html', data = dataFiles)


@app.route('/process_data/', methods = ["GET", "POST"])
def process_data():
    if request.method == "GET":
        filename = request.args['dataFile']
        fileExt = filename.rsplit('.', 1)[1].lower()
        delimiter = ","
        global cm
        cm = Classifier_Model()
        if fileExt == "tsv":
            delimiter = "\t"
        modelList = ["SGD_Classifier"]
        baseRates, totalDocs, uniqueWords = cm.getBaseRates(filename, delimiter) 
        return render_template("after_processing.html", filename = filename, baseRates = baseRates, totalDocs = totalDocs, uniqueWords = uniqueWords, models = modelList)
    

@app.route("/train_model/", methods = ["GET", "POST"])
def train_model():
    if request.method == "GET":
        
        filename = request.args['filename']
        model = request.args['model']
        representation = request.args['representation']
        numFeats = request.args['numFeatures']
        validate = request.args['val']
        nIters = request.args['nIters']

        try:
            numFeats = int(numFeats)
        except:
            numFeats = int(request.args['uniqueWords'])*2

        
        if validate == "Validation On":
            val = True
        else:
            val = False
        
        try:
            nIters = int(nIters)
        except:
            nIters = None

        print "Training the model..."
        res = []
        res += [cm.fit_sgd(filename, numFeats, nIters, val)]
        return render_template("after_training.html", results = res, D = numFeats)

    elif request.method == "POST":
        file = request.files['test_file']
        dimensionality = int(request.form['dim'])
        if file and allowed_file(file.filename):
            print 'file is allowed'
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_TEST_FOLDER'], filename))
            print "going to upload the test file"
            print filename, dimensionality
            return redirect(url_for('test_model', test_fn=filename, D= dimensionality) )
        flash("You're file could not be uploaded")
        return redirect(url_for('process_data'))

        

@app.route('/inference/<test_fn>/<D>/', methods= ["GET","POST"])
def test_model(test_fn, D):
    fn = test_fn.rsplit('.', 1)[0].lower()
    fileExt = test_fn.rsplit('.', 1)[1].lower()
    delimiter = ","
    if fileExt == "tsv":
        delimiter = "\t"
    #testData = cm.getData(test_fn, myDelimiter = delimiter, numFeats=D, test=True)
    testValues = cm.predict(test_fn, True)
    #testValues = cm.convertBack(testValues)
    respText = ""
    for val in testValues:
        respText += val + "\n"
    response = make_response(respText)
    # This is the key: Set the right header for the response
    # to be downloaded, instead of just printed on the browser
    response.headers["Content-Disposition"] = "attachment; filename="+fn+"_results.csv"
    return response
    #return render_template('after_testing.html', results = testValues)



if __name__ == '__main__':
	#app.run()
    app.run(host='0.0.0.0')
    #app.run(host='0.0.0.0' , port = 8000)
