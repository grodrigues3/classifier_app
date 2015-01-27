#!/usr/bin/python

import sqlite3, os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, make_response, send_from_directory
from contextlib import closing
from werkzeug import secure_filename
import numpy as np
from multiprocessing import Pool, Process, Pipe, Queue
import pdb
from multiprocessing import Manager
from threading import Thread    


import classifier_model as CM
import classifier_view as CV
#config settings
ALLOWED_EXTENSIONS = set(['csv', 'tsv', 'png', 'txt', 'jpg'])
UPLOAD_TRAIN_FOLDER = './data/train/'
UPLOAD_TEST_FOLDER = './data/test/'
DATABASE = "./flaskr.db"
DEBUG = True #change this to True for development  
SECRET_KEY = 'development key'
USERNAME = 'metamind'
PASSWORD = 'metamind'

#create the application
app = Flask(__name__)

#Loads the above configuration settings based on the config params set above
app.config.from_object(__name__)

@app.route('/', methods = ["GET", "POST"])
def upload_training(error = None):
    if request.method == "GET":
        if 'logged_in' in session and session['logged_in']:
            return CV.show_upload_page(error) 
        else:
            return CV.show_login()
    elif request.method == 'POST':
        #upload the test file
        file = request.files['file']
        if file and allowed_file(file.filename):
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
    #either a get request or there was an error
    #either way, go back to the login page
    return CV.show_login(error) 

@app.route('/logout')
def logout():
	session.pop('logged_in', None)
	flash('You were logged out')
	return redirect(url_for('login'))

def allowed_file(filename):
    """
    Checks the given file's exetension to verify that it can be uploaded
    param: filename
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/select_page/', methods = ["GET"])
def select_training(error = None):
    flash('Data was successfully uploaded. Select your dataset and choose a model')
    dataFiles = os.listdir(app.config["UPLOAD_TRAIN_FOLDER"]) 
    return CV.show_training_files(dataFiles, error)

@app.route('/process_data/', methods = ["GET", "POST"])
def process_data():
    if request.method == "GET":
        #Get the params from the request
        filename = request.args['dataFile']
        modelList = ["SGD_Classifier", "Logistic Regression", "Garrett's Logistic Regression"] #should read this from a configuration file
        try:
            baseRates, totalDocs, uniqueWords = CM.getBaseRates(filename)
        except IndexError:
            error = "The file you upload is not correctly comma or tab separated. It should be in the format <label> <delimiter> <text>"
            return redirect(url_for('upload_training', error = error))
        return CV.show_post_processing(filename, baseRates, totalDocs, uniqueWords, modelList) 


def process_results(args):
    app, filename,  numFeats,  nIters,  uniqueWords = args
    with app.app_context():
        res = CM.fit_sgd(filename, numFeats, nIters, uniqueWords)
        return CV.show_post_training(res)

@app.route("/train_model/", methods = ["GET", "POST"])
def train_model():
    if request.method == "GET":
        filename = request.args['filename']
        model = request.args['model']
        representation = request.args['representation']
        #potential args: representation, numFeatures, regValues, nIters
        otherArgs = {arg:request.args[arg] for arg in request.args}
        ###Change this to just a fit method that takes in the modeltype too
        numFeatures, res = CM.fit(model, filename, **otherArgs)
        return CV.show_post_training(res, numFeatures)
    #return "Your data is being processed.  The page will automatically update after"

    elif request.method == "POST":
        file = request.files['test_file']
        dimensionality = int(request.form['dim'])
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_TEST_FOLDER'], filename))
            return redirect(url_for('test_model', test_fn=filename, D= dimensionality) )
        flash("You're file could not be uploaded")
        return redirect(url_for('process_data'))

        

@app.route('/inference/<test_fn>/<D>/', methods= ["GET","POST"])
def test_model(test_fn, D, score = None):
    fn = test_fn.rsplit('.', 1)[0].lower()
    fileExt = test_fn.rsplit('.', 1)[1].lower()
    delimiter = ","
    if fileExt == "tsv":
        delimiter = "\t"
    testValues = CM.predict(test_fn, D, True)
    respText = ""
    for val in testValues:
        respText += str(val) + "\n"
    response = make_response(respText)
    # This is the key: Set the right header for the response
    # to be downloaded, instead of just printed on the browser
    response.headers["Content-Disposition"] = "attachment; filename="+fn+"_results.csv"
    return response
    



if __name__ == '__main__':
	app.run()
    #app.run(host='0.0.0.0')
    #app.run(host='0.0.0.0' , port = 8000)
