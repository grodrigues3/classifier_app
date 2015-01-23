import sqlite3, os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, make_response, send_from_directory
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
DEBUG = True #change this to True for development  
SECRET_KEY = 'development key'
USERNAME = 'metamind'
PASSWORD = 'metamind'

class Classifier_Views:
    def upload_training(self, error = None):
        if request.method == "GET":
            if 'logged_in' in session and session['logged_in']:
                return render_template('upload_training.html', error = error)
            else:
                return redirect(url_for('login'))
            print "HELLO WORLD"
        elif request.method == 'POST':
            #upload the test file
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_TRAIN_FOLDER'], filename))
                return redirect(url_for('select_training'))
            flash("You're file could not be uploaded")
            return redirect(url_for('upload_training'))


    def login(self ):
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

    def logout(self ):
	    session.pop('logged_in', None)
	    flash('You were logged out')
	    return redirect(url_for('login'))

    def allowed_file(self, filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




    def select_training(self, error = None):
        flash('Data was successfully uploaded. Select your dataset and choose a model')
        dataFiles = os.listdir(app.config["UPLOAD_TRAIN_FOLDER"]) 
        return render_template('select_page.html', data = dataFiles, error = error)


    def process_data(self):
        if request.method == "GET":
            filename = request.args['dataFile']
            fileExt = filename.rsplit('.', 1)[1].lower()
            delimiter = ","
            global cm
            cm = Classifier_Model()
            if fileExt == "tsv":
                delimiter = "\t"
            modelList = ["SGD_Classifier"]
            try:
                baseRates, totalDocs, uniqueWords = cm.getBaseRates(filename)
            except IndexError:
                error = "The file you upload is not correctly comma or tab separated. It should be in the format <label> <delimiter> <text>"
                return redirect(url_for('upload_training', error = error))
                
            return render_template("after_processing.html", filename = filename, baseRates = baseRates, totalDocs = totalDocs, uniqueWords = uniqueWords, models = modelList)
        

    def train_model(self):
        if request.method == "GET":
            filename = request.args['filename']
            model = request.args['model']
            representation = request.args['representation']
            numFeats = request.args['numFeatures'] #should be a comma separated list of features
            nIters = request.args['nIters']
            uniqueWords = request.args['uniqueWords']

            try:
                numFeats = int(numFeats)
            except:
                if uniqueWords != "":
                    numFeats = int(uniqueWords)*10
                else:
                    numFeats = 2**20

            res = cm.fit_sgd(fn = filename, dList = numFeats, nIters = nIters,uniqueWords =  uniqueWords)

            ###Change this to just a fit method that takes in the modeltype too
            return render_template("after_training.html", results = res, D = numFeats)

        elif request.method == "POST":
            file = request.files['test_file']
            dimensionality = int(request.form['dim'])
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_TEST_FOLDER'], filename))
                return redirect(url_for('test_model', test_fn=filename, D= dimensionality) )
            flash("You're file could not be uploaded")
            return redirect(url_for('process_data'))

            

    def test_model(self, test_fn, D, score = None):
        fn = test_fn.rsplit('.', 1)[0].lower()
        fileExt = test_fn.rsplit('.', 1)[1].lower()
        delimiter = ","
        if fileExt == "tsv":
            delimiter = "\t"
        testValues = cm.predict(test_fn, True)
        respText = ""
        for val in testValues:
            respText += val + "\n"
        response = make_response(respText)
        # This is the key: Set the right header for the response
        # to be downloaded, instead of just printed on the browser
        response.headers["Content-Disposition"] = "attachment; filename="+fn+"_results.csv"
        return response
    



if __name__ == '__main__':
	#app.run()
    app.run(host='0.0.0.0')
    #app.run(host='0.0.0.0' , port = 8000)
