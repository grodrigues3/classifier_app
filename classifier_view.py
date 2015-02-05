from flask import url_for, redirect, flash, render_template

def show_upload_page(error = None):
    return render_template('upload_training.html', error = error)

def show_login(error = None):
    return render_template('login.html', error=error)

def show_training_files(dataFiles = [], error = None):
    return render_template('select_page.html', data = dataFiles, error = error)


def show_post_processing(filename, baseRates, totalDocs, uniqueWords, modelList):
        return render_template("after_processing.html", filename = filename, baseRates = baseRates, totalDocs = totalDocs, uniqueWords = uniqueWords, models = modelList)

def show_post_training(trainingResults, numFeatures):
        print 'succeeded!'
        return render_template("after_training.html", results = trainingResults, D = numFeatures)

