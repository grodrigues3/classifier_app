"""
This is the controller for my Classification Web Application
    It should have:
        1) Model Object
        2) View Object
        3) Flask Application Object

"""
__author__ = "Garrett Rodrigues"
from classifier_view_object import Classifier_Views
from classifier_model import Classifier_Model
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, make_response, send_from_directory
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



View = Classifier_Views()
Model = Classifier_Model()
app = Flask(__name__)

def home():
    return .View.upload_training()

def run_app():
    .app.add_url_rule("/", 'whatthehell', .home)
    .app.run()

if __name__ == "__main__":
    C = Classifier_Controller()
    C.run_app()
