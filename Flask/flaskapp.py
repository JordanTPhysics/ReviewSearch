# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 12:24:14 2022
flask app for displaying analytics - reviewsearch
@author: starg
"""

from flask import Flask, url_for, render_template, request
import os, re, io
import pandas as pd
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from io import BufferedReader




os.environ['FLASK_ENV'] = 'development'
app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def main():
    file = None
    
    if request.method == 'POST':
        
        file = request.files['csvfile']
        file.name = file.filename
        file = BufferedReader(file)
        
        with open(file,'r+') as f:
           df = f.read_csv()
           
            
        
        
        
        print(dir(df))
        
        return df
    
    if request.method == 'GET':
        return render_template('formpage.html')
    
@app.route('/analysis', methods=['GET'])
def analysis():
    
    return render_template('analysis.html')    
    

if __name__ == '__main__':
    app.run(debug=True)
    
