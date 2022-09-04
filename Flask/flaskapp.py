# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 12:24:14 2022
flask app for displaying analytics - reviewsearch
@author: starg
"""


from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from flask import (Flask,
                   url_for,
                   render_template,
                   request,
                   redirect,
                   make_response)
import os, re, io
import pandas as pd
from werkzeug.utils import secure_filename
import nlpscript
from nlpscript import NLP as nlp
import json





os.environ['FLASK_ENV'] = 'development'
app = Flask(__name__)

def analyzing():
    
    df = pd.read_csv('data.csv', index_col=False)
    for i in test_cols:
        assert i in df.columns,f'the column {i} was not found in df' 
    processed = nlp.preprocess(df)
    model = nlp.topiclize(processed)
    
    from io import BytesIO
    import base64
    print('generating report...')
    
    
    topics = nlp.format_topics(model)
    
    topics = {x:y for x, y in topics}  
    


@app.route('/', methods=['POST','GET'])
def main():
    
    
    if request.method == 'POST':
        
        

        payload = request.files['csvfile']
        dataset = pd.read_csv(payload, on_bad_lines='skip', index_col=False)
        
        dataset.to_csv('data.csv', index=False)
        
        return redirect(url_for('analysis'))
    
    if request.method == 'GET':
        return render_template('formpage.html')
    
@app.route('/analysis', methods=['GET'])
def analysis():
    
    df = pd.read_csv('data.csv', index_col=False)
    for i in test_cols:
        assert i in df.columns,f'the column {i} was not found in df' 
    processed = nlp.preprocess(df)
    processed.to_csv('data.csv', index=False)
    model = nlp.topiclize(processed)
    fig1 = nlp.freqdist(processed)
    fig2 = nlp.topic_freq(processed, model)
    fig3 = nlp.topic_sent(processed, model)
    fig4 = nlp.plt_polarity(processed)
    topics = nlp.format_topics(model)
    
    topics = {x:y for x, y in topics}  
   
    
    return render_template('analysis.html', topics=topics)    


@app.route('/images/freqdist',methods=['GET'])
def getfd():
    df = pd.read_csv('data.csv', index=False)
    fig = nlp.freqdist(df)
    
    image_binary = None
    response = make_response(image_binary)
    response.headers.set('Content-Type', 'image/png')
    response.headers.set(
        'Content-Disposition', 'attachment', filename='%s.png' % fig)
    return response

test_cols = ['review', 'date', 'company_id']

if __name__ == '__main__':
    app.run(debug=True)
    
