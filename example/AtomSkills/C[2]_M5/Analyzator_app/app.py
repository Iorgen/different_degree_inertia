from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
import os
import OSP
import numpy as np
from keras import backend as K
import glob
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pandas as pd
UPLOAD_FOLDER = 'control_results'
PREDICTION_FOLDER = 'prediction_results'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'dat', 'csv'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    l_frame = pd.DataFrame()
    filename = 0
    return render_template('main.html', l_frame=l_frame, filename=filename)


@app.route('/control_results')
def results():
    # TODO get all files and get results for them
    return render_template('results.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    l_datadrame = pd.DataFrame()
    filename = 0
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # TODO Send file to model
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            defects_start, defects_length = prediction("control_results/" + filename)
            d = {'start': defects_start, 'length': defects_length}
            print(defects_start, defects_length)
            # defects_length.append(1)
            l_datadrame = pd.DataFrame(data=d)
            l_datadrame.to_csv(app.config['PREDICTION_FOLDER'] + '/prediction' + filename)
            #TODO save in txt all result for that results, based on data and filename+ 'results' .txt

    return render_template('main.html', l_frame=l_datadrame, filename=filename)


def prediction(file_path):
    # Before prediction
    K.clear_session()
    sop = OSP.SOP(file_path)
    sop.read_control_results()
    sop.smooth_control_results()
    sop.fix_shift_issue()
    sop.split_by_defects()
    # ready long defects
    X_L = np.array(sop.longitudinal_defects_set)
    X_L = X_L.reshape(1, 1024, 12)
    X_T = np.array(sop.transverse_defects_set)
    X_T.reshape(1, 1024, 4)

    # ----------------------
    # THIS IS MODEL FOR longitudinal DEFECTS
    # load json and create model
    json_file = open('models/model_L.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_longitudinal = model_from_json(loaded_model_json)
    # load weights into new model
    model_longitudinal.load_weights("models/model_L.h5")
    print("Loaded TRANSVERS model from disk")
    longitudinal_prediction = model_longitudinal.predict(X_L, verbose=1)
    longitudinal_prediction = longitudinal_prediction > 0.38
    longitudinal_prediction = longitudinal_prediction.astype(int)
    longitudinal_prediction = longitudinal_prediction.reshape(longitudinal_prediction.shape[1])
    # longitudinal_prediction = longitudinal_prediction.tolist()
    start_l_list = []
    long_defects = []
    start_l_def = False
    defect_l_length = 0
    for idx, value in enumerate(longitudinal_prediction):
        print(value)
        if value == 1:
            if not start_l_def :
                start_l_def = True
                start_l_list.append(idx)
            defect_l_length+=1
        if value ==0:
            if start_l_def:
                long_defects.append(defect_l_length)
            defect_l_length =0
            start_l_def = False

    print(start_l_list)
    print(long_defects)

    # ----------------------
    # After prediction
    K.clear_session()
    return start_l_list, long_defects



if __name__ == '__main__':
    app.run(debug=True)
