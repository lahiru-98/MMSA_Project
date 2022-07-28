from MSA_FET import FeatureExtractionTool, get_default_config
import pickle as pkl
import pandas as pd
from flask import Flask , request
import json
from pathlib import Path
import tempfile 

app = Flask(__name__)

config_v = get_default_config('openface')

def convertPickle(file):
    '''
    Converting a pickle file to a csv File
    Accepting a disctionary of extracted features
    '''
    with open(file, "rb") as f:
        object = pkl.load(f)
    keys = []
    values = []
    items = object.items()
    for item in items:
        keys.append(item[0]), values.append(item[1])
    newdf = pd.DataFrame(values[0])
    df2 = newdf.iloc[:, 142: 159].copy()
    df2.columns = [" AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r", " AU07_r", " AU09_r", " AU10_r", " AU12_r",
                   " AU14_r", " AU15_r", " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r"]
    #df2.to_csv("AUFeatures.csv")
    return df2


def extractFeatures(file_path):
    
    fet = FeatureExtractionTool("openface")
    try:
        feature = fet.run_single(file_path)
        print(feature)
    except FileNotFoundError as e:
        print(e.strerror)

    fet = FeatureExtractionTool(config_v, tmp_dir="/tmp")
    
    feature_dict = fet.run_single(in_file=file_path)
    # call the function by passing the generated pkl file
    #df  = convertPickle("feature.pkl")
    return feature_dict


@app.route('/')
def index():
    return "Your App is Working!!!"

@app.route("/getdata")
def get_data():
    res = extractFeatures()
    #json_object = json.dumps(res, indent = 4)
    return str(res)



@app.route('/video', methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        with tempfile.TemporaryDirectory() as td:
            temp_filename = Path(td) / 'uploaded_video'
            uploaded_file.save(temp_filename)

            res = extractFeatures(temp_filename)
            return str(res)
    else:
        return "Something Went Wrong"



if __name__ == "__main__":
    app.run()
    
    
    
    