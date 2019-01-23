from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import sys
from datetime import timedelta
from model import DarknetModel

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
dnet = DarknetModel()

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    f = request.files['file']
 
    if not (f and allowed_file(f.filename)):
        return jsonify({"error": 1001, "msg": "png、PNG、jpg、JPG、bmp"})

    
    basepath = os.path.dirname(__file__)  
    upload_path = os.path.join(basepath, 'static', 'images', secure_filename(f.filename))  
    f.save(upload_path)
    img = cv2.imread(os.path.join(basepath, 'static', 'images', secure_filename(f.filename)))
    ans = dnet.predict(upload_path)
    cv2.imwrite(os.path.join(basepath, 'static', 'results', secure_filename(f.filename)), ans)
    return render_template("upload_ok.html", width=img.shape[1], height=img.shape[0], fname=secure_filename(f.filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
