import flask
import werkzeug
import inference
from deeplab import DeepLabModel

app = flask.Flask(__name__)

@app.route('/', methods=['GET','POST'])
def handle_request():
    imagefile= flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("Received image File name : "+ imagefile.filename)
    imagefile.save(filename)

    seg_map = inference.run_visualization(filename, model)
    return {"segmap":seg_map.tolist()}
    # return "Flask Server Good"

if __name__ == "__main__":
    model=DeepLabModel('')
    app.run(host="0.0.0.0", port=5000, debug=True)