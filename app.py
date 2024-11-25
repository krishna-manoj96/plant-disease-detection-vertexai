import os
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

# Path to your service account JSON keys
storage_key_path = "gcp_storage.json"
vertex_key_path = "plant-disease-vertex-api.json"

# Authenticate with Google Cloud Storage using the first service account
storage_credentials = service_account.Credentials.from_service_account_file(storage_key_path)
storage_client = storage.Client(credentials=storage_credentials, project="codevipasana-442804")
bucket_name = 'bq-gemini-plant-disease-image'  # replace with your GCS bucket name
bucket = storage_client.bucket(bucket_name)

# Authenticate with Vertex AI using the second service account
vertex_credentials = service_account.Credentials.from_service_account_file(vertex_key_path)
PROJECT_ID = "codevipasana-442804"
vertexai.init(credentials=vertex_credentials, project=PROJECT_ID, location="us-central1")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            image_url = upload_image(file)
            prediction_result = predict_image(image_url)
            return render_template('result.html', prediction=prediction_result)
    return render_template('upload.html')

@app.route('/result')
def result():
    # Assuming prediction_result is passed from the upload_and_predict route
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

@app.errorhandler(Exception)
def handle_error(e):
    return render_template('error.html', error_message=str(e))


def upload_image(file):
    prefix = 'uploads/'
    destination_blob_name = os.path.join(prefix, file.filename)
    # Create a new blob and upload the file
    #blob = bucket.blob(file.filename)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file)
    gs_url = f"gs://{bucket_name}/{destination_blob_name}"
    return gs_url

def predict_image(image_url):
    
    model = GenerativeModel("gemini-1.5-flash-002")
    
    response = model.generate_content(
        [
            Part.from_uri(
                image_url,
                mime_type="image/jpeg",
            ),
            "Identify the plant in this image. Describe the disease affecting the plant, and give 2-3 remediation steps. Keep the language concise and accessible. The response should be key value pair, plant name, disease details, remediation",
        ]
    )
    print(response)
    text = response.to_dict()
    #text1 = text[0]['candidates']
    final_text = text['candidates'][0]['content']['parts'][0]['text'].split("\n")[1:]
    #print(final_text.split("\n"))
    final_text = [ i.replace('*','') for i in final_text if i!=""]
    #final_dict = {item.split(":")[0]: item.split(":")[1] for item in final_text}
    plant_name = final_text[0].split(":")[1].strip()
    disease_name = final_text[1].split(":")[1].strip()  # Extract disease name (assuming it's the first word after "Disease Details:")
    remediation = {
    '1': final_text[3].split(":")[1].strip(),
    '2': final_text[4].split(":")[1].strip(),
    '3': final_text[5].split(":")[1].strip(),
    }
    result_dict = {
    'plant_name': plant_name,
    'disease_name': disease_name,
    'remediation': remediation
    }
    print(final_text,"\n",result_dict)
    return result_dict


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
