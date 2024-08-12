# pytorch_homework

## Training

Install libraries:
   
   ```
   pip install -r ./training/requirements.txt
   ```

Run
    ```
    python3 ./training/train.py
    ```
It saves the `state_dict()` of best modell to the `outputs/final_model.pth`


Copied the stat_dict() to a GCP bucket.
1. Created GCP account
    ```
    pip3 install gcloud
    gcloud auth login
    gcloud auth application-default login
    ```
2. Copied `application_default_credentials.json` to the folder 
3. Created a manually a bucket with name `continental_neural_network`
3. Copied outputs folder to the bucket
    ```
    pip 3install gsutil
    gsutil cp -r outputs gs://continental_neural_network
    ```

## App

1. Created `labels.json` to make labels readable
2. Run to be able to use the neural network in `neural_network.py` from other folder
    ```
    export PYTHONPATH=./
    ```
3. Saved 3 images from taining data to the test_images folder for test the app
4. Built a local app to give a prediction for an image using the `final_model.pth` from the GCP bucket
5. Run it
    ```
    python3 ./app/deploy.py 
    ```
    And test it from Postman with 
    ```
    curl -X POST -H "Content-Type: multipart/form-data" http://localhost:5000/predict -F "file=./test_images/boot.png"
    ```
6. Made Dockerfile to create an artifact and load it to GCP Artifact Registry
    ```
    docker build --tag=gcr.io/hallowed-chain-432118-r8/pytorch_predict_flask .
    ```

    Had an issue with `pip install torch` so the required .whl file was downloaded in advance from https://files.pythonhosted.org/packages/bf/55/b6c74df4695f94a9c3505021bc2bd662e271d028d055b3b2529f3442a3bd/torch-2.4.0-cp312-cp312-manylinux1_x86_64.whl \
    File was also too big for upload to Github

7. Tested and uploaded to the registry
    ```
    docker run -v "${cred_path}/application_default_credentials.json":/gcp/creds.json:ro --env GOOGLE_APPLICATION_CREDENTIALS=/gcp/creds.json -it gcr.io/hallowed-chain-432118-r8/pytorch_predict_flask

    docker push gcr.io/hallowed-chain-432118-r8/pytorch_predict_flask
    ```
