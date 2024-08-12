from google.cloud import aiplatform

CUSTOM_PREDICTOR_IMAGE_URI = "gcr.io/hallowed-chain-432118-r8/pytorch_predict_flask"

APP_NAME = "classifier_flask"
VERSION = 1
model_display_name = f"{APP_NAME}-v{VERSION}"
model_description = "PyTorch based text classifier with custom container"

MODEL_NAME = APP_NAME
health_route = "/ping"
predict_route = f"/predictions/{MODEL_NAME}"
serving_container_ports = [7080]

model = aiplatform.Model.upload(
    display_name=model_display_name,
    description=model_description,
    serving_container_image_uri=CUSTOM_PREDICTOR_IMAGE_URI,
    serving_container_predict_route=predict_route,
    serving_container_health_route=health_route,
    serving_container_ports=serving_container_ports,
)

model.wait()

print(model.display_name)
print(model.resource_name)