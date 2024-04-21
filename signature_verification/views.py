# views.py
from django.http import JsonResponse
from django.conf import settings
from rest_framework import status
import os
import numpy as np
from .models import Signature
import tensorflow as tf
from rest_framework.decorators import api_view
from keras.models import load_model
from PIL import Image
from .preprocessing import preprocess_image
import requests

# Define the URL where your model file is hosted on the cloud storage service
MODEL_URL = "https://drive.usercontent.google.com/download?id=1qTP4BLyTKTu4ZUixjp-374fuEHRhZIBq&export=download&authuser=0&confirm=t&uuid=baea5119-2f97-4e00-9556-276859637661&at=APZUnTV-lM4SLT_L_1gjGEbnpsKn%3A1713535933249"

# Download the model file only once when the Django application starts up
model_path = os.path.join(settings.BASE_DIR, 'model.h5')
if not os.path.exists(model_path):
    print("Downloading model file...")
    response = requests.get(MODEL_URL)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model Download completed")
else:
    print("\nModel file already exists\n")

# Load the pre-trained .h5 model
model = load_model(model_path)

# Define the classify_signature view using the imported preprocessing functions
@api_view(['POST'])
def classify_signature(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        if preprocessed_image is not None:
            img_array = tf.keras.preprocessing.image.img_to_array(preprocessed_image)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make predictions
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            if predicted_class_index == 0:
                result = 'Genuine'
            else:
                result = 'Fake'

            # Calculate confidence
            confidence = np.max(predictions)

            # Convert confidence to native Python float
            confidence = float(confidence)
            signature = Signature(image=image, result=result, confidence=confidence)
            signature.save()
            # Create JSON response
            response_data = {
                'message': 'Signature classified successfully',
                'result': result,
                'confidence': confidence
            }
            print("Response Data: ", response_data)
            return JsonResponse(response_data, status=status.HTTP_200_OK)
        else:
            # Return an error response if image preprocessing fails
            return JsonResponse({'error': 'Image preprocessing failed'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        # Return an error response if no image is provided
        return JsonResponse({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
