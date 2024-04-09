from django.http import JsonResponse
from django.conf import settings
from rest_framework import status
import os
import numpy as np
import tensorflow as tf
from rest_framework.decorators import api_view
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained .h5 model
model = load_model(os.path.join(settings.BASE_DIR, 'cnn_model.h5'))


# Define the classify_signature view
@api_view(['POST'])
def classify_signature(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        # Load the image
        img = Image.open(image)
        img = img.convert('L')  # Convert to grayscale

        # Resize the image to match the model input shape
        img = img.resize((400, 100))  # Assuming the model expects input shape (100, 400)

        # Convert image to array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        prediction = predictions[0][0]  # Assuming a binary classification (fake/genuine)

        # Calculate confidence
        confidence = float(prediction)

        # Return results
        if prediction >= 0.5:
            result = 'Genuine'
        else:
            result = 'Fake'
            confidence = 1 - confidence

        # Save the received image
        received_images_dir = os.path.join(settings.MEDIA_ROOT, 'received_images')
        if not os.path.exists(received_images_dir):
            os.makedirs(received_images_dir)
        received_image_path = os.path.join(received_images_dir, image.name)
        img.save(received_image_path)

        # Create JSON response
        response_data = {
            'message': 'Signature classified successfully',
            'result': result,
            'confidence': confidence
        }
        print("Response Data: ", response_data)
        return JsonResponse(response_data, status=status.HTTP_200_OK)
    else:
        # Return an error response if no image is provided
        return JsonResponse({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
