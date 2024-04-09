# views.py

from django.http import JsonResponse
from django.conf import settings
from rest_framework import status
import os
import numpy as np
import tensorflow as tf
from rest_framework.decorators import api_view
from tensorflow.keras.models import load_model
from PIL import Image
from .preprocessing import normalize_image, resize_image, crop_center

# Load the pre-trained .h5 model
model = load_model(os.path.join(settings.BASE_DIR, 'cnn_model.h5'))


# Define the classify_signature view using the imported preprocessing functions
@api_view(['POST'])
def classify_signature(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        # Load the image and preprocess it
        img = Image.open(image)
        img = img.convert('L')  # Convert to grayscale
        normalized_img = normalize_image(img)  # Normalize the image
        resized_img = resize_image(normalized_img, (100, 400))  # Resize the image

        # Convert NumPy array back to PIL Image
        resized_img_pil = Image.fromarray((resized_img * 255).astype(np.uint8))

        # Save the preprocessed image
        preprocessed_images_dir = os.path.join(settings.MEDIA_ROOT, 'preprocessed_images')
        if not os.path.exists(preprocessed_images_dir):
            os.makedirs(preprocessed_images_dir)
        preprocessed_image_path = os.path.join(preprocessed_images_dir, image.name)
        resized_img_pil.save(preprocessed_image_path)  # Save the preprocessed image

        img_array = tf.keras.preprocessing.image.img_to_array(resized_img_pil)
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
