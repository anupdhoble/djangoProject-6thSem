from django.db import models


class Signature(models.Model):
    image = models.ImageField(upload_to='images/')
    result = models.CharField(max_length=50)
    confidence = models.FloatField()

    def __str__(self):
        return f"Signature #{self.id}"
