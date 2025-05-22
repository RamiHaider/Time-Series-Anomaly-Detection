# Time-Series-Anomaly-Detection

- model.pt, Is the weights of a Yolo11 model, pretrained on time series data up to 350 Epochs, using Google Colab for Training, achives >0.80 in MAP50-80
- main.py is the inferencing script, accepts an image as input, resizes to 640x640px, runs the inference with:
-         # Run inference
        results = model(img_resized, conf=0.2, iou=0.01, agnostic_nms=True), and returns a JSON of the bounding box location, along with the confidence, and is mapped back to the original image dimensions
- The dockerfile simply uses the latest ultralytics image, copies and installs the requirements, and then runs the main.py file.
- EC2 instance with 20gb was instantiated, all files copied over, Containerized using Docker, which is then hosted at: http://3.145.7.224:8000/docs. Then, I use a front-end, to 


Bare bones basic implementation of a dockerized computer vision model that makes inferences:

This is hosted in an EC2 instance, and you can access it via:
https://api.ramihaider.me/docs#/

Or the more UI friendly version on my website:

https://ramihaider.me/portfolio/automation-tasks.html
