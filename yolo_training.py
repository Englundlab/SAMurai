from ultralytics import YOLO
import cv2, matplotlib.pyplot as plt
%matplotlib inline
import numpy as np


# 1. Load a pretrained model
model = YOLO("yolov8n.pt")

# 2. Start training
model.train(data=f"dataInteg.yaml", epochs=200, imgsz=768, device=0)

# 3. Predict with your new weights
imagetest = "NIS_L_Image_219323.png"
bbox_model = YOLO("runs/detect/train-1/weights/best.pt")
results = bbox_model.predict(source="path/images for testing training/" + imagetest,conf=0.15)
annotatedimage = results[0].plot(labels=False, conf=False, line_width=1)
annotatedimage = cv2.cvtColor(annotatedimage, cv2.COLOR_BGR2RGB)
cv2.imwrite("yolo_result_"+imagetest, annotatedimage)
plt.figure(figsize=(10, 8), dpi=200)
plt.imshow(annotatedimage)
plt.axis("off")
plt.show()

boxes = results[0].boxes.xyxy.cpu().numpy()
conf = results[0].boxes.conf.cpu().numpy()
np.save("test_boxes_conf/"+"boxes"+imagetest+".npy", boxes)
np.save("test_boxes_conf/"+"confs"+imagetest+".npy", conf)
