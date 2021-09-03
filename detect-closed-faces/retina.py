import cv2
from matplotlib import pyplot as plt
from retinaface.pre_trained_models import get_model
import glob

model = get_model("resnet50_2020-07-20", max_size=2048)
model.eval()

def eval_image(path):
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict_jsons(image_rgb)
    print(results)

    for r in results:
        bbox = r['bbox']
        if not bbox: continue
        cv2.rectangle(image_rgb, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),thickness=10)

    plt.imshow(image_rgb)
    plt.show()

image_paths = glob.glob("edited/*.png")
for path in image_paths:
    eval_image(path)
