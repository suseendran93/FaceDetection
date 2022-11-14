from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
# Load the model
model = load_model('keras_model.h5')
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR,"images")


current_id=0
label_ids={}
y_labels=[]
x_train=[]
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            print(label,path)
            # creating label ids
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            pil_image=Image.open(path).convert("L") #grayscale
            size=(550,550)
            final_image=pil_image.resize(size,Image.Resampling.LANCZOS)
            image_array=np.array(final_image,"uint8")
            print(image_array)
            size = (224, 224)
            # image = ImageOps.fit(pil_image, size, Image.ANTIALIAS)
            #detector
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array
            prediction = model.predict(data)
            for x,y,w,h in prediction:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


# run the inference

print(prediction)
