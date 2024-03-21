from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request,jsonify
import os
import numpy as np
import openai
import cv2
import base64

app = Flask(__name__)
# custom_objects = {'CustomLayerName': CustomLayerClass}
model = load_model("garbage.h5",compile = False)
video = cv2.VideoCapture(0)
index = ['Non-Recyclable', 'Organic', 'Recyclable']


OPENAI_KEY = "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"
client = openai.OpenAI(api_key=OPENAI_KEY)

# def classify_frame(frame):
#     success, frame=video.read()
#     if frame is None or frame.size == 0:
#         print("Received empty frame")
#         return "Error: Empty frame"

#     print("Received frame with shape:", frame.shape)

#     img = cv2.resize(frame, (224, 224))
#     img = np.expand_dims(img, axis=0)

#     predictions = model.predict(img)
#     predicted_classes = np.argmax(predictions, axis=-1)
#     p = predicted_classes[0]

#     result = index[p]
#     return result
    # success, frame = video.read()
    # cv2.imwrite("image.jpg", frame)
    # img = image.load_img("image.jpg", target_size=(224, 224))

    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)

    # predictions = model.predict(x)
    # predicted_classes = np.argmax(predictions, axis=-1)
    # p = predicted_classes[0]

    # cv2.putText(frame, "Waste classification is: " + str(index[p]), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
    # new_width = frame.shape[1] + 200  # Increase the width by 200 pixels
    # new_height = frame.shape[0] + 100  # Increase the height by 100 pixels

    # cv2.imshow("showcasewindow", cv2.resize(frame, (new_width, new_height)))
    # if frame is None or frame.size == 0:
    #     return "Error: Empty frame"
    # img = cv2.resize(frame, (224, 224))
    # img = np.expand_dims(img, axis=0)
    # predictions = model.predict(img)
    # predicted_class = np.argmax(predictions, axis=-1)[0]
    # result = index[predicted_class]

    # return result

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/liveanalysis')
def liveanalysis():
    return render_template("liveanalysis.html")

@app.route('/classify', methods=['POST'])
def classify():
    # Decode image from base64 string
    img_str = request.json['image']
    img_data = base64.b64decode(img_str.split(',')[1])
    nparr = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
    
    # Processing image for model prediction
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_batch)
    index = ['Non-Recyclable', 'Organic', 'Recyclable']
    predicted_class = index[np.argmax(pred, axis=1)[0]]

    return jsonify({'result': predicted_class})


@app.route('/chatbot')
def chatbot():
    return render_template("chatbot.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method=='POST':
        f = request.files['images']
        basepath=os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        
        img = image.load_img(filepath,target_size =(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
        pred =np.argmax(model.predict(x),axis=1)
        index =['Non-Recyclable','Organic','Recyclable']
        text="The classified Garbage is : " +str(index[pred[0]])
        return text    

@app.route('/generate_response', methods=['POST'])
def generate_response():
    user_input = request.form['user_input']
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Based on the description given by the user about the waste, please generate disposal instructions for the described waste in bullet points and life cycle bullet points."},
            {"role": "user", "content": user_input},
        
        
        ],
        max_tokens =200,
        temperature=0.5
    )
    
    # Check the structure of the response to access the content correctly
    
    generated_response = response.choices[0].message.content
    # audio_input = generated_response
    # from pathlib import Path
    
    

    # speech_file_path = Path(_file_).parent / "speech.mp3"
    # response = client.audio.speech.create(
    # model="tts-1",
    # voice="alloy",
    # input=audio_input,
    # )

    # response.stream_to_file(speech_file_path)
    
    
    return jsonify({'response': generated_response}) 

# @app.route('/classify', methods=['POST'])
# def classify_video():
#     data = request.json
#     image_data = data['image']
#     print("Received image data:", image_data[:50])  # Print the first 50 characters of the image data
#     image_bytes = base64.b64decode(image_data)
#     print("Decoded image bytes:", image_bytes[:50])  # Print the first 50 bytes of the decoded image
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     if frame is None or frame.size == 0:
#         print("Received invalid frame data")
#         return jsonify({'result': 'Error: Invalid frame'})
    
#     result = classify_frame(frame)
    
#     return jsonify({'result': result})

if __name__=='__main__':
   app.run(host="0.0.0.0", port=8080,debug=True)