import os
import cv2
from flask import render_template, request, Response
from app.emotion_gender_recognition import pipeline
import matplotlib.image as matimg
# from app import emotion_gender_recognition


upload_folder = 'static/upload'

def index():
    return render_template('index.html')

def code_gender():
    return render_template('code_gender.html')

def about():
    return render_template('about.html')

def code_emotion():
    return render_template('code_emotion.html')

def app():
    return render_template('app.html')

def image():
    if request.method == 'POST':
        f = request.files['image_name']
        name = f.filename
        path = os.path.join(upload_folder, name)
        f.save(path)
        pred_img, preds = pipeline(path)
        print('Predicted Successfully')
        pred_name = 'prediction_image.jpg'
        cv2.imwrite(f'./static/prediction/{pred_name}', pred_img)

        report = []
        for i, obj in enumerate(preds):
            gray = obj['roi']
            eigen = obj['eig_img']
            gender = obj['prediction_gender']
            emotion = obj['prediction_emo']
            emo_score = round(obj['emo_score']*100, 2)
            gender_score = round(obj['gender_score']*100, 2)

            gray_name = f'roi_{i}.jpg'
            eig_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/prediction/{gray_name}', gray, cmap='gray')
            matimg.imsave(f'./static/prediction/{eig_name}', eigen, cmap='gray')

            report.append([gray_name, eig_name, gender, gender_score, emotion, emo_score])
        return render_template('image.html', fileupload=True, report=report)
            

    return render_template('image.html')


def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cam = cv2.VideoCapture(0)
    while True:
        success, frame = cam.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', pipeline(frame)[0])
            frame = buffer.tobytes()
            # pred_img = cv2.imencode('.jpg', pipeline(frame)[0])[1].tobytes()
        
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def camera():
    return render_template('camera.html')


def video():
    if request.method == 'POST':
        f = request.files['video_name']
        name = f.filename
        path = os.path.join(upload_folder, name)
        f.save(path)


        cap = cv2.VideoCapture(path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('./static/prediction/video_pred.mp4',cv2.VideoWriter_fourcc(*'mpv4'), 10, (frame_width,frame_height))
        
        while(True):
            ret, frame = cap.read()
            
            if ret == True: 
                
                out.write(pipeline(frame)[0])
            
            else:
                break 
            
        cap.release()
        out.release()

        return render_template('video.html', fileupload=True)
            

    return render_template('video.html')