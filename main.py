from flask import Flask
from app import views
from app.views import *
from flask_socketio import SocketIO, emit
import eventlet
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*',logger=True)
app.add_url_rule(rule='/', endpoint='home', view_func=views.index)
# app.add_url_rule(rule='/webcam/', endpoint='webcam', view_func=views.webcam)
app.add_url_rule(rule='/app/', endpoint='app', view_func=views.app)
app.add_url_rule(rule='/app/image/', endpoint='image', 
                                view_func=views.image_pred, 
                                methods=['GET', 'POST'])
app.add_url_rule(rule='/app/video/', endpoint='video', 
                                view_func=views.video, 
                                methods=['GET', 'POST'])
app.add_url_rule(rule='/app/camera/', endpoint='camera', 
                                view_func=views.camera)
app.add_url_rule(rule='/app/camera/popup/', endpoint='popup', 
                                view_func=views.popup)
app.add_url_rule(rule='/code/code_gender/', endpoint='code_gender', view_func=views.code_gender)
app.add_url_rule(rule='/code/code_emotion/', endpoint='code_emotion', view_func=views.code_emotion)
app.add_url_rule(rule='/about/', endpoint='about', view_func=views.about)
  

@socketio.on('image')
def image(data_image):
    eventlet.sleep(0.2)
    frame = (readb64(data_image))

    imgencode = cv2.imencode('.jpeg', frame,[cv2.IMWRITE_JPEG_QUALITY,3])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

if __name__ == "__main__":
    # app.run(web, debug=True)
    socketio.run(app, debug=True)