from flask import Flask
from app import views
app = Flask(__name__)

app.add_url_rule(rule='/', endpoint='home', view_func=views.index)
app.add_url_rule(rule='/app/', endpoint='app', view_func=views.app)
app.add_url_rule(rule='/app/image/', endpoint='image', 
                                view_func=views.image, 
                                methods=['GET', 'POST'])
app.add_url_rule(rule='/app/video/', endpoint='video', 
                                view_func=views.video, 
                                methods=['GET', 'POST'])
app.add_url_rule(rule='/app/camera/', endpoint='camera', 
                                view_func=views.camera)
app.add_url_rule(rule='/app/camera/popup/', endpoint='popup', 
                                view_func=views.video_feed)
app.add_url_rule(rule='/code/code_gender/', endpoint='code_gender', view_func=views.code_gender)
app.add_url_rule(rule='/code/code_emotion/', endpoint='code_emotion', view_func=views.code_emotion)
app.add_url_rule(rule='/about/', endpoint='about', view_func=views.about)

if __name__ == "__main__":
    app.run(debug=True)
    