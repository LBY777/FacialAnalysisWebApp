import cv2
import pickle


haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
model_gender =  pickle.load(open('./model/gender/model_svc.pickle',mode='rb'))
model_emotion =  pickle.load(open('./model/emotion/model_vote.pickle',mode='rb'))

pca_model_with_mean_face_gender = pickle.load(open('./model/gender/pca_mean_face.pickle',mode='rb'))
model_pca_gender = pca_model_with_mean_face_gender['pca']
mean_face_gender = pca_model_with_mean_face_gender['mean_face']

pca_model_with_mean_face_emo = pickle.load(open('./model/emotion/pca_mean_face.pickle',mode='rb'))
model_pca_emo = pca_model_with_mean_face_emo['pca']
mean_face_emo = pca_model_with_mean_face_emo['mean_face']

def pipeline(img):
    '''
    Takes in a file path or image array.
    Return the image with emotion predicted on the faces that appear on the image.
    '''
    if isinstance(img, str):
        img = cv2.imread(img)
    label_dict = dict(zip(range(8), ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']))
    gender_dict = {1: 'male', 0: 'female'}
    gray =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    faces = haar.detectMultiScale(gray,1.5,3)
    predictions = []
    for x,y,w,h in faces:
        roi = gray[y:y+h,x:x+w]
        roi = roi / 255.0
        if len(roi) > 100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)

        eigen_image_gender = model_pca_gender.transform(roi_resize.reshape(1,10000) - mean_face_gender)
        eigen_image_emo = model_pca_emo.transform(roi_resize.reshape(1,10000) - mean_face_emo)
        eig_inv_image = model_pca_emo.inverse_transform(eigen_image_emo).reshape(100, 100)

        emo_preds = model_emotion.predict(eigen_image_emo)
        emo_prob_score = model_emotion.predict_proba(eigen_image_emo)
        emo_max_score = emo_prob_score.max()

        gender_preds = model_gender.predict(eigen_image_gender)
        gender_prob_score = model_gender.predict_proba(eigen_image_gender)
        gender_max_score = gender_prob_score.max()

        if gender_preds[0] == 1:
            color = (168, 23, 10)
        else:
            color = (255,50,255)

        emo_text = "%s : %d"%(label_dict[emo_preds[0]], emo_max_score*100)
        gender_text = "%s : %d"%(gender_dict[gender_preds[0]], gender_max_score*100)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(img,(x,y-80),(x+w,y),color,-1)
        cv2.putText(img,emo_text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
        cv2.putText(img,gender_text,(x,y-40),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
        output = {
                'roi':roi,
                'eig_img': eig_inv_image,
                'prediction_gender': gender_dict[gender_preds[0]],
                'gender_score': gender_max_score,
                'emo_score': emo_max_score,
                'prediction_emo': label_dict[emo_preds[0]]
                }
        predictions.append(output)

    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, predictions

def real_time(*args):
    '''
    Takes in a path to a video. 
    If not input is given, then use camera to capture video.
    Press 'q' to exit prediction.
    '''
    if len(args) == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args[0])

    while True:
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        pred_img = pipeline(frame)[0]
        
        cv2.imshow('prediction',pred_img)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
