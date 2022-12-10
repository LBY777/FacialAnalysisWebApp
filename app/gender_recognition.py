import cv2
import pickle


haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
model_svm =  pickle.load(open('./model/model_svc.pickle',mode='rb'))
pca_model_with_mean_face = pickle.load(open('model/pca_mean_face.pickle',mode='rb'))
model_pca = pca_model_with_mean_face['pca']
mean_face = pca_model_with_mean_face['mean_face']

def pipeline(img):
    '''
    Takes in a file path or image array.
    Return the image with gender predicted on the faces that appear on the image.
    '''
    if isinstance(img, str):
        img = cv2.imread(img)
    gender_dict = {1: 'Male', 0: 'Female'}
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

        eigen_image = model_pca.transform(roi_resize.reshape(1,10000) - mean_face)
        eig_inv_image = model_pca.inverse_transform(eigen_image)
        preds = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        max_score = prob_score.max()

        if preds[0] == 1:
            color = (168, 23, 10)
        else:
            color = (255,50,255)

        text = "%s : %d"%(gender_dict[preds[0]], max_score*100)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
        # output = {
        #         'roi':roi,
        #         'eig_img': eig_inv_image,
        #         'prediction_gender': preds[0],
        #         'score':max_score}

    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

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
        
        pred_img = pipeline(frame)
        
        cv2.imshow('prediction',pred_img)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
