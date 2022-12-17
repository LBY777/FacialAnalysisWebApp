Facial Analysis
===============

**The web app is driven by data and machine learning model, aimed to predict users' biological gender and emotion in real-time based on facial characteristics.**





* * *

  
  

Download the app
====

Though the webapp can be [accessed remotely](https://facial-analysis-webapp.herokuapp.com/), it is still recommended to directly download the app from this repository for better experience on Video and Camera features.

Make sure you have installed [pip](https://pypi.org/project/pip/).

After cloning or downloading the repository, run the following command at the repository's root for all the dependencies.

```bash
pip3 install -r requirements.txt
```

Then, you can run

```bash
python3 main.py
```

You can then browse to `http://127.0.0.1:5000` in your browser to view the application.



* * *

  
  

Sample Output
====

Here is a sample output if the input is a **video**.
<br>
  
  
  ![](/static/images/home.gif)
  
  
Here is a sample output if the input is a **image**.

  <img src="/static/prediction/sample_pred.png" alt="drawing" width="600"/>

* * *

  
  

Data
====

The data that gender model trains on is from [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), from which 13809 faces of male and female form the training and test sets. The data that emotion model trains on is a modified version of [AffectNet](http://mohammadmahoor.com/affectnet/) dataset from [Kaggle](https://www.kaggle.com/datasets/tom99763/affectnethq?select=anger), from which 9000 faces belonging to 5 emotional states (happy, anger, surprise, sad, neutral) forms the training and test sets.

  
  
  
  

* * *

  
  

Model
====

The gender prediction uses Support Vector Machine model, which reaches 83.9% accuracy and 83.8% AUC on the test set, outperforming the other candidate models -- Random Forest and Logistic Regression. The emotion prediction uses a Voting Classifier that combines Support Vector Machine, Random Forest Classifier, Logistic Regression and K Neighber Classifier, and reaches 51.9% accuracy on the test set. Principle Component Analysis is used on both prediction tasks.

  
  
  

* * *

  
  

What I did
===


  

The process is similarly for both model. Generally, here are the steps:  
  
1\. Detect and crop out the faces in images using open-cv [haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades) face detector.  
2\. Resize the images containing cropped faces into 100 x 100 format. Flatten and normalize them.  
3\. Perform PCA, taking the 50 Principle Components, which cover over 80% of the explained variance.  
4\. Train the candidate models on the training set produced by PCA, with hyperparameter tuned by Grid Search.  
5\. Model evaluation and selection.  
6\. Form a pipeline with all the steps above that can be used on both image and video input.  

  
  

  
  

* * *
