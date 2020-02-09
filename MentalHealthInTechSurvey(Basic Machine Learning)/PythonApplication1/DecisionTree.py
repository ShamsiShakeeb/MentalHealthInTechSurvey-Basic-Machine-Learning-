from sklearn import tree
from info_gain import info_gain
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import csv




col_names = ['Gender','self_employed','family_history','remote_work','work_interfere','treatment']
# load dataset
pima = pd.read_csv("surveyEncoded.csv", header=None, names=col_names)

feature_cols = ['Gender','self_employed','family_history','remote_work','work_interfere']
X = pima[feature_cols] # Features
y = pima.treatment # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#prediction = clf.predict([['1','3','1','4','0']]);

#print(prediction);

#Predict the response for test dataset
y_pred = ((clf.predict(X_test)))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['Yes','No'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DecisionTreeOfTechPeopleCausesIssueOrNot.png')
Image(graph.create_png())