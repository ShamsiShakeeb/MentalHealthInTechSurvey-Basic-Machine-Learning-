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


Gender = [];
Self_Eployee=[];
Family_History=[];
Remote_Work=[];
Work_Interface=[];
Treatment=[];


with open('surveyEncoded.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
           # print(f'{", ".join(row)}')
            line_count += 1
        else:
            Gender.append(row[0]);
            Self_Eployee.append(row[1]);
            Family_History.append(row[2]);
            Remote_Work.append(row[3]);
            Work_Interface.append(row[4]);
            Treatment.append(row[5]);
            line_count += 1
   # print(f'Processed {line_count} lines.')
    print("Initial Information Gain With All Feature Column with Target Column Treatment")
    print();
   



IgGenderTreatment  = info_gain.info_gain(Gender,Treatment)
IgSelfEmployeeTreatment = info_gain.info_gain(Self_Eployee,Treatment)
IgFamily_HistoryTreatment=info_gain.info_gain(Family_History,Treatment)
IgRemote_WorkTreatment=info_gain.info_gain(Remote_Work,Treatment)
IgWork_InterfaceTreatment=info_gain.info_gain(Work_Interface,Treatment)
print("Gender & Treatment InformationGain: ",IgGenderTreatment)
print("Self Employee & Treatment InformationGain: ",IgSelfEmployeeTreatment)
print("Family_History & Treatment InformationGain: ",IgFamily_HistoryTreatment)
print("Remote Work & Treatment InformationGain: ",IgRemote_WorkTreatment)
print("Work Interface & Treatment InformationGain: ",IgWork_InterfaceTreatment)
