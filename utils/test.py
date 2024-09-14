import pandas as pd
from sklearn import tree
import graphviz
import numpy as np
df = pd.read_csv('data.txt')
df.head(10)
df['色泽']=df['色泽'].map({'浅白':1,'青绿':2,'乌黑':3})
df['根蒂']=df['根蒂'].map({'稍蜷':1,'蜷缩':2,'硬挺':3})
df['敲声']=df['敲声'].map({'清脆':1,'浊响':2,'沉闷':3})
df['纹理']=df['纹理'].map({'清晰':1,'稍糊':2,'模糊':3})
df['脐部']=df['脐部'].map({'平坦':1,'稍凹':2,'凹陷':3})
df['触感'] = np.where(df['触感']=="硬滑",1,2)
df['好瓜'] = np.where(df['好瓜']=="是",1,0)
x_train=df[['色泽','根蒂','敲声','纹理','脐部','触感']]
y_train=df['好瓜']
print(df)
id3=tree.DecisionTreeClassifier(criterion='entropy')
id3=id3.fit(x_train,y_train)
print(id3)
id3=tree.DecisionTreeClassifier(criterion='entropy')
id3=id3.fit(x_train,y_train)
labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']
dot_data = tree.export_graphviz(id3,out_file=None,feature_names=labels,class_names=["好瓜","坏瓜"],filled=True,rounded=True)
graph = graphviz.Source(dot_data)
print(graph)
