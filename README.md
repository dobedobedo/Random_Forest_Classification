# Random Forest Classification
Load an Excel file and process a random forest classifcation based on the user selected features and label
  
Before executing the script, the user need to modify the _inputfile_ variable to the desired Excel spreadsheet. The default reading mode will take the first row as the header and the first column as the index to a pandas dataframe.  
  
Once the data is successfully loaded, it will prompt the user to select _**one or more**_ features for classification, then _**one**_ feature as the class label.  This random forest nodes will grow parallel and randomly pick **80%** of the samples for training. The _warm start_ is enabled for the classifier to adjust the nodes during 300 iteration times. The weights for different classes will be adjusted in case the input labels are imbalanced. Once the classifier is trained completely to achieve a stable out-of-bag error, the classifier will be apply to all samples for classification.
  
The output will be two figures. The first one is the confusion matrix of the classification results with the accuracy annotated. The second figure is the change of out-of-bag accuracy durint the iteration.  
  
**Dependencies**: _numpy_, _pandas_, _scikit-learn_, and _matplotlib_.
