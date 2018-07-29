"""
    AstroMOOC Week06
    Ensemble of Classifiers
    July 2018
    """
# imports
import numpy as np

# import file which is an np array and look at file
data = np.load('/Users/RAhmed/Astro_MOOC/Week06/galaxy_catalogue.npy')
print('shape of array{}'.format(data.shape))
print('column names are {}'.format(data.dtype.names))
print('first row is {}'.format(data[0]))
# neater way given in MOOC
for name, value in zip(data.dtype.names, data[0]):
    print('{:10} {:.6}'.format(name, value))


# ASSIGNMENT:  function to split the data into training and testing sets
def splitdata_train_test(data, fraction_training):
    # need to randomise shuffle of data
    np.random.seed(0)
    shuffled_data = np.random.shuffle(data)
    # indices of shuffled_data are ordered 0 to len(data) -1; use this to split
    split_index = int(fraction_training * len(data))
    training_set = data[:split_index]
    testing_set = data[split_index:]
    
    return training_set, testing_set

Train, Test = splitdata_train_test(data, 0.7)
print(Train.shape, Test.shape)


# ASSIGNMENT:  function to to calculate the concentration values for the u, r and z filters
def generate_features_targets(data):
    # complete the function by calculating the concentrations
    
    targets = data['class']

    features = np.empty(shape=(len(data), 13))
    features[:, 0] = data['u-g']
        features[:, 1] = data['g-r']
        features[:, 2] = data['r-i']
        features[:, 3] = data['i-z']
        features[:, 4] = data['ecc']
        features[:, 5] = data['m4_u']
        features[:, 6] = data['m4_g']
        features[:, 7] = data['m4_r']
        features[:, 8] = data['m4_i']
        features[:, 9] = data['m4_z']
    
        # fill the remaining 3 columns with concentrations in the u, r and z filters
        # concentration in u filter
        features[:, 10] = data['petroR50_u']/data['petroR90_u']
            # concentration in r filter
            features[:, 11] = data['petroR50_r']/data['petroR90_r']
            # concentration in z filter
            features[:, 12] = data['petroR50_z']/data['petroR90_z']
                
                return features, targets

# ASSIGNMENT:  complete the dtc_predict_actual function by following the Python comments. The purpose of the function is to perform a held out validation
# and return the predicted and actual classes for later comparison

# imports
import sklearn
from sklearn.tree import DecisionTreeClassifier

def dtc_predict_actual(data):
    """
        > Globally we're taking the large 780 dataset and splitting into a 70:30 train0:test1 split
        > We then split the train0 into (again) a train00 and target00, so the we effectively train the train0
        > Based on that we can test the result on the test1, which tests train11 against test11
        """
    # split the data into training and testing sets using a training fraction of 0.7
    train, test = splitdata_train_test(data, 0.7)
    # RA note: the above gives two large np arrays
    
    # generate the feature and targets for the training and test sets
    # i.e. train_features, train_targets, test_features, test_targets
    """
        RA note: we take each above array and split again into train and target, see comments at top
        the 'generate_features_targets' used creates derived variables that we want (features) and a single
        target column
        """
    train_features, train_targets = generate_features_targets(train)
    test_features, test_targets = generate_features_targets(test)
    
    # instantiate a decision tree classifier
    dtc = DecisionTreeClassifier()
    
    # train the classifier with the train_features and train_targets
    dtc.fit(train_features, train_targets)
    
    # get predictions for the test_features
    predictions = dtc.predict(test_features)
    
    # return the predictions and the test_targets
    return predictions, test_targets
    
    # get predictions for the test_features
    predictions = dtc.predict(test_features)
    
    # return the predictions and the test_targets
    return predictions, test_targets

z = dtc_predict_actual(data)
print(z[:30])

# NOTE from course on Accuracy: sklearn has methods to get the model score.
# Most models will have a score method which in the case of the decision tree
# classifier uses the above formula. The cross_val_score function in the model_
# selection module can be used to get k cross validated scores

# ASSIGNMENT: complete the calculate_accuracy function. The function should
# calculate the accuracy: the fraction of predictions that are correct (
# i.e. the model score)

# imports
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
# below looks like an exclusive package to the AstroMOOC, sadly as lovely confusion matrix!
# from support_functions import plot_confusion_matrix, generate_features_targets

def calculate_accuracy(predicted, actual):
    count = 0
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            count += 1
    return (count/len(predicted))


predicted, actual =  dtc_predict_actual(data)
print(calculate_accuracy(predicted, actual))

# testing all this with lovely confusion matrix design (all code below is given on course)
# split the data
features, targets = generate_features_targets(data)

# train the model to get predicted and actual classes
dtc = DecisionTreeClassifier()
predicted = cross_val_predict(dtc, features, targets, cv=10)

# calculate the model score using your function
model_score = calculate_accuracy(predicted, targets)
print("Our accuracy score:", model_score)


# ASSIGNMENT:  create the rf_predict_actual function.
# It returns the predicted and actual classes for our galaxies using a
# random forest 10-fold with cross validation.

# imports
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier


# IMPORTANT: you didn';t need a train/test split, as the cross_val_predict takes care of this!

def rf_predict_actual(data, n_estimators):
    # generate the features and targets
    features, targets = generate_features_targets(data)
    
    # instantiate a random forest classifier using n estimators
    rfc = RandomForestClassifier(n_estimators=n_estimators)
    
    # get predictions using 10-fold cross validation with cross_val_predict
    predicted = cross_val_predict(rfc, features, targets, cv=10)
    
    # return the predictions and their actual classes
    return predicted, targets

number_estimators = 50
predicted, targets = rf_predict_actual(data, number_estimators)
for i in range(10):
    print(predicted[i], targets[i])
