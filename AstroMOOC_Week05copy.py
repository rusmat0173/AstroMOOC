"""
AstroMOOC Week05
Using stored np array (.npy file) given with course

"""
# import numpy
import numpy as np

# load .npy file and check details
data = np.load('/Users/RAhmed/Astro_MOOC/Week05/sdss_galaxy_colors.npy')
print('shape of array{}'.format(data.shape))

count = -5
print('first {} rows'.format(-count))

while count < 0:
    print(data[5 + count])
    count += 1

length = (len(data[0]))
print('length of row {}'.format(length))

# print  column labelled u
print(data['u'][:5])

# print column names, etc.
print('column names are {}'.format(data.dtype.names))
print('column 7 is {}'.format(data.dtype.names[6]))

#  find each row's type
print('row type is {}'.format(type(data[0])))
#  numpy void is a flexible data type, can include strings. Can do some interesting things
print(data['redshift'])


# ASSIGNMENT TASK: split array into m*4 and m*1 arrays
# this should be easy but is not due to the structure of this array, so have to be crafty
target = data['redshift']
print(target.size, type(target))
features = data[['u', 'g', 'r', 'i']]
print(features.size, type(features))

print(features[:5])

# this isn't quite what is wanted though. It's supposed to be an array with derived column variables, e.g. 'u - g'
# desired columns are u - g, g - r, r - i and i - z

# this toy example works nicely
foo0 = data['u']- data['g']
print(foo0[:5])


def get_features_targets(void_array):
    # get the names of variables
    names = []
    for name in void_array.dtype.names:
        names.append(name)
    # create correct size zero matrix, you want 4 variables long
    # unfortunately you have to hard code the columns you want, rather than have them as inputs
    features = np.zeros((len(void_array),4))
    features[:, 0] = void_array[names[0]] - void_array[names[1]]
    features[:, 1] = void_array[names[1]] - void_array[names[2]]
    features[:, 2] = void_array[names[2]] - void_array[names[3]]
    features[:, 3] = void_array[names[3]] - void_array[names[4]]

    targets = void_array[names[-2]]
    return features, targets

features, targets = get_features_targets(data)

# ASSIGNMENT: do decision tree regression
# imports
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()
dtr.fit(features, targets)
predictions = dtr.predict(features)

print(predictions[:4])

count = range(4)
for c in count:
    print(predictions[c], targets[c])

# ASSIGNMENT TASK: function to check MEDIAN differences between all predictions and target
print("shape of predictions: {}".format(predictions.shape))
print("shape of targets: {}".format(targets.shape))

def median_diff(predictions, targets):
    temp = np.abs(predictions - targets)
    return np.median(temp)

# print differences
print(median_diff(predictions, targets))

# clearly too accurate as uses all data to train. Need train/test split
# ASSIGNMENT: use median_differences function inside a validate_model function that splits the data into training and testing subsets
# uses a 'manual' split function as per course, rather than train_test_split
def validate_model(model, features, targets):
    # split the data into training and testing features and predictions
    split = features.shape[0] // 2
    train_features = features[:split]
    test_features = features[split:]
    train_targets = targets[:split]
    test_targets = targets[split:]
    # train the model (RA note: you have to assume model is initiated externally!!!)
    model.fit(train_features, train_targets)

    # get the predicted_redshifts
    predictions = model.predict(test_features)

    # use median_diff function to calculate the accuracy
    return median_diff(predictions, test_targets)

model = DecisionTreeRegressor()
z = validate_model(model, features, targets)
print('median difference is {:f}'.format(z))


# ASSIGNMENT: plot a redhsift plot in matplotlib
import numpy as np
import matplotlib.pyplot as plt

"""
We want x-axis as u-g, y-axis as r-i, and color mapping as the redshift
"""
x = features[:, 0]
y = features[:, 2]
colour = data['redshift']
print(colour)
# get a cmap from amtplotlib documenation
cmap = plt.get_cmap('OrRd')
plot = plt.scatter(x, y, s=2, lw=0, c=colour, cmap=cmap, alpha=0.5)
cb = plt.colorbar(plot)
cb.set_label('Redshift')
# plt.show()


# ASSIGNMENT: function to return  median difference for both the testing and training data sets
# for each of the tree depths in depths.

# N.B. Decision Tree Regressor is initiated outside this function
def accuracy_by_treedepth(features, targets, depths):
    # split the data into testing and training sets
    split = features.shape[0] // 2
    train_features = features[:split]
    test_features = features[split:]
    train_targets = targets[:split]
    test_targets = targets[split:]

    # initialise arrays or lists to store the accuracies for the below loop
    train_diff  = []
    test_diff = []

    # loop through depths
    for depth in depths:
        # initialize model with the maximum depth.
        dtr = DecisionTreeRegressor(max_depth=depth)

        # train the model using the training set
        dtr.fit(train_features, train_targets)

        # get the predictions for the training set and calculate their median_diff
        predictions = dtr.predict(train_features)
        train_diff.append(median_diff(predictions, train_targets))

        # get the predictions for the testing set and calculate their median_diff
        predictions = dtr.predict(test_features)
        test_diff.append(median_diff(predictions, test_targets))

    # return the accuracies for the training and testing sets
    return(train_diff, test_diff)


depths = [3, 5, 7]
z = accuracy_by_treedepth(features, targets, depths)
print(z)

# ASSIGNMENT: same again with k-fold validation
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt

# N.B. Decision Tree Regressor is initiated outside this function
def cross_validate_model(model, features, targets, k):
    kf = KFold(n_splits=k, shuffle=True)

    # initialise a list to collect median_diffs for each iteration of the loop below
    diff = []

    for train_indices, test_indices in kf.split(features):
        train_features = features[train_indices]
        train_targets = targets[train_indices]
        test_features = features[test_indices]
        test_targets = targets[test_indices]

        # fit the model for the current set
        model.fit(train_features, train_targets)

        # predict using the model
        prediction = model.predict(test_features)

        # calculate the median_diff from predicted values and append to results array
        diff.append(median_diff(prediction, test_targets))

    # return the list with your median difference values
    return diff

# initialize model with a maximum depth of 19, as per the course instructions
dtr = DecisionTreeRegressor(max_depth=19)

# call your cross validation function
diffs = cross_validate_model(dtr, features, targets, 10)
print(diffs)


# ASSIGNMENT: function to cross_validate_predictions. very similar to the previous except
# instead of returning the med_diff accuracy measurements return a predicted value for each of the galaxies.
# Your function should return a single variable. The returned variable should be a 1-D numpy array of length,
# where  is the number of galaxies in our data set.

# N.B. Decision Tree Regressor is initiated outside this function
def cross_validate_predictions(model, features, targets, k):
    kf = KFold(n_splits=k, shuffle=True)

    # declare an array for predicted redshifts from each iteration
    all_predictions = np.zeros_like(targets)

    for train_indices, test_indices in kf.split(features):
        # split the data into training and testing
        train_features = features[train_indices]
        train_targets = targets[train_indices]
        test_features = features[test_indices]
        test_targets = targets[test_indices]

        # fit the model for the current set
        model.fit(train_features, train_targets)

        # predict using the model
        predictions = model.predict(test_features)

        # put the predicted values in the all_predictions array defined above
        all_predictions[test_indices] = predictions

    # return the predictions
    return all_predictions

# now test
# initialize model
dtr = DecisionTreeRegressor(max_depth=19)

# call your cross validation function
predictions = cross_validate_predictions(dtr, features, targets, 10)

# calculate and print the rmsd as a sanity check
diffs = median_diff(predictions, targets)
print('Median difference: {:.3f}'.format(diffs))

# plot the results to see how well our model looks
plt.scatter(targets, predictions, s=0.4)
plt.xlim((0, targets.max()))
plt.ylim((0, predictions.max()))
plt.xlabel('Measured Redshift')
plt.ylabel('Predicted Redshift')
# plt.show()

# LAST ASSIGNMENT: function to split data containing both 2 types of galaxies
# into two arrays containing each one respectively. Function takes single data
# argument and returns two NumPy arrays

# imports
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

"""
Two things to note:
 i. the strings are stored as bytesm so have to be b'GALAXY', not 'GALAXY'
 ii. use 'masking': galaxies = data[data['spec_class'] == b'GALAXY']  - the inner data[...]
 gives indices that are used in the outer data[data[...] = ...]

"""
def split_galaxies_qsos(data):
    # sadly the style here is to hard code the things yiyu want, not put them in the argument of the function
    galaxies = data[data['spec_class'] == b'GALAXY']
    qsos = data[data['spec_class'] == b'QSO']

    return galaxies, qsos

# test and visualise output
# initialize model
dtr = DecisionTreeRegressor(max_depth=19)

# call your cross validation function
predictions = cross_validate_predictions(dtr, features, targets, 10)

# calculate and print the rmsd as a sanity check
diffs = median_diff(predictions, targets)
print('Median difference: {:.3f}'.format(diffs))

# plot the results to see how well our model looks
plt.scatter(targets, predictions, s=0.4)
plt.xlim((0, targets.max()))
plt.ylim((0, predictions.max()))
plt.xlabel('Measured Redshift')
plt.ylabel('Predicted Redshift')

plt.show()
