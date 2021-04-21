"""
Created on Wed Dec 3

@author: Shaneer, Daniel, Akaylan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import time

def read_file(filename):
    df = pd.read_csv('databasebasketball/' + filename)
    return df

# normalize features
def min_max_scaling(data):
    data_norm = data.copy()
    for column in data_norm.columns:
        data_norm[column] = (data_norm[column] - data_norm[column].min()) / (
                data_norm[column].max() - data_norm[column].min())
    return data_norm

# helper function to link two datasets
def get_team_id_from_name(team_name):
    query = "name=='" + team_name + "'"
    return teams_data.query(query)['team'].iloc[0]

# helper function to link two datasets
def get_team_season_data_from_id(team_id, year):
    query = "team=='" + team_id + "' & year=='" + str(year) + "'"
    df = team_season_data.query(query).iloc[0]
    return df.drop(['team', 'year', 'leag'], axis=0)


# Read raw data from files
game_outcome = []
team_list = []
for year in range(1999, 2005, 1):
    filename = 'game_outcome_' + str(year) + '.txt'
    games = read_file(filename)
    # remove games between teams which did not play in all the selected seasons
    games = games[(games['Visitor/Neutral'] != 'Charlotte Hornets') & (games['Home/Neutral'] != 'Charlotte Hornets')
                  & (games['Visitor/Neutral'] != 'Memphis Grizzlies') & (games['Home/Neutral'] != 'Memphis Grizzlies')
                  & (games['Visitor/Neutral'] != 'New Orleans Hornets')
                  & (games['Home/Neutral'] != 'New Orleans Hornets')
                  & (games['Visitor/Neutral'] != 'Vancouver Grizzlies')
                  & (games['Home/Neutral'] != 'Vancouver Grizzlies')]
    if year == 1999:
        games = games.drop(['Date', 'Unnamed: 5', 'Unnamed: 6', 'Attend.', 'Notes'], axis=1)
    else:
        games = games.drop(['Date', 'Start (ET)', 'Unnamed: 6', 'Unnamed: 7', 'Attend.', 'Notes'], axis=1)
    game_outcome.append(games)

team_season_data = read_file('team_season.txt')
# Filter by year and only NBA, no ABA
team_season_data = team_season_data[(team_season_data['year'] >= 1999) & (team_season_data['leag'] == 'N')]

# teams_data is used to link game_outcome to team_season_data
teams_data = read_file('teams.txt')
teams_data = teams_data[teams_data['leag'] == 'N']  # Only NBA
teams_data['name'] = teams_data['location'] + " " + teams_data['name']
teams_data = teams_data.drop(['location', 'leag'], axis=1)


def get_X_y(game_outcome_data, year):
    y = []
    X = []
    empty = True
    for i in range(len(game_outcome_data)):  # foreach game
        # Get visitor and home team data
        game = game_outcome_data.iloc[i]
        visitor = get_team_season_data_from_id(get_team_id_from_name(game[0]), year)
        home = get_team_season_data_from_id(get_team_id_from_name(game[2]), year)
        # add to X and y
        y.append([game[1], game[3]])  # visitor_points, home_points
        x = pd.concat([visitor, home], axis=1)
        if empty:
            X = x
            empty = False
        else:
            X = pd.concat([X, x], axis=1)
    # Orientate correctly, and normalize the columns
    X = X.transpose()
    X = min_max_scaling(X)
    # Concatenate each (visitor, home) pair into one row for the neural network
    X_visitor = X.iloc[::2]
    X_home = X.iloc[1::2]
    X_visitor.reset_index(drop=True, inplace=True)
    X_home.reset_index(drop=True, inplace=True)
    X = pd.concat([X_visitor, X_home], axis=1, ignore_index=True).to_numpy(dtype='float')
    y = np.array(y)
    return X, y


def get_nn_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(66,)))
    model.add(Dense(2, activation='relu'))
    model.compile(optimizer='RMSProp', loss='mse', metrics=['accuracy'])
    return model


def evaluate_binary_model(model, X_test, y_test):
    correct = 0
    for i in range(len(y_test)):
        result = model.predict(X_test[i].reshape(1, -1))
        if result == y_test[i]:
            correct += 1
    return correct


def evaluate_difference_model(model, X_test, y_test):
    correct = 0
    for i in range(len(y_test)):
        result = model.predict(X_test[i].reshape(1, -1))
        if result >= 0:
            if y_test[i] >= 0:
                correct += 1
        elif y_test[i] < 0:
            correct += 1
    return correct


# How many years of training data? Accepts 1-5
training_years = 5

print("Reading training data for " + str(training_years) + " seasons...")
print("...2003")
X_train, y_train = get_X_y(game_outcome[4], 2003)
for i in range(training_years - 1):
    if i <= 3:
        print("..." + str(2002 - i))
        X, y = get_X_y(game_outcome[3 - i], 2002 - i)
        X_train = np.concatenate((X_train, X), axis=0)
        y_train = np.concatenate((y_train, y), axis=0)

print("Reading testing data for season 2004...")
X_test, y_test = get_X_y(game_outcome[5], 2004)

timer = []

print("Training neural network...")
X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train, test_size=0.15)
timer.append(time.time())  # 0 = start
nn_model = get_nn_model()
history = nn_model.fit(X_train_nn, y_train_nn, verbose=0, epochs=500,
                       validation_data=(X_val, y_val))
timer.append(time.time())  # 1 = end of nn training

print("Testing models...")
correct_list = []

# Neural network
points_off = 0
correct = 0
for i in range(len(y_test)):
    res = nn_model.predict(X_test[i].reshape(1, 66))
    win_pred = res[0, 0] > res[0, 1]
    win = y_test[i, 0] > y_test[i, 1]
    if win == win_pred:
        correct += 1
    points_off += abs(res[0, 0] - y_test[i, 0]) + abs(res[0, 1] - y_test[i, 1])

correct_list.append(correct)

timer.append(time.time())  # 2 = end of testing nn

# change y labels to difference between team points
# visitor points - home points
y_train = 1 * np.array(y_train[:, 0] - y_train[:, 1]).T
y_test = 1 * np.array(y_test[:, 0] - y_test[:, 1]).T

timer.append(time.time())  # 3 = start of training linear regression

linear_model = LinearRegression().fit(X_train, y_train)
timer.append(time.time())  # 4 = end of training linear regression
correct_list.append(evaluate_difference_model(linear_model, X_test, y_test))
timer.append(time.time())  # 5 = end of testing linear regression

nearest_neighbors_regression_model = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
timer.append(time.time())  # 6 = end of training k-nn
correct_list.append(evaluate_difference_model(nearest_neighbors_regression_model, X_test, y_test))
timer.append(time.time())  # 7 = end of testing k-nn

# change y labels to binary
# visitor win = 1
# visitor loss = 0
y_train = 1 * np.array(y_train > 0).T
y_test = 1 * np.array(y_test > 0).T

timer.append(time.time())  # 8 = start of training svm
logistic_model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
timer.append(time.time())  # 9 = end of training logistic regression
correct_list.append(evaluate_binary_model(logistic_model, X_test, y_test))
timer.append(time.time())  # 10 = end of testing logistic regression

svm_model = svm.SVC(C=2)
svm_model.fit(X_train, y_train)
timer.append(time.time())  # 11 = end of training svm
correct_list.append(evaluate_binary_model(svm_model, X_test, y_test))
timer.append(time.time())  # 12 = end of testing svm

forest_model = RandomForestClassifier(max_depth=5, random_state=0)
forest_model.fit(X_train, y_train)
timer.append(time.time())  # 13 = end of training random forests
correct_list.append(evaluate_binary_model(forest_model, X_test, y_test))
timer.append(time.time())  # 14 = end of testing random forests

def time_diff(time0, time1):
    return round((time1 - time0) * 1000)

# calculate running times
timer_results = []
timer_results.append(time_diff(timer[0], timer[1]))
timer_results.append(time_diff(timer[1], timer[2]))
timer_results.append(time_diff(timer[3], timer[4]))
timer_results.append(time_diff(timer[4], timer[5]))
timer_results.append(time_diff(timer[5], timer[6]))
timer_results.append(time_diff(timer[6], timer[7]))
timer_results.append(time_diff(timer[8], timer[9]))
timer_results.append(time_diff(timer[9], timer[10]))
timer_results.append(time_diff(timer[10], timer[11]))
timer_results.append(time_diff(timer[11], timer[12]))
timer_results.append(time_diff(timer[12], timer[13]))
timer_results.append(time_diff(timer[13], timer[14]))

total_tests = len(y_test)
print('{:>23}  {:>12}  {:>8}  {:>10}  {:>11}  {:>11}'.format('', 'Total Tests', 'Correct', 'Percentage', 'Training ms',
                                                             'Testing ms'))
for idx, name in enumerate(['Neural network', 'Linear regression', 'K nearest neighbours', 'Logistic regression',
                            'Support vector machine', 'Random forests']):
    correct = correct_list[idx]
    percent = str(round(correct / total_tests * 100, 2))
    print('{:>23}  {:>12}  {:>8}  {:>10}  {:>11}  {:>11}'.format(name, total_tests, correct, percent,
                                                                 timer_results[idx * 2], timer_results[idx * 2 + 1]))

# neural network training graphs
accuracy = history.history['accuracy']
loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(1, len(accuracy) + 1)

# Accuracy Graph
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Neural Network Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss graph
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Neural Network Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.axis([0, 500, 0, 200])
plt.legend()

plt.show()
