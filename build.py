import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, brier_score_loss, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from scipy.stats import uniform, randint
from matplotlib.pyplot import figure
import seaborn as sns
import itertools
import pickle
warnings.filterwarnings('ignore')

games = pd.read_csv('bdb-datasets/games.csv')
players = pd.read_csv('bdb-datasets/players.csv')
pff = pd.read_csv('bdb-datasets/pffScoutingData.csv')
plays = pd.read_csv('bdb-datasets/plays.csv')
week1 = pd.read_csv('bdb-datasets/week1.csv')
week2 = pd.read_csv('bdb-datasets/week2.csv')
week3 = pd.read_csv('bdb-datasets/week3.csv')
week4 = pd.read_csv('bdb-datasets/week4.csv')
week5 = pd.read_csv('bdb-datasets/week5.csv')
week6 = pd.read_csv('bdb-datasets/week6.csv')
week7 = pd.read_csv('bdb-datasets/week7.csv')
week8 = pd.read_csv('bdb-datasets/week8.csv')

df = pd.concat([week1, week2, week3, week4, week5, week6, week7, week8])

ball_snaps = df[(df['event'] == 'ball_snap')]

ball_loc = ball_snaps[(ball_snaps['team'] == 'football')][['gameId', 'playId', 'x', 'y']].rename(columns = {'x': 'ball_x', 'y': 'ball_y'})
ball_snaps = pd.merge(ball_snaps, ball_loc, on = ['gameId', 'playId'])

ball_snaps_right = ball_snaps[(ball_snaps['playDirection'] == 'right')]
ball_snaps_right['rel_x'] = ball_snaps_right['x'] - ball_snaps_right['ball_x']
ball_snaps_right['rel_y'] = (ball_snaps_right['y'] - ball_snaps_right['ball_y'])

ball_snaps_left = ball_snaps[(ball_snaps['playDirection'] == 'left')]
ball_snaps_left['rel_x'] = (120 - ball_snaps_left['x']) - (120 - ball_snaps_left['ball_x'])
ball_snaps_left['rel_y'] = (53.33 - ball_snaps_left['y']) - (53.33 - ball_snaps_left['ball_y'])

players_select = players[['nflId','officialPosition']]

ball_snaps_rel = pd.merge(pd.concat([ball_snaps_right, ball_snaps_left], axis=0), players_select, on = 'nflId')

oline = ball_snaps_rel[(ball_snaps_rel['officialPosition'].isin(['T', 'G', 'C']))]
def maxmin(x):
  mx = x.rel_y.max()
  mn = x.rel_y.min()
  return pd.Series({'oline_min': mn, 'oline_max': mx})
oline_min_max = oline.groupby(['gameId', 'playId']).apply(maxmin).reset_index()
oline_min_max['oline_width'] = oline_min_max['oline_max'] - oline_min_max['oline_min']

ball_snaps_rel = pd.merge(ball_snaps_rel, oline_min_max, on = ['gameId', 'playId'])

qb_x_y = ball_snaps_rel[(ball_snaps_rel['officialPosition'] == 'QB')].rename(columns = {'rel_x' : 'qb_rel_x', 'rel_y': 'qb_rel_y'})[['gameId', 'playId', 'x', 'y', 'ball_x', 'ball_y', 'team', 'qb_rel_x', 'qb_rel_y']]
num_qbs = ball_snaps_rel[(ball_snaps_rel['officialPosition'] == 'QB')].groupby(['gameId', 'playId']).count().reset_index().rename(columns = {'officialPosition' : 'num_qbs'})[['gameId', 'playId', 'num_qbs']]

qb_x_y = pd.merge(qb_x_y, num_qbs, on = ['gameId', 'playId'])

qb_x_y_one_qb = qb_x_y[(qb_x_y['num_qbs'] == 1)].drop('num_qbs', axis = 1)

qb_x_y_two_qbs = qb_x_y[(qb_x_y['num_qbs'] == 2)]

qb_x_y_two_qbs['y_diff'] = abs(qb_x_y_two_qbs['y'] - qb_x_y_two_qbs['ball_y'])

two_qbs_keep = qb_x_y_two_qbs.loc[qb_x_y_two_qbs.groupby(['gameId', 'playId']).y_diff.idxmin()].drop(['y_diff', 'num_qbs'], axis = 1)

qb_x_y = pd.concat([qb_x_y_one_qb, two_qbs_keep])
qb_x_y['qb_dist_from_ball'] = np.sqrt(np.square(qb_x_y['x'] - qb_x_y['ball_x'])  + np.square(qb_x_y['y'] - qb_x_y['ball_y']))

qb_x_y = qb_x_y.rename(columns = {'x': 'qb_x', 'y': 'qb_y', 'team': 'qb_team'})[['gameId', 'playId', 'qb_x', 'qb_y', 'qb_dist_from_ball', "qb_team", "qb_rel_x", "qb_rel_y"]]

all_dist_prep = pd.merge(ball_snaps_rel, qb_x_y, on = ['gameId', 'playId'])

defense_x_y = all_dist_prep[(all_dist_prep['team'] != all_dist_prep['qb_team'])]
defense_x_y['dist_from_qb'] = np.sqrt(np.square(defense_x_y['x'] - defense_x_y['qb_x'])  + np.square(defense_x_y['y'] - defense_x_y['qb_y']))

defense_keep = defense_x_y[['gameId', 'playId', 'nflId', 'rel_x', 'rel_y', 's', 'a', 
             'dir', 'o', 'ball_x', 'ball_y', 'officialPosition',
             'oline_min', 'oline_max', 'oline_width', 'qb_dist_from_ball',
             'qb_rel_x', 'qb_rel_y', 'dist_from_qb']]

def replace_pos(x):
    pos=x['officialPosition']
    if pos in ['DB', 'RB', 'G', 'LB']:
        return 'Other'
    else:
      return pos

defense_keep['officialPosition']=defense_keep.apply(replace_pos, axis=1)  

plays_select = plays[['gameId', 'playId', 'down', 'yardsToGo', 'absoluteYardlineNumber', 'offenseFormation', 'personnelO', 'defendersInBox', 'personnelD']]
plays_select = plays_select[(plays['down'] != 0)][['gameId', 'playId', 'down', 'yardsToGo', 'absoluteYardlineNumber', 'offenseFormation', 'personnelO', 'defendersInBox', 'personnelD']]
plays_select[['num_rb', 'num_te', 'num_wr']] = plays_select['personnelO'].str.split(', ', 2, expand=True)
plays_select['num_rb'] = (plays_select['num_rb'].str[:1]).fillna(0).astype(int)
plays_select['num_te'] = (plays_select['num_te'].str[:1]).fillna(0).astype(int)
plays_select['num_wr'] = (plays_select['num_wr'].str[:1]).fillna(0).astype(int)

plays_select[['num_dl', 'num_lb', 'num_db']] = plays_select['personnelD'].str.split(', ', 2, expand=True)
plays_select['num_dl'] = (plays_select['num_dl'].str[:1]).fillna(0).astype(int)
plays_select['num_lb'] = (plays_select['num_lb'].str[:1]).fillna(0).astype(int)
plays_select['num_db'] = (plays_select['num_db'].str[:1]).fillna(0).astype(int)

plays_select.drop(columns = ['personnelO', 'personnelD'], inplace = True)

sacks = pff[['gameId', 'playId', 'nflId', 'pff_sack']]
sacks['pff_sack'] = sacks['pff_sack'].fillna(0)

sacks_plays_joined = pd.merge(plays_select, sacks, on = ['gameId', 'playId']).dropna()
sacks_plays_joined['down'] = sacks_plays_joined['down'].astype('category')

sacks_df = pd.merge(sacks_plays_joined, defense_keep, on = ['gameId', 'playId', 'nflId'])
sacks_df = sacks_df[(sacks_df['officialPosition'] != 'Other')]

sacks_plays_joined_no_ids = sacks_df.drop(columns = ['gameId', 'playId', 'nflId', 'dir', 'o'])
sacks_plays_joined_no_ids = pd.get_dummies(sacks_plays_joined_no_ids, columns = ['down', 'officialPosition', 'offenseFormation'])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, test_index in sss.split(sacks_plays_joined_no_ids, sacks_plays_joined_no_ids['pff_sack']):
    strat_train_set = sacks_plays_joined_no_ids.iloc[train_index]
    strat_test_set = sacks_plays_joined_no_ids.iloc[test_index]

X_train = strat_train_set.drop(columns = ['pff_sack'])
Y_train = strat_train_set['pff_sack']
X_test = strat_test_set.drop(columns = ['pff_sack'])
Y_test = strat_test_set['pff_sack']

## LOGISITIC REGRESSION
# LR = LogisticRegression()
# LR.fit(X_train, Y_train)
# LR_pred = pd.DataFrame(LR.predict_proba(X_test), columns = ['no_sack', 'sack'])[['sack']]
# print('Brier Score: ', brier_score_loss(Y_test, LR_pred))

## RANDOM FOREST
# RF = RandomForestClassifier()
# RF.fit(X_train, Y_train)
# RF_pred = pd.DataFrame(RF.predict_proba(X_test), columns = ['no_sack', 'sack'])[['sack']]
# print('Brier Score: ', brier_score_loss(Y_test, RF_pred))

## XGBClassifier
XGB = XGBClassifier(objective="binary:logistic", random_state=42)
XGB.fit(X_train, Y_train)
XGB_pred = pd.DataFrame(XGB.predict_proba(X_test), columns = ['no_sack', 'sack'])[['sack']]
print('Brier Score: ', brier_score_loss(Y_test, XGB_pred))

## TUNED XGBClassifier
# XGB = XGBClassifier(objective="binary:logistic", random_state=42)
# XGB.fit(X_train, Y_train)
# params = {
#     "colsample_bytree": uniform(0.7, 0.3),
#     "gamma": uniform(0, 0.5),
#     "learning_rate": uniform(0.01, 0.3), # default 0.1 
#     "max_depth": randint(2, 8), # default 3
#     "n_estimators": randint(50, 200), # default 100
#     "subsample": uniform(0.6, 0.4)
# }
# search = RandomizedSearchCV(XGB, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=3, n_jobs=1, return_train_score=True)
# search.fit(X_train, Y_train)
# XGB_tuned = XGBClassifier(objective='binary:logistic',
#                           colsample_bytree = search.best_params_['colsample_bytree'],
#                           gamma = search.best_params_['gamma'],
#                           learning_rate = search.best_params_['learning_rate'],
#                           max_depth = search.best_params_['max_depth'],
#                           n_estimators = search.best_params_['n_estimators'],
#                           subsample = search.best_params_['subsample'],
#                          random_state = 42)
# XGB_tuned.fit(X_train, Y_train)
# XGB_tuned_pred = pd.DataFrame(XGB_tuned.predict_proba(X_test), columns = ['no_sack', 'sack'])[['sack']]
# print('Brier Score: ', brier_score_loss(Y_test, XGB_tuned_pred))

sorted_idx = XGB.feature_importances_.argsort()
feature_importance = pd.DataFrame(X_train.columns[sorted_idx], XGB.feature_importances_[sorted_idx])

# XGB.save_model('xgb_sack')
# feature_importance.to_csv('feature_importance.csv')
# sacks_preds.to_csv('sacks_preds.csv')

