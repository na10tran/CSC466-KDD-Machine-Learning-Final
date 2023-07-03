import numpy as np
import pandas as pd

NUM_BINS = 6

allStats = ['rating','impact','kdr','dmr','kpr','apr','dpr','spr','opk_ratio','opk_rating','wins_perc_after_fk',
            'fk_perc_in_wins','multikill_perc','rating_at_least_one_perc','is_sniper','clutch_win_perc']

def get_players():
    allPlayers = []
    for t in range(2):
        for p in range(5):
            allPlayers.append('t' + str(t + 1) + '_player' + str(p + 1))
    return allPlayers

def get_players_per_stat():
    allPlayers = get_players()
    allPlayersPerStat = {}
    for stat in allStats:
        temp = []
        for player in allPlayers:
            playerStat = player + '_' + stat
            temp.append(playerStat)
        allPlayersPerStat[stat] = temp
    return allPlayersPerStat

def get_teams():
    t1 = []
    t2 = []
    for p in range(5):
        t1.append('t1' + '_player' + str(p + 1))
        t2.append('t2' + '_player' + str(p + 1))
    return (t1, t2)

def get_player_stats(team, player):
    playerStats = []
    for stat in allStats:
        playerStats.append('t' + str(team) + '_player' + str(player) + '_' + stat)
    return playerStats
    
def bin_cols(X, cols, bins = NUM_BINS):
    X = X.copy()
    for col in cols:
        binned_col, temp = pd.cut(X[col], bins, retbins = True)
        X[col] = binned_col
    return X

def normalize_cs(cs):
    cs_norm = cs.copy()
    allPlayersPerStat = get_players_per_stat()
    teams = get_teams()
    teamStats = ['points', 'world_rank', 'h2h_win_perc']
    for stat in teamStats:
        cs_norm[stat] = cs_norm['t1_' + stat] - cs_norm['t2_' + stat]
        cs_norm = cs_norm.drop('t1_' + stat, axis = 1)
        cs_norm = cs_norm.drop('t2_' + stat, axis = 1)

    for stat in allPlayersPerStat.keys():
        t1 = [p + "_" + stat for p in teams[0]]
        t2 = [p + "_" + stat for p in teams[1]]
        cs_norm[stat] = (np.sum(cs_norm[t1], axis = 1) - np.sum(cs_norm[t2], axis = 1))
        cs_norm = cs_norm.drop(t1, axis = 1)
        cs_norm = cs_norm.drop(t2, axis = 1)

    cs_norm = cs_norm.drop(['match_date', 'team_1', 'team_2', 'points'], axis = 1)
    return cs_norm
    

def compute_priors(y):
    priors = {}
    y = y.sort_values()
    for x in y.unique():
        y_counts = y.value_counts()
        priors[y.name + "=" + str(x)] = y_counts.loc[x]/y_counts.sum()
    return priors

def specific_class_conditional(x,xv,y,yv):
    import pandas as pd
    prob = ((x == xv) & (y == yv)).sum()/(y == yv).sum()
    return prob

def class_conditional(X,y):
    probs = {}
    for column in X.iteritems():
        for att in column[1].unique():
            for cond in y.unique():
                name = "%s=%s|%s=%s"%(column[0], str(att), y.name, str(cond))
                probs[name] = specific_class_conditional(column[1], att, y, cond)
                
    return probs

def posteriors(probs,priors,x):
    post_probs = {}
    norm = 0
    for d in priors:
        ypart = d
        prob = 1
        xparts = []
        for c in x.index:
            xpart = "%s=%s"%(c,x.loc[c])
            xparts.append(xpart)
            key = "%s|%s"%(xpart,ypart)
            if key not in probs:
                prob = prob * 0
            else:
                prob = prob * probs[key]
        post_probs["%s|"%ypart+",".join(xparts)] = priors[ypart]*prob
        norm += priors[ypart]*prob
    
    if sum(post_probs.values()) == 0:
        for ele in post_probs:
            post_probs[ele] = 1/len(post_probs)
        norm = 1
        
    for ele in post_probs:
        post_probs[ele] = post_probs[ele]/norm
 
    return post_probs

def train_test_split(X,y,test_frac=0.5):
    inxs = list(range(len(y)))
    np.random.shuffle(inxs)
    X = X.iloc[inxs,:]
    y = y.iloc[inxs]
    Xlen = round(len(X) * test_frac)
    ylen = round(len(y) * test_frac)
    Xtrain, Xtest = X.iloc[:Xlen], X.iloc[Xlen:]
    ytrain, ytest = y.iloc[:ylen], y.iloc[ylen:]
    return Xtrain,ytrain,Xtest,ytest

def exercise_6(Xtrain,ytrain,Xtest,ytest):
    probs = class_conditional(Xtrain, ytrain)
    priors = compute_priors(ytrain)
        
    ypred = []
    for x in Xtest.iterrows():
        x = x[1]
        post_prob = posteriors(probs, priors, x)
        max = -1
        maxkey = ""
        for key in post_prob.keys():
            if (post_prob[key] > max):
                max = post_prob[key]
                maxkey = key
        values = maxkey.split('=')
        values = values[1].split('|')
        value = values[0]
        ypred.append(int (value))

    accuracy = sum(ypred == ytest)/ytest.size
    return accuracy

def exercise_7(Xtrain,ytrain,Xtest,ytest, npermutations = 10):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = exercise_6(Xtrain,ytrain,Xtest,ytest)
    # now carry out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtest2 = Xtest.copy()
            Xtest2[col] = Xtest[col].sample(frac=1, replace=False).values
            #calculate accuracy of Xtest2 and add to importances[col]
            perm_acc = exercise_6(Xtrain, ytrain, Xtest2, ytest)
            importances[col] += perm_acc
        #importances[col] = importances[col]/npermutations
        importances[col] = orig_accuracy - importances[col]/npermutations
    return importances

def exercise_8(Xtrain,ytrain,Xtest,ytest, npermutations = 20):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = exercise_6(Xtrain,ytrain,Xtest,ytest)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtrain2 = Xtrain.copy()
            Xtrain2[col] = Xtrain[col].sample(frac=1, replace=False).values
            #calculate accuracy of Xtrain2 and add to importances[col]
            perm_acc = exercise_6(Xtrain2, ytrain, Xtest, ytest)
            importances[col] += perm_acc
        importances[col] = orig_accuracy - importances[col]/npermutations
    return importances