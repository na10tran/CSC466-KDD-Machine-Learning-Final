from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings
#suppress warnings
warnings.filterwarnings('ignore')
#normalize data
#

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
    
def activation(net):
    return 1/(1+np.exp(-net))

def train(X,t,nepochs=400,n=0.01,test_size=0.3,val_size=0.3,seed=0):
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size,random_state=seed)
    X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=val_size,random_state=seed)
    
    train_accuracy = []
    val_accuracy = []
    nfeatures = X.shape[1]
    np.random.seed(seed)
    w = 2*np.random.uniform(size=(nfeatures,)) - 1

    for epoch in range(nepochs):
        y_train2 = X_train2.apply(lambda x: activation(np.dot(w,x)),axis=1)    
        y_val = X_val.apply(lambda x: activation(np.dot(w,x)),axis=1)

        train_accuracy.append(sum(t_train2 == np.round(y_train2))/len(t_train2))
        val_accuracy.append(sum(t_val == np.round(y_val))/len(t_val))
        for j in range(len(w)):
            w[j] -= n*np.dot((y_train2 - t_train2)*y_train2*(1-y_train2),X_train2.iloc[:,j])
            
    results = pd.DataFrame({"epoch": np.arange(nepochs)+1, 'train_accuracy':train_accuracy,'val_accuracy':val_accuracy,
                            "n":n,'test_size':test_size,'val_size':val_size,'seed':seed
                           }).set_index(['n','test_size','val_size','seed'])
    return w,X_test,t_test,results

def evaluate_baseline(t_test,t_train2,t_val):
    accuracy_test = max(t_test.value_counts()) / len(t_test)
    accuracy_train2 = max(t_train2.value_counts()) / len(t_train2)
    accuracy_val = max(t_val.value_counts()) / len(t_val)
    return accuracy_test,accuracy_train2,accuracy_val

# zero rule algorithm for classification
#def zero_rule_algorithm_classification(train, test):
#	output_values = [row[-1] for row in train]
#	prediction = max(set(output_values), key=output_values.count)
#	predicted = [prediction for i in range(len(test))]
#	return predicted

#df = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
#s = pd.Series([1, 1, 2, 1])
#df.dot(s)
def predict(w,X,threshold=0.5):
    y = np.round(X.apply(lambda x:activation(np.dot(w,x)),axis=1))
    return y

def confusion_matrix(t,y,labels):
    cm = pd.DataFrame(columns=labels,index=labels)
    cm[0][0] = cm[0][1] = cm[1][1] = cm[1][0] = 0
    y_counts = y.value_counts().tolist()   
    t_counts = t.value_counts().tolist()   
    for row in range(len(y)):
        if (t.iloc[row] == 1 and y.iloc[row] == 1): 
            cm[1][1]  = cm[1][1] + 1
        elif (t.iloc[row] == 0 and y.iloc[row] == 0): 
            cm[0][0] = cm[0][0]  + 1   
        elif (t.iloc[row] == 0 and y.iloc[row] == 1): 
            cm[0][1] =  cm[0][1] + 1   
        else: 
            cm[1][0] = cm[1][0] + 1    
    return cm

def evaluation(cm,positive_class=1):
    if(positive_class == 1):
        TP = cm[1][1]
        FN = cm[1][0]
        FP = cm[0][1]
        TN = cm[0][0] 
    else:
        TP = cm[0][0]
        FN = cm[1][0]
        FP = cm[1][0]
        TN = cm[1][1] 
    accuracy = (TP + TN) / (TP + FN + FP + TN)  # correctly predictd / total amount
    sensitivity = TP / (TP + FN) #true positive rate
    specificity = TN / (TN + FP) #true negative rate
    precision = TP / (TP + FP)  #positive predictive value
    F1 = TP / (TP + 0.5*(FP + FN)) #F1 --> CORRECT 
    stats = {"accuracy":accuracy, "sensitivity/recall":sensitivity, "specifity":specificity, "precision":precision, "F1":F1}
    return stats

def importance(X,t,seeds):
    importances = pd.Series(np.zeros((X.shape[1],)),index=X.columns) 
    for seed in seeds:  #train NN for each seed  
        w = {}
        importance = pd.Series(np.zeros((X.shape[1],)),index=X.columns) #create importance to average later
        w[seed] = train(X ,t,seed = seed)[0]
        seed_total = 0          
        max_val = np.sqrt(pow(w[seed][0],2))   #gets first w list
        for i in range(len(w[seed])):
            if(max_val < np.sqrt(pow(w[seed][i],2)) ):
                max_val =  np.sqrt(pow(w[seed][i],2))
        total = 0
        for i in range(len(importance)):
            temp = np.sqrt(pow(w[seed][i],2)) / (max_val * 1.0)
            importances.iloc[i] = importances.iloc[i] + temp 
    for i in range(len(importances)):
        importances.iloc[i] = importances.iloc[i] / (len(seeds) * 1.0)
    return importances

