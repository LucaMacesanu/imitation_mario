


#################################################################
#################################################################
#################################################################
#Algo to find best hyperparameters###############################
#################################################################
#################################################################
################################################################# 
boosting_parameters = {'n_estimators':[50+10*i for i in range(100)]}
RFC_parameters = {'n_estimators':[100+10*i for i in range(100)], 'criterion':['gini','entropy','log_loss'], 'max_depth':[0+i for i in range(100)]}
cv = 5
tuning_ADABooster = BoostingAgent()
tuning_RFC = EnsembleAgent()

tuned_RFC_parameters = tuning_RFC.evaluate(features, labels, cv, RFC_parameters)
tuned_ADABooster_parameter = tuning_ADABooster.evaluate(features, labels, cv, boosting_parameters)


#################################################################
#################################################################
#################################################################
#Algo to load data into training for the RFC Agent###############
#################################################################
#################################################################
#################################################################

RFC = EnsembleAgent(n_estimators = tuned_RFC_parameters['n_estimators'], criterion = tuned_RFC_parameters['criterion'], max_depth = tuned_RFC_parameters['max_depth'])
ADABooster = BoostingAgent(n_estimators = tuned_ADABooster_parameter)

for i in range(batchsize):
    alldata = dataloader(filepath="./recordings", batch_size=100) #Note: arbitrarily set the batch size
    states, actions = alldata.get_next_batch()

    #States is a np array, actions is a np array
    RFC.fit(states[i], actions[i])
    ADABooster.fit(states[i], actions[i])
    








