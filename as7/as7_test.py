import as7
#test data
prior =[.1,.2,.4,.2,.1] #prior probabilities of hypotheses
likelihood_lime=[0,.25,.5,.75,1] #likelihood of lime candy, given the corresponding hypothesis.
training=[1,1,1,1,1] #training set, evidences in 5 consecutive draws. 1 for lime, 0 for cherry.
tPoint=1 #value of the test data point, 1 for lime, 0 for cherry.

#calculate the posterior probabilities
tPosProb=as7.posteriorFunc(prior, likelihood_lime, training)
print(tPosProb) #should be [0.0, 0.0012195121951219514, 0.07804878048780489, 0.2963414634146342, 0.6243902439024391]

#run the prediction
tPredictProb= as7.predictionFunc(prior, likelihood_lime, training, tPoint)#Probability of drawing value tPoint 
print(tPredictProb)# should be 0.8859756097560977