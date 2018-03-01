# reads all csv files in current dir. that come from from one experiment and gives the best run results
# "best"-->fitness function, can be adapted to fit a particular need
# re-uses the data reading method from the plotting script
# cmd line params: python findBestRun.py modelID evaluationMethod <useMRL>

from __future__ import division
from plotOneExp import readResults
import sys, os
import re, numpy as np, math,pickle

from plotOneExp import readResults
import sys, os, numpy as np


# fitness function
# takes a list of 3 np arrays, containing the accuracies over time from D1D1,D2D2, D2D1 (in that order)
def measureQuality(D):
    return (D[0][:, 1]).max()  # criterion: highest initial training accuracy


def measureQualityWithAvg(D, task):
    temp_acc = 0
    for idx, acc_D2 in enumerate(D[1][:, 1]):
        if (round(acc_D2) != 0) and (0 <= acc_D2 - temp_acc <= 0.5):
            D1_weight, D2_weight = getWeightsForAvg(task)
            acc_D1 = D[2][idx, 1]
            return np.average([acc_D1, acc_D2], weights=[D1_weight, D2_weight])
        else:
            temp_acc = acc_D2


def measureQualityWithPcnt(D, task):
    maxVal = D[1][:, 1].max()
    for idx, acc_D2 in enumerate(D[1][:, 1]):
        if acc_D2 >= maxVal * 0.98:
            D1_weight, D2_weight = getWeightsForAvg(task)
            acc_D1 = D[2][idx, 1]
            return np.average([acc_D1, acc_D2], weights=[D1_weight, D2_weight])


# best performance on D1
# quality = performance auf gesamtdatensatz zum ZP i
# structure of D is always (D1D1,D2D2,D2D1,D2DAll)
def measureQualityAlexD1(D,w1,w2,**kwargs):
    #print w1,w2 ;
    if D is None:
      return -1.0 ;
    for d in D:
      if d.shape[0] < 20:
        return -1.0 ;
    D1D1 = D[0] ;
    return D1D1[:,1].max() ;


# evaluqtes based on D2 perf only
# stop criterion is 99.9% of maximal D2 test performance while retraining
# i = wann erreich D2D2 k M seines max?
# quality = performance auf gesamtdatensatz zum ZP i
# structure of D is always assumed to be (D1D1,D2D2,D2D1,D2DAll)
def measureQualityAlexD2(D,wD1,wD2,**kwargs):
    if D is None:
      return -1.0 ;
    D2D2 = D[1]
    D2D1 = D[2] ;
    for d in D:
      if d.shape[0] < 20:
        return -1.0 ;
    maxD2D2 = D2D2[:, 1].max() * 0.999
    for i in xrange(0, D2D2.shape[0]):
        if D2D2[i, 1] >= maxD2D2:
            #return wD2 * D2D2[i, 1] + wD1 * D2D1[i, 1]
            return D[3][i,1]


# i = wann erreich D2D2 k M seines max?
# quality = performance auf gesamtdatensatz zum ZP i
# aber: lies performance auf D1 aus Datei _D2D-1 ab
# structure of D is always (D1D1,D2D2,D2D1,D2DAll)
def measureQualityAlexDAll(D,wD1,wD2,**kwargs):
    if D is None:
      return -1.0 ;
    if len(D) < 4:
      return -1
    for d in D:
      if d.shape[0] < 20:
        return -1.0 ;

    D2DAll = D[3]
    maxD2DAll = D2DAll[:, 1].max() * 0.999
    return maxD2DAll ;
    for i in xrange(0, D2DAll.shape[0]):
        if D2D2[i, 1] >= maxD2D2:
          return wD1 * D[3][i, 1] + wD2 * D[1][i, 1]              


def extractTask(runID):
    fields = runID.split("_")
    params=fields[2] ;
    for f in fields[3:]:
      params = params+"_"+f;

    return fields[1],params


def getWeightsForAvg(task):
    p1 = float(re.search(r'\d+', task.split("-")[0]).group())
    p2 = float(re.search(r'\d+', task.split("-")[1]).group())
    return p1 / (p1+p2), p2 / (p1+p2)

def createLookupTables (expDict):
  taskLookup = {}
  paramLookup = {}
  invTaskLookup = {} ;
  invParamLookup = {} ;

  taskCount = 0 ;
  paramCount=0;
  for key in expDict:
    _t,_p = extractTask(key) ; 
    if taskLookup.has_key(_t)==False:
       taskLookup[_t]=taskCount ;
       invTaskLookup[taskCount]=_t ;
       taskCount+=1;
    if paramLookup.has_key(_p)==False:
       paramLookup[_p]=paramCount ;
       invParamLookup[paramCount]=_p ;
       paramCount+=1;

  return taskLookup,paramLookup,invTaskLookup,invParamLookup ;

"""
Creates a 2D matrix containing a numerical quality measure for each pair of task (axis0) / params set (axis1).
Arguments:
expDict: contains information about all conducted experiments. Key: runID, Value; List of arrays representing values in csv files
qualityMeasure: measure for non-MRL
qualityMeasureMRL: measure for MRL
useMRL: *g*
"""
def calcPerfMatrix(expDict,qualityMeasure,qualityMeasureMRL,useMRL=False):
  # create dictionaries that map tasks to integers
  # create dictionaries that map paramater sets to integers
  # and vice versa
  taskLookup,paramLookup, invTaskLookup, invParamLookup = createLookupTables (expDict) ;

  # allocate res matrix
  resultMatrix = np.zeros([len(taskLookup.keys()),len(paramLookup.keys())]) ;

  validExps = 0 ;
  for key,value in expDict.iteritems():
    if len(value.keys()) >= 3:
      #print "valid exp", key ;
      validExps += 1 ;

      task,params = extractTask(key) ;
      
      fitness = qualityMeasure(readResults(key,pathString),*(getWeightsForAvg(task))) ;
      if useMRL==True:
        fitness = qualityMeasureMRL(readResults(key,pathString),*(getWeightsForAvg(task))) ;  
      resultMatrix[taskLookup[task],paramLookup[params]] = fitness ;      

    else:
      print "invalid exp", key, len(value.keys()) ;
  return resultMatrix,taskLookup,paramLookup,invTaskLookup,invParamLookup ;
  
def printResultMatrix(resultMatrix,taskLookup,paramLookup):
  tpm= resultMatrix.transpose()
  for task,taskI in taskLookup.iteritems():
    print "%6s"%(task),
  print
  for param,paramI in paramLookup.iteritems():

    for task,taskI in taskLookup.iteritems():
      print "%.4f"%(resultMatrix[taskI,paramI]),
    print param

def writeMatrixToFile(name, resultMatrixTrainRetrain, taskLookup,paramLookup):
  pickle.dump((resultMatrixTrainRetrain, taskLookup,paramLookup),file(name,"wb")) ;

def listExperiments(pathString, expID):
  csvfiles = [f for f in os.listdir(pathString) if (f.find(".csv") != -1 and (f.split("_"))[0] == expID)]
  expDict = {}

  for f in csvfiles:
    fields = f.replace(".csv", "").split("_")
    action = fields[-1]
    runID = f.replace("_" + action + ".csv", "")
    # print runID,action
    if expDict.has_key(runID) == False:
      expDict[runID] = {}
      expDict[runID][action] = f
    else:
      expDict[runID][action] = f
  return expDict ;

# paramStr is supposed to be of the form: fieldName_value{_fieldName_value}
def getParamValue(paramStr,fieldName):
  spl = paramStr.split('_') ;
  fieldNames = spl[0::2] ;
  values = spl[1::2] ;

  for i in xrange(0,len(fieldNames)):
    if fieldNames[i] == fieldName:
      return values[i] ;
  return None;
  


# -------------------------main---------------------------------

expID = sys.argv[1]
pathString = "./"
evalMode = sys.argv[2] ;
useMRL = False
if len(sys.argv) >= 4:
    useMRL = True


# expDict: keys are runIDs composed of dataset_params
#          values are lists of csv files
expDict = listExperiments(pathString,expID) ;
# tasks contains just the dataset without the params


"""
steps for each task:
1) find model (spec. by. topology, D1 learning rate) that performs best on D1
2) from all models that share topology and D1 learning rate from 1), select the best on D2
"""
if evalMode == "realistic":
  resultMatrixTrain,taskLookup,paramLookup,invTaskLookup,invParamLookup = calcPerfMatrix(expDict,measureQualityAlexD1,measureQualityAlexD1,useMRL) ;
  resultMatrixRetrain,taskLookup,paramLookup,invTaskLookup,invParamLookup = calcPerfMatrix(expDict,measureQualityAlexD2,
                                                                                           measureQualityAlexDAll,useMRL) ;

  printResultMatrix(resultMatrixTrain, taskLookup, paramLookup) ;
  print "!!"
  printResultMatrix(resultMatrixRetrain, taskLookup, paramLookup) ;

  resDict = {}
  latexStr = expID+ " " ;
  for task,taskI in taskLookup.iteritems():
    bestParamIonD1 = resultMatrixTrain[taskI,:].argmax() ;
    bestModelOnD1 = invParamLookup[bestParamIonD1] ;
    #print bestModelOnD1;
    d1arch = getParamValue(bestModelOnD1,"layers") ;
    d1lr = getParamValue(bestModelOnD1,"lr") ;
    bestPerfMeasure=-1.0 ;
    for params,paramI in paramLookup.iteritems():
      if getParamValue(params,"layers")==d1arch and getParamValue(params,"lr")==d1lr:
        perfMeasure = resultMatrixRetrain[taskI,paramI] ;
        #print "searching model", params, perfMeasure ;
        if perfMeasure>bestPerfMeasure:
          bestPerfMeasure=perfMeasure ;
          bestParams = params ;
    print 'Task',task, "model=",expID+"_"+task+"_"+bestParams,"retrain perf incremental=",bestPerfMeasure
    resDict[task] = "%.02f"%(bestPerfMeasure); 
    
  print sorted(resDict.keys())
  for key in sorted(resDict.keys()):
    latexStr = latexStr+" & "+resDict[key] ;
  print latexStr ;

elif evalMode.find("realisticDebug") != -1:
  resultMatrixTrain,taskLookup,paramLookup,invTaskLookup,invParamLookup = calcPerfMatrix(expDict,measureQualityAlexD1,measureQualityAlexD1,useMRL) ;
  resultMatrixRetrain,taskLookup,paramLookup,invTaskLookup,invParamLookup = calcPerfMatrix(expDict,measureQualityAlexDAll,
                                                                                           measureQualityAlexD2DAll,useMRL) ;

  #printResultMatrix(resultMatrixRetrain,taskLookup,paramLookup) ;

  modelStr = evalMode.replace("realisticDebug","") ;
  specTask,specParams = extractTask(modelStr) ;
  for task,taskI in taskLookup.iteritems():
    if task!=specTask:
      continue ;
    paramI = paramLookup[specParams] ;
    measureD1 = resultMatrixTrain[taskI,paramI] ;
    perfMeasure = resultMatrixRetrain[taskI,bestParamI] ;
    bestModel = modelStr ;
    print 'Task',task, "model=",expID+"_"+task+"_"+bestModel,"retrain perf incremental=",perfMeasure



elif evalMode == "prescient":
  resultMatrixTrainRetrain,taskLookup,paramLookup,invTaskLookup,invParamLookup = calcPerfMatrix(expDict,measureQualityAlexDAll,
                                                                                                measureQualityAlexDAll,useMRL) ;

  writeMatrixToFile(expID+".pkl", resultMatrixTrainRetrain, taskLookup,paramLookup)   ;

  resDict = {}
  for task,taskI in taskLookup.iteritems():
    bestParamI = resultMatrixTrainRetrain[taskI,:].argmax() ;
    bestModel =invParamLookup[bestParamI] ;
    perfMeasure = resultMatrixTrainRetrain[taskI,bestParamI] ;
    print 'Task',task, "model=",expID+"_"+task+"_"+bestModel,"retrain perf incremental=",perfMeasure
    resDict [task] = "%.02f"%(perfMeasure);

  latexStr = expID+" "
  print sorted(resDict.keys())
  for key in sorted(resDict.keys()):
    latexStr = latexStr+" & "+resDict[key] ;
  print latexStr ;



"""
invalid_tasks = []
for key in tasks:
    if bestRunID[key] is not None:
      print "Task ", key, ": best/worst run was", bestRunID[key],"/",worstRunID[key], " with a fitness of ", bestFitness[key],      "/",worstFitness[key], "mean/var=",sumX[key]/(count[key]+0.001),math.sqrt((sumX2[key]/(count[key]+0.001)-(sumX[key]/(count[key]+0.001))**2.)) ;
    elif key not in invalid_tasks:
        invalid_tasks.append(key)

if invalid_tasks:
    print "Some invalid experiment results for %s were omitted" % invalid_tasks
"""


