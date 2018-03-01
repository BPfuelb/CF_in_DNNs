# generates bash files for doing excperiments
# each experiments produces exactly 4 outputs:
# modelID_runID_D1D1.csv, D2D1.csv, D2D2.csv, D2D-1.csv
# D1D1: training on D1, test on D1
# D2D1: training on D2, test on D1
# D2D2: training on D2, test on D2
# D2D-1: training on D2, test on D1uD2 using all readout layers. This should be identical to test 
# with one readout layer only for non-MRL experiments
# 
import os, sys, itertools


def getScriptName(expID):
    if expID in ["fc", "D-fc", "fc-MRL", "D-fc-MRL"]:
        return "./Dropout_Experiments/dropout_more_layers.py "
    elif expID in ["conv", "D-conv", "conv-MRL", "D-conv-MRL"]:
        return "./Dropout_Experiments/dropout_more_layers.py --dnn_model cnn "
    elif expID in ["LWTA-fc", "LWTA-fc-MRL"]:
        return "./Dropout_Experiments/dropout_more_layers.py --dnn_model lwta "
    elif expID == "EWC" or expID=="D-EWC":
        return "./ewc_with_options.py"


# takes a task ID and returns the repartition of classes
# D1,D2: initial and retraining
# D3: union for baseline comp.
def generateTaskString(task):
    D1 = ""
    D2 = ""
    D3 = ""
    if task == "D5-5":
        D1 = "0 1 2 3 4"
        D2 = "5 6 7 8 9"
    elif task == "D5-5b":
        D1 = "0 2 4 6 8"
        D2 = "1 3 5 7 9"
    elif task == "D5-5c":
        D1 = "3 4 6 8 9"
        D2 = "0 1 2 5 7"
    elif task == "D5-5d":
        D1 = "0 2 5 6 7"
        D2 = "1 3 4 8 9"
    elif task == "D5-5e":
        D1 = "0 1 3 4 5  "
        D2 = "2 6 7 8 9"
    elif task == "D5-5f":
        D1 = "0 3 4 8 9"
        D2 = "1 2 5 6 7"
    elif task == "D5-5g":
        D1 = "0 5 6 7 8"
        D2 = "1 2 3 4 9"
    elif task == "D5-5h":
        D1 = "0 2 3 6 8"
        D2 = "1 4 5 7 9"
    elif task == "D9-1":
        D1 = "0 1 2 3 4 5 6 7 8"
        D2 = "9"
    elif task == "D9-1b":
        D1 = "1 2 3 4 5 6 7 8 9"
        D2 = "0"
    elif task == "D9-1c":
        D1 = "0 2 3 4 5 6 7 8 9"
        D2 = "1"
    elif task == "D8-1-1":
        D1 = "0 2 3 4 5 6 7 8"
        D2 = "9"
        D3 = "1"
    elif task == "D7-1-1-1":
        D1 = "2 3 4 5 6 7 8"
        D2 = "9"
        D3 = "1"
        D4 = "0"
    elif task == "DP10-10":
        D1 = "0 1 2 3 4 5 6 7 8 9"
        D2 = "0 1 2 3 4 5 6 7 8 9"
    elif task == "DP5-5":
        D1 = "0 1 2 3 4"
        D2 = "5 6 7 8 9"
    elif task == "D5a-1a":
        D1 = "0 1 2 3 4"
        D2 = "5"
    elif task == "D5a-1b":
        D1 = "0 1 2 3 4"
        D2 = "6"
    elif task == "D5a-1c":
        D1 = "0 1 2 3 4"
        D2 = "7"
    elif task == "D5a-1d":
        D1 = "0 1 2 3 4"
        D2 = "8"
    elif task == "D5a-1e":
        D1 = "0 1 2 3 4"
        D2 = "9"
    elif task == "D5b-1a":
        D1 = "3 4 6 8 9"
        D2 = "0"
    elif task == "D5b-1b":
        D1 = "3 4 6 8 9"
        D2 = "1"
    elif task == "D5b-1c":
        D1 = "3 4 6 8 9"
        D2 = "2"
    elif task == "D5b-1d":
        D1 = "3 4 6 8 9"
        D2 = "5"
    elif task == "D5b-1e":
        D1 = "3 4 6 8 9"
        D2 = "7"        
    return D1, D2, D1+" "+D2 ;

def generateUniqueId(expID,params):
  h1 = params[3] ;
  h2 = params[4] ;
  if len(params) > 5:
    h3 = params[5] ;
  else:
    h3=0 ;
  return expID + "_" + params[0] + "_lr_" + str(params[1]) + "_retrainlr_"+str(params[2])+"_layers_"+str(h1)+"%"+str(h2)+"%"+str(h3) ;

  

# not complete!!!
def generateCommandLine(expID,scriptName, action, params,maxSteps=2000):

    # create layer conf parameters
    if len(params) == 5:
        nrHiddenLayers = 2
    else:
        nrHiddenLayers = 3
    hidden_layers = ""
    
    for i in range(0, nrHiddenLayers):
        hidden_layers += "--hidden" + str(i + 1) + " " + str(params[3 + i]) + " "

    D1, D2, D3 = generateTaskString(params[0])

    mlrExperiment = False;
    if expID.find("MRL") != -1:
      mlrExperiment = True;

    trainingReadoutStr = " --training_readout_layer 1" ;
    testingReadoutStr = " --testing_readout_layer 1" ;
    testing2ReadoutStr = " --testing2_readout_layer 1" ;
    testing3ReadoutStr = " --testing3_readout_layer 1" ;
    if mlrExperiment == True:
      if action=="D1D1":
        pass ;
      elif action== "D2DAll":
        trainingReadoutStr = " --training_readout_layer 2" ;
        testingReadoutStr = " --testing_readout_layer 1" ;
        testing2ReadoutStr = " --testing2_readout_layer 2" ;
        testing3ReadoutStr = " --testing3_readout_layer -1" ;

    if expID=="EWC" and action =="D2D-1":
      return "# no D2D-1";
      


    model_name = generateUniqueId(expID,params)
    print(model_name)

    # execString that is command to all experiments..
    execStr = scriptName + " " + hidden_layers + "--max_steps "+str(maxSteps)+" " ;

    if action == "D1D1":
        train_classes = " --train_classes " + D1 + trainingReadoutStr
        test_classes = " --test_classes " + D1 + testingReadoutStr
        train_lr = " --learning_rate " + str(params[1])
        if params[0]  in ["DP10-10","DP5-5"]:
            execStr = execStr + " --permuteTrain 0 --permuteTest 0 "
        execStr = execStr + " " + train_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D1D1 --plot_file " + model_name + "_D1D1.csv" + " --start_at_step 0"
    elif action=="D2DAll":     
        train_classes = " --train_classes " + D2 + trainingReadoutStr
        test_classes = " --test_classes " + D1 + testingReadoutStr
        test2_classes = " --test2_classes " + D2 + testing2ReadoutStr
        test3_classes = " --test3_classes " + D1+" "+D2+" " + testing3ReadoutStr
        supp = "_"+action ;       
        plotFile1 = " --plot_file " + model_name + "_D2D1.csv"
        plotFile2 = " --plot_file2 " + model_name + "_D2D2.csv"
        plotFile3 = " --plot_file3 " + model_name + "_D2D-1.csv"

        retrain_lr = " --learning_rate " + str(params[2])
        if params[0]  in ["DP10-10","DP5-5"]:
            execStr = execStr + "--permuteTrain 1 --permuteTest 0 --permuteTest2 1 --permuteTest3 0"
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + " "+test2_classes+" "+test3_classes+" "+ \
                  " --load_model " + model_name + "_D1D1 --start_at_step "+str(maxSteps)+" "+plotFile1+" "+plotFile2+" "+plotFile3 ;


    elif action == "baseline":
        train_classes = " --train_classes " + D3 + trainingReadoutStr
        test_classes = " --test_classes " + D3 + testingReadoutStr
        train_lr = " --learning_rate " + str(params[1])
        if params[0]  in ["DP10-10","DP5-5"]:
            execStr = execStr + " --permuteTrain 1 --permuteTest 0 --joinTrainTest "
        execStr = execStr + " " + train_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D1D1 --plot_file " + model_name + "_baseline.csv" + " --start_at_step 0"
      
    else:
        return "??" + action

    # Dropout is default in the programs, this disables dropout
    if scriptName.find("D-") == -1:
      if scriptName.find("conv") != -1:
        execStr = execStr + " --dropout 1"
      else:
        execStr = execStr + " --dropout_hidden 1 --dropout_input 1"
    else:
    # Dropout is default in the programs, this enables dropout
      if expID.find("conv") != -1:
        execStr = execStr + " --dropout 0.5"
      else:
        execStr = execStr + " --dropout_hidden 0.8 --dropout_input 0.5"
    

    return execStr.replace("\n"," ")


expID = sys.argv[1]
N_files = sys.argv[2]   # number of files the experiment is divided into!

scriptName = "python "+getScriptName(expID)
# tasks
"""
fc
D-fc
D-fc-MRL
LWTA-fc
LWTA-fc-MRL
conv
conv-MRL
D-conv
D-conv-MRL
EWC
D-EWC
"""
# mondayRuns
tasks = ["DP5-5","DP10-10", "D5-5", "D5-5b", "D5-5c", "D9-1", "D9-1b", "D9-1c"]  # missing D8-1-1, D7-1-1-1 for now
# cvprRuns
tasks.extend( ["D5a-1a","D5a-1b","D5a-1c","D5a-1d","D5a-1e","D5b-1a","D5b-1b","D5b-1c","D5b-1d","D5b-1e"]) ;
# ijcnn runs
tasks = ["DP10-10", "D5-5", "D5-5b", "D5-5c", "D5-5d", "D5-5e", "D5-5f","D5-5g", "D5-5h", "D9-1", "D9-1b", "D9-1c"]  # missing D8-1-1, D7-1-1-1 for now

train_lrs = [0.001]
retrain_lrs = [0.001,0.0001, 0.00001]
# layerSizes = [0,200,400,800]
if expID.find("conv") != -1:
    layerSizes = [1]
else:
    layerSizes = [0,200,400,800]

def validParams(t):
  task,lrTrain,lrRetrain,h1,h2,h3 = t;
  if h1==0 or h2==0:
    return False;
  else:
    return True;

def correctParams(t):
  task,lrTrain,lrRetrain,h1,h2,h3 = t;
  if h3==0:
    return (task,lrTrain,lrRetrain,h1,h2);
  else:
    return t ;

#def removeCheckpoints(checkpointDir,uniqueID):
#  list = 

combinations = itertools.product(tasks, train_lrs, retrain_lrs, layerSizes, layerSizes, layerSizes)
validCombinations = [correctParams(t) for t in combinations if validParams(t)]
#print len(validCombinations) ;

maxSteps = 2500 ;
limit=40000 ;
n = 0
index=0 ;
alreadyDone={}
files = [file(expID + "-part-" + str(n) + ".bash","w") for n in xrange(0,int(N_files))] ;
for t in validCombinations:
    uniqueID = generateUniqueId(expID,t) ;
    #print uniqueID
    if alreadyDone.has_key(uniqueID):
      print "CONT"
      continue;
    alreadyDone[uniqueID]=True;
    f = files[n] ;
    #f.write(generateCommandLine(expID,scriptName, "baseline", t,maxSteps=maxSteps) + "\n")   # initial training
    f.write(generateCommandLine(expID,scriptName, "D1D1", t,maxSteps=maxSteps) + "\n")   # initial training
    f.write(generateCommandLine(expID,scriptName, "D2DAll", t,maxSteps=maxSteps) + "\n")  # retraining and eval on D1
    f.write("rm checkpoints/"+uniqueID+"*\n")
    zipfilename = expID + "-part-" + str(n) + "_csv.zip"
    f.write ("zip "+zipfilename+" "+uniqueID+"_D1D1.csv\n") ;
    f.write ("zip "+zipfilename+" "+uniqueID+"_D2D1.csv\n") ;
    f.write ("zip "+zipfilename+" "+uniqueID+"_D2D2.csv\n") ;
    f.write ("zip "+zipfilename+" "+uniqueID+"_D2D-1.csv\n") ;
    f.write ("rm "+uniqueID+"_D1D1.csv\n") ;
    f.write ("rm "+uniqueID+"_D2D1.csv\n") ;
    f.write ("rm "+uniqueID+"_D2D2.csv\n") ;
    f.write ("rm "+uniqueID+"_D2D-1.csv\n") ;

    n += 1
    if n >= int(N_files):
        n = 0
    index+=1;
    if index>=limit:
      break ;

for f in files:
  f.close() ;
#print alreadyDone
