# plots thee csv files from cf experiments into a single png file that is named according to the experiments parameters
import numpy as np ;
import os,re ;

def getWeightsForAvg(task):
    p1 = float(re.search(r'\d+', task.split("-")[0]).group())
    p2 = float(re.search(r'\d+', task.split("-")[1]).group())
    return p1 / (p1+p2), p2 / (p1+p2)


# takes a model string and returns a list of three numpy arrays with the experimental results stored in 2D matrices
# dim0 - iteration, dim1 - accuracy
def readResults(modelString, pathString):
  F = [modelString+"_D1D1.csv", modelString+"_D2D2.csv",modelString+"_D2D1.csv"] ;
  if os.path.exists(pathString + "/" + modelString+"_D2D-1.csv"):
    F.append(modelString+"_D2D-1.csv") ;

  try:
    L = [file(pathString + "/" + f,"r").readlines() for f in F]
  except IOError:
    return None ;

  _D = [[l.strip().split(",") for l in lines if len(l)>2 ] for lines in L] ;

  d = [None for i in F]
  i=0;
  for _data in _D:
    d[i] = [(float(_d[0]),float(_d[1])) for _d in _data]  
    i+=1

  D = [np.zeros([len(dv),2]) for dv in d] ;

  j=0;
  for _d in D:
    i=0;
    for tup in d[j]:
      #print tup
      _d[i,:] = tup ;
      i+=1;
    j+=1;

  return D ;





if __name__=="__main__":
  import matplotlib ;
  matplotlib.use('Agg') ;
  import matplotlib.pyplot as plt ;
  import os, sys, numpy as np ;
  from matplotlib.ticker import MultipleLocator
  
  params = sys.argv[1].split("_") ;
  titleStr = "Model: "+params[0]+", Task: "+params[1] ;
  pathString = "./"
  if len(sys.argv) > 2:
    pathString = sys.argv[2]

  fig = plt.figure(1) ;
  ax = plt.gca() ;


  D = readResults(sys.argv[1], pathString) ;
  w1,w2 = getWeightsForAvg(params[1]) ;
  Dagg = w1*D[2][:,1]+w2 * D[1][:,1] ;


  ax.plot(D[0][:,0],D[0][:,1], linewidth=3,label='train:D1,test:D1')
  ax.plot(D[1][:,0],D[1][:,1], linewidth=3,label='train:D2,test:D2')
  #ax.plot(D[2][:,0],D[2][:,1], linewidth=3,label='train:D2,test:D1')
  #ax.plot(D[2][:,0],Dagg, linewidth=3,label='train:D2,test:All')
  if len(D)>3:
    ax.plot(D[3][:,0],D[3][:,1], linewidth=3,label='train:D2,test:All')
 
  ax.set_title (titleStr, size=25)
  ax.set_xlabel ("iteration", size=30)
  ax.set_ylabel ("test accuracy", size=30)
  ax.tick_params(labelsize=22)
  majXTick = (D[1][:,0].max()+50)/5 ;
  ax.xaxis.set_major_locator(MultipleLocator (majXTick)) ;
  ax.yaxis.set_major_locator(MultipleLocator (0.1)) ;
  ax.yaxis.set_minor_locator(MultipleLocator (0.05)) ;
  ax.legend(fontsize=15,loc='lower left')
  ax.grid(True,which='both');
  x = np.arange(0,(D[1][:,0]).max()+50,1) ;
  ax.fill_between(x,0,1,where=(x>(x.shape[0]/2)),facecolor='gray',alpha=0.3)
  plt.tight_layout()
  figName = sys.argv[1]+".png" ;
  plt.savefig("fig.png") ;
  plt.savefig(figName) ;

