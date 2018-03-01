# takes a bash script that runs experiments. Analyzes each command line for a field with .csv in it, and checks if that file exists. If it does, the line is not copied to stdout
# in this way, a new bash script is generated that contains only experiments that were not done before
# param 1. prefix to add before.bash ion new scripts
# params 2-\infty: script names
import os, os.path as osp ;
import sys ;


def processOneScript(scriptName,outstream):
  d = [l  for l in file(scriptName,"r").readlines()] ;
  
  missingIDs = {} ;
  
  for line in d:
      fields = line.strip().split(" ") ;
      foundCsv = False ;
      csvTag = "" ;
      for f in fields:
        if f.find(".csv") != -1:
          csvTag = f;
          break ;
          
      if csvTag == "":
        continue ;
      else:
        if osp.exists(csvTag)==False or (osp.exists(csvTag)==True and len(file(csvTag,"r").readlines()) < 20):
          expID = csvTag.replace(".csv","").replace("_D1D1","").replace("_D2D2","").replace("_D2D-1","").replace("_D2D1","") ;
          missingIDs[expID] = True ;
        

  for line in d:
      fields = line.strip().split(" ") ;
      foundCsv = False ;
      csvTag = "" ;
      for f in fields:
        if f.find(".csv") != -1:
          csvTag = f;
          break ;
          
      if csvTag == "":
        outstream.write(line) ;
      else:
        expID = csvTag.replace(".csv","").replace("_D1D1","").replace("_D2D2","").replace("_D2D-1","").replace("_D2D1","") ;        
        if expID in missingIDs:
          outstream.write(line) ;

        
        
                                                                                   
                                                             
if __name__ == "__main__":    
  for scriptName in sys.argv[2:]: 
    print "script is ", scriptName, "prefix is", sys.argv[1] ;
    outfile = file(scriptName.replace(".bash",sys.argv[1]+".bash"),"w") ;
    processOneScript (scriptName,outfile) ;
    outfile.close() ;
                                                                                                  