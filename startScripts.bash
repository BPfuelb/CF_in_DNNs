# param 1 experiment ID
# param 2 parallel processes to run
# optional: param #3 prefix before .bash of the generated scripts
killall -9 sh
killall -9 python
if [ "${3}" == "" ] ; then
  echo "generating scripts.."
  python doExperiments_saad.py ${1} ${2}
else
  echo "generating remaining scripts"
  python processBashScript.py ${3} ./${1}-part-?.bash
fi
chmod 777 *.bash
rm nohup.out
for ((i = 0 ; i < ${2} ; i = i + 1 )) ; do
  nohup ./${1}-part-${i}${3}.bash &
done
  