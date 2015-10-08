#!/bin/bash
set -e


#USEAGE INSTRUCTIONS:
# PARAMETER 1
#  1= GENERATE NEW RANDOMISED TRAINING AND TEST IMAGES
#  2= USE CURRENT IMAGES IN 'CLASS' AND 'NOVELIMGS'

# Check for input about using same images
if [ $# -eq 0 ]
  then
  echo "Command line input not detected."
  # If not parameter detected then get input
  while [ 1 ]
  do
    echo  "Please enter Y to generate new training and test images or N to use the current ones."
     read useImgs
     echo this was your answer: $useImgs
     if [ "$useImgs" == "Y" ]
     then
       echo "you entered Y"
       break
     elif [ "$useImgs" == "N" ]
     then
       echo "you entered N continuing"
       break
     else
       echo "your input was not recognised."
     fi
  done
elif [ $1 == "1" ]
then
  echo "Generating new images."
  useImgs="N"
elif [ $1 == "0" ]
then
  echo "Using current images."
  useImgs="Y"
else
  echo "Input parameter not recognised as 1 or 0. Exiting"
  exit 1
fi


cd build/
cmake ..
make
echo made and built

Repeats=2
imgCounter=0

# Only run tests once if static images used
if [ "$useImgs" == "Y" ]
then
Repeats=1
fi


echo this is the number of repeats $Repeats
exit

while [ $imgCounter -lt $Repeats ]
do
  if [ "$useImgs" == "N" ]
  then
	~/Desktop/TEST_IMAGES/CapturedImgs/./ranImgs.sh
	echo images randomised and allocated
  fi

	#Attempts=(5 10 15 20)
	Scale=(6 5)

	#inputType="attempts"
	#inputType="texDict"
	inputType="scale"
	#inputType="cropping"

	firstGo=1
	counter=0
	for i in ${Scale[@]}; do

		if [ $counter -gt 0 ]
		then
		  echo not first iteration
		  firstGo=0
		fi

		echo starting program
		#Note 1.inputValue  2.Input Type 3.firstLineFlag
		echo this is the bool value $firstGo
		./multiDimen ${i} $inputType $firstGo
		echo Test Type is: $inputType
		echo iteration $counter
		echo
		let counter=counter+1
	done
	let imgCounter=imgCounter+1

done
echo Done
