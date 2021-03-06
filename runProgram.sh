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
       echo "you entered Y. Generating New Training Images."
       break
     elif [ "$useImgs" == "N" ]
     then
       echo "you entered N. Continuing with current Images."
       break
     else
       echo "your input was not recognised."
     fi
  done
elif [ $1 == "1" ]
then
  echo "Generating new images."
  useImgs="Y"
elif [ $1 == "0" ]
then
  echo "Using current images."
  useImgs="N"
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
if [ "$useImgs" == "N" ]
then
Repeats=1
fi

echo this is the number of repeats $Repeats

while [ $imgCounter -lt $Repeats ]
do
  if [ "$useImgs" == "Y" ]
  then
	~/Desktop/TEST_IMAGES/CapturedImgs/./ranImgs.sh
	echo images randomised and allocated
  fi
	
	## SECOND VARIABLE VALS
	#Attempts=(5 10 15 20)
	Scale=(9 8 7 6 5 4 3 2 1 0)
#	Cropping=(10 20 40 60 80 100 120 140) ## Scale 8
	
	## FIRST VARIABLE TYPES
	#inputType="attempts"
	#inputType="texDict"
	inputType="scale"
#	inputType="cropping"
	
	secondVal=(0 1) ## For blank secondVal
#	secondVal=(0 0.25 0.5 0.75 1 1.5 2) ##Thresold
	
	## SECOND VARIABLE TYPES

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
		for sevVal in ${secondVal[@]}; do

			./multiDimen ${i} $inputType $firstGo $secVal $secType
		done
		echo Test Type is: $inputType
		echo iteration $counter
		echo
		let counter=counter+1
	done
	let imgCounter=imgCounter+1

done
echo Done
