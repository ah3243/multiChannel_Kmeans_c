#!/bin/bash
set -e

############################################################
###                 CROPPING VARIATION                   ###
############################################################


## cd to build dir and make
cd ../build/
cmake ..
make
echo made and built


  # Generate new images if flag input
  if [ "$useImgs" == "Y" ]
  then
	~/Desktop/TEST_IMAGES/CapturedImgs/./ranImgs.sh
	echo images randomised and allocated
  fi

	## VARIABLE TYPES
	inputType="cropping"
	## SCALE VALUES
	Scale=8
  testRepeats=10

	firstGo=1 # Prevents results labels being printed after the first iteration
	counter=0

    ## Assign correct cropping sizes for current scale
    if [ $Scale -eq 9 ]
    then
        echo "Scale == 9"
      #  Cropping=(5 10 20 30 35 40 50 60 70)
       Cropping=(35)  # missing half of height value
    elif [ $Scale -eq 8 ]
    then
        echo "Scale == 8"
       Cropping=(10 20 40 60 70 80 100 120 140)
      #  Cropping=(70)  # missing half of height value
        # Cropping=(61 62 63 64 65 66 67 68 69 70 71 72) # Additional Values around target croppping size
    elif [ $Scale -eq 7 ]
    then
        echo "Scale == 7"
      #  Cropping=(25 50 80 100 105 125 150 175 200)
       Cropping=(105) # missing half of height value
        # Cropping=(100 102 104 106 107 108 109 110 111 112 113) # Additional Values around target croppping size

    elif [ $Scale -eq 6 ]
    then
        echo "Scale == 6"
        Cropping=(40 80 120 140 160 180 200 240 280)
    fi

	for i in ${Cropping[@]}; do
		# If not first iteration then set flag as 0
		if [ $counter -gt 0 ]
		then
		  echo not first iteration
		  firstGo=0
		fi

		echo starting program

		#Note 1.inputValue  2.Input Type 3.firstLineFlag
		echo Test Type is: $inputType InputParam: ${i} First go?: $firstGo Scale: $Scale testRepeats: $testRepeats
		echo iteration $counter
		echo
		./multiDimen ${i} $inputType $firstGo $Scale $testRepeats
		let counter=counter+1
	done

echo Done
