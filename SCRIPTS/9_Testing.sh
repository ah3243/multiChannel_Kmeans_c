#!/bin/bash

echo Testimg Images

cd ../build3
#rm -rf *

cmake ..
make

echo finished building
echo
        #Program and Parameters
    #Image Scale
    scale=9
    #Corresponding Segment Size to maintain 6 segments
    if [ $scale -eq 9 ] 
    then
        cropSize=35
    elif [ $scale -eq 8 ] 
    then
        cropSize=70
    elif [ $scale -eq 7 ] 
    then
        cropSize=105
    elif [ $scale -eq 6 ] 
    then
        cropSize=140
    fi
    echo The scale is $scale and the Cropping Size is $cropSize
    echo

    imgLocation=../testImage.jpg

    # 1:"flag" 2:"image type" 3:"scale" 4:"cropSize"
    ./multiDimen "testing" $imgLocation $scale $cropSize

echo
echo Finished
