#!/bin/bash

## Check for Model and texton Dictionary, Create if not present

if [ ! -f "../build3/dictionary.xml" ]
then
    echo Texton Dictionary Not found. Generating.
    
else 
    echo Texton Dictionary was found
fi

if [ ! -f "../build3/models.xml" ] 
then
    echo Model Collection Not Found. Generating.
else 
    echo Model Collection was found
fi

    
