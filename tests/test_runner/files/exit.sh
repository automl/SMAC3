#!/bin/bash

# Set arguments first
for argument in "$@"
do
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)   

    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}" 
    fi
done

# We simply set the cost to our parameter
exit 1

# Return everything
echo "cost=$cost; status=SUCCESS"