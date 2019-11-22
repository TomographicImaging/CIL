#! /bin/bash

ok=0
nok=0
for i in {1..1000}; 
  do 
    python debug_subset.py ; 
    if [ ! $? == 0 ] ; 
      then let "nok+=1" ; 
    else 
      let "ok+=1"; 
    fi
    echo "OK $ok   NOK $nok"
done

echo "Successful run $ok / 1000"
echo "Unsuccessful run $nok / 1000"