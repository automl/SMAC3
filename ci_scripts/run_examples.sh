cd examples

for script in rf.py rosenbrock.py svm.py
do
    python $script
    rval=$?
    if [ "$rval" != 0 ]; then
        echo "Error running example $script"
        exit $rval
    fi
done

cd spear_qcp
bash run.sh
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running example QCP"
    exit $rval
fi
