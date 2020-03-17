cd examples

for script in *.py
do
    python $script
    rval=$?
    if [ "$rval" != 0 ]; then
        echo "Error running example $script"
        exit $rval
    fi
done

cd spear_qcp
bash run_SMAC.sh
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running example QCP"
    exit $rval
fi

bash run_ROAR.sh
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running example QCP"
    exit $rval
fi


python SMAC4AC_spear_qcp.py
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running python example QCP"
    exit $rval
fi

cd ..

cd branin
python branin_fmin.py
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running example QCP"
    exit $rval
fi

python ../../scripts/smac --scenario scenario.txt
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running example QCP"
    exit $rval
fi
