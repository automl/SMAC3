cd examples

for script in *.py
do
    echo '###############################################################################'
    echo '###############################################################################'
    echo "Starting to test $script"
    echo '###############################################################################'
    python $script
    rval=$?
    if [ "$rval" != 0 ]; then
        echo "Error running example $script"
        exit $rval
    fi
done

echo '###############################################################################'
echo '###############################################################################'
echo "Starting to test Spear QCP SMAC"
echo '###############################################################################'
cd spear_qcp
bash run_SMAC.sh
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running example QCP"
    exit $rval
fi

echo '###############################################################################'
echo '###############################################################################'
echo "Starting to test Spear QCP ROAR"
echo '###############################################################################'
bash run_ROAR.sh
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running example QCP"
    exit $rval
fi

echo '###############################################################################'
echo '###############################################################################'
echo "Starting to test Spear QCP Successive halving"
echo '###############################################################################'
python SMAC4AC_SH_spear_qcp.py
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running python example QCP"
    exit $rval
fi

cd ..

echo '###############################################################################'
echo '###############################################################################'
echo "Starting to test branin_fmin.py"
echo '###############################################################################'
cd branin
python branin_fmin.py
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running example branin_fmin.py"
    exit $rval
fi

echo '###############################################################################'
echo '###############################################################################'
echo "Starting to test branin from the command line"
echo '###############################################################################'
python ../../scripts/smac --scenario scenario.txt
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running example QCP"
    exit $rval
fi
