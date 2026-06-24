export SMACVERSION="2.4.0"
make clean 
make build
pip install uv
rm -r smac_test || true
uv venv --python=3.12 smac_test
source smac_test/bin/activate
uv pip install dist/smac-$SMACVERSION.tar.gz
python -c 'import smac'
