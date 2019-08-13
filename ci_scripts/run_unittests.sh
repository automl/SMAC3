echo Testing revision $(git rev-parse HEAD) ...
echo Testing from directory `pwd`
conda list
make test && bash scripts/test_no_files_left.sh
