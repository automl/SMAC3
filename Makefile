
.PHONY: test
test:
	python3.6 -m nose -sv --with-coverage --cover-package=smac

.PHONY: test-fast
test-fast:
	python3.6 -m nose -a '!slow' -sv --with-coverage --cover-package=smac

.PHONY: test-runtimes
test-runtimes:
	# requires nose-timer
	python3.6 -m nose -sv --with-timer --timer-top-n 15

.PHONY: doc
doc:
	make -C doc html

.PHONY: clean
clean: clean-data
	make -C doc clean

.PHONY: clean-data
clean-data:
	# remove all files that could have been left by test cases or by manual runs
	# feel free to add more lines
	find . -maxdepth 3 -iname 'smac3-output_*-*-*_*' | tac | while read -r TESTDIR ; do rm -Rf "$${TESTDIR}" ; done
	rm -Rf run_*
	rm -Rf test/test_files/scenario_test/tmp_output_*
	rm -Rf test/test_files/test_*_run1
	rm  -f test/test_files/test_scenario_options_to_doc.txt
	rm  -f test/test_files/validation/test_validation_rh.json
	rm -Rf test/run_*
	rm -Rf test/test_smbo/run_*
	rm -Rf test/test_files/test_restore_state/
	rm -Rf test/test_files/test_restored_state/
	rm -Rf test/test_files/validation/test/
	rm  -f test/test_files/validation/validated_runhistory.json*
	rm  -f test/test_files/validation/validated_runhistory_EPM.json*
