# Makefile for Sphinx documentation

SPHINXBUILD   = sphinx-build

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable. Alternatively you can add the directory with the executable to your PATH. If you don't have Sphinx installed, grab it from http://sphinx-doc.org/)
endif

.PHONY: help clean buildapi html

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean      remove rendered files"
	@echo "  buildapi   to rebuild the api-documentation"
	@echo "  html       to make standalone HTML files"
	@echo "  all        clean, buildapi, html"

clean:
	rm -rf apidoc
	rm -rf html
	rm -rf pages/examples

buildapi:
	rm -rf apidoc

	sphinx-apidoc -fePTMo apidoc/ ../smac/ --templatedir themes/smac/templates/apidoc
	rm apidoc/smac.rst
	rm apidoc/smac.configspace*

html:
	sphinx-build -b html . html -a

all:
	make clean
	make buildapi
	make html