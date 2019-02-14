# Minimal makefile for Sphinx documentation
#

# Locale
export LC_ALL=C

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = PyTorchTutorials
SOURCEDIR     = .
BUILDDIR      = _build
GH_PAGES_SOURCES = $(SOURCEDIR) Makefile

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile docs

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -v

download:
	# IMPORTANT NOTE: Please make sure your dataset is downloaded to *_source/data folder,
	# otherwise CI might silently break.

	# transfer learning tutorial data
	# wget -N https://download.pytorch.org/tutorial/hymenoptera_data.zip
	# unzip -o hymenoptera_data.zip -d beginner_source/data
	
docs:
	make download
	make html
	rm -rf docs
	cp -r $(BUILDDIR)/html docs
	touch docs/.nojekyll

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	bash .jenkins/remove_invisible_code_block_batch.sh "$(BUILDDIR)/html"
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean-cache:
	make clean
	rm -rf advanced beginner intermediate
