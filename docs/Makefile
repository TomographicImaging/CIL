# You can set these variables from the command line.
JEKYLLOPTS    =
JEKYLLBUILD   = bundle exec jekyll build
JEKYLLSRCDIR  = pages
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
BUILDSUBDIR   = $(shell git rev-parse --abbrev-ref HEAD)
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo extra options
	@echo '  web-deps    to install website dependencies'
	@echo '  web         to build website'
	@echo '  dirhtml     to build docs'
	@echo '  serve       to serve website & docs'
.PHONY: help Makefile serve web-deps web dirhtml

serve:
	@python -m http.server -d "$(BUILDDIR)"
web-deps:
	@bundle install
web:
	@$(JEKYLLBUILD) -s "$(JEKYLLSRCDIR)" $(JEKYLLOPTS)
	@ls "$(JEKYLLSRCDIR)"/*.md | xargs -II basename I .md | xargs -II rm -rf "$(BUILDDIR)/I"
	@rm -rf "$(BUILDDIR)/assets"
	@mv _site/* "$(BUILDDIR)/"
dirhtml:
	@rm -rf "$(BUILDDIR)/$(BUILDSUBDIR)"
	@./mkdemos.py
	@$(SPHINXBUILD) -b dirhtml $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/$(BUILDSUBDIR)"
	@./mkversions.py

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
