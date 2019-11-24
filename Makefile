demo:
	@echo "Running demo..."
	compression.py data/test2.txt


paper.pdf:
	@echo "Making paper..."
	cd DOC; make paper.pdf
	mv DOC/paper.pdf .


# erases the *.o files AND the DOC directory
clean:
	@echo "Making clean..."
	cd DOC; rm *.aux *.log *.out

# erases paper files
spotless: clean
	@echo "Making spotless..."
	@\rm -f paper.tex  all.tar all.tar.gz

all.tar:
	@tar cvf all.tar compression.py DOC Makefile README.md

all.tar.gz: all.tar
	@echo "Making tarfile..."
	@gzip all.tar
