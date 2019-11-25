demo:
	@echo "Running demo..."
	@compression.py data/test2.txt


paper.pdf:
	@echo "Making paper..."
	@cp DOC/paper.pdf .

test:
	@echo "Running tests..."
	@test.py

# erases the *.o files AND the DOC directory
clean:
	@echo "Making clean..."
	@cd DOC; rm *.aux *.log *.out

# erases paper files
spotless: clean
	@echo "Making spotless..."
	@\rm -f paper.tex  all.tar all.tar.gz

all.tar:
	@tar cvf all.tar compression.py graph_stats.py DOC Makefile README.md

all.tar.gz: all.tar
	@echo "Making tarfile..."
	@gzip all.tar
