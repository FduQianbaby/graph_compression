demo:
	@echo "Running demo..."
	@compression.py data/test2.txt

paper.pdf:
	@echo "Making paper..."
	@cp DOC/paper.pdf .

test:
	@echo "Running tests..."
	@test.py

clean:
	@echo "Making clean..."
	@cd pkl_files; rm *.pkl
	@cd plots; rm *.png

spotless: 
	@echo "Making spotless..."
	@\rm -f paper.tex  all.tar all.tar.gz

all.tar:
	@tar cvf all.tar compression.py graph_stats.py DOC Makefile README.md

all.tar.gz: all.tar
	@echo "Making tarfile..."
	@gzip all.tar
