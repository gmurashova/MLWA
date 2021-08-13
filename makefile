MODULENAME = MLWA 

init:
	conda env create --prefix ./envs --file environment.yml
    
install:
	pip install -e .

docs:
	pdoc3 --force --html --output-dir ./docs $(MODULENAME)

lint:
	pylint $(MODULENAME) 

doclint:
	pydocstyle $(MODULENAME)

test:
	pytest -v $(MODULENAME) 
    
UML:
	pyreverse -ASmy -o png $(MODULENAME)
	mv *.png ./docs/images

.PHONY: init docs lint test 
