.PHONY: install run check
install:
	python -m pip install -r requirements.txt
run:
	python -m app.main
check:
	python -m compileall -q app
