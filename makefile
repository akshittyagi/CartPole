run:
	python DecisionProcess.py
run_TD:
	python TDLearning.py -k $(order)
clean:
	rm *.pyc
	rm *.png

