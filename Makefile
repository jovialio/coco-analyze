all:
    # install pycocotools locally
	#pip install -r requirements.txt
	python3 setup.py build_ext --inplace
	rm -rf build

install:
	# install pycocotools to the Python site-packages
	python3 setup.py build_ext install
	rm -rf build
