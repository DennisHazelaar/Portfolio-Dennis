IMAGE=straattaal-backend
WHEEL=$(shell ls dist/slanggen-*.whl 2>/dev/null)
MODEL=artefacts/model.pth

.PHONY: all wheel train build run

all: run

wheel:
	uv build --clean

train:
	python src/slanggen/main.py

check-wheel:
ifndef WHEEL
	$(MAKE) wheel
endif

check-model:
ifeq ($(wildcard $(MODEL)),)
	$(MAKE) train
endif

build: check-wheel check-model
	docker build -t $(IMAGE) .

run: build
	docker run -p 80:80 $(IMAGE)
