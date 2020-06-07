# Jasmin-demo
A home assignment for the Jasmin interview process

## Tasks
* modify a pre-trained model by adding an extra output to the model which provides an embedding of the image. (keep the original predictions)
* Optimize the model for faster inference and write why you choose the selected method(s)

* Pre-processing: original image --> "zoom in" by 150% --> Randomly crop 100x100 rectangle --> pad it with zeros so
the image can be fed to the network

* Post-processing:
    * Prediction output: softmax output --> first and second most probable predictions with the "softmax
confidences"
    * Embedding output: embedding output --> get the mean and variance (per image)

* Compare the runtime of the python and C++ implementation


## Prerequisites
#### Minimal
Docker with your local user added to the user group docker.
#### Running locally
python3 and pip3 

To access the jupyter notebooks, you also need to have to `pip install jupyterlab`,
then start the server so it has access to the `notebooks/` folder.

## Installation
The project is containerized, use `sh pull_docker.sh` from the project root to get the correct container.

If you don't want to pull it, use `sh build_docker.sh`.

## Usage
`sh run_docker.sh` will start you the container. This container starts the demo script.
You can also check the Jupyter notebook in the `notebooks` folder to get a more detailed explanation of the demo. 
If you want to run it without the container, make sure you create a virtual environment, and run `pip3 install -r .docker/requirements.txt` in it.

## Design decisions
PyTorch + ResNet50: industry standard backend with a moderately sized model which provides relatively high accuracy. 
Possible alternatives would provide higher overall speed (eg. MobileNet-v2) or higher overall accuracy (ResNeXt 50)

## Jupyter notebook
jupyter is available in the dockerfile. to run it, have the container locally, then execute `sh start_jupyter.sh`
You will find the notebooks in the `notebooks/` folder.

## Further development possibilities
A possible improvement could be made by using a CUDA-enabled device. Deployment to such an environment is possible using `nvidia-docker`.

A possible deployment option would be to use TVM as it provides a lightweight runtime API so we don't need the complete framework on the deployment target.
Only the compiling machine needs the whole framework, cross-compilation is supported (ideal for deployment targets with different architecture)

Another improvement would be to use Dali to do distributed GPU pre-processing.

Faster inference could be achieved with model optimization, quantization or destillation.

A possible deployment pipeline would be: 
1. Code change triggers CI (Jenkins, Gitlab, etc.)
2. TVM compiles and optimizes the model for runtime performance
3. Using the artifacts, we generate a docker container
4. Use kubernetes to control deployment on various environments and architectures 
where depending on the GPU capabilities we enable DALI to feed the datasets to the model.

## Alternative deployment options
Depending on the deployment environment it would be possible to get better results with using C++.
We could reason that said environment has disadvantages in terms of performance or other multi-threading bottlenecks.
Training the model is still possible in Python, so prototyping would be faster, then use
TorchScript to translate the model to C++.