# Get onnx model
wget -P ../models https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx 

# Set log level
export TVM_LOG_DEBUG=DEFAULT=1

# Compile onnx model with tvmc
python -m tvm.driver.tvmc compile \
--target "llvm" \
--input-shapes "data:[1,3,224,224]" \
--output resnet50-v2-7-tvm.tar \
../models/resnet50-v2-7.onnx

# Check the compiled module
mkdir compiled
tar -xvf resnet50-v2-7-tvm.tar -C compiled
ls compiled

# Fetch input image and preprocess inputs
python preprocess.py

# Execute
python -m tvm.driver.tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz \
resnet50-v2-7-tvm.tar

# Postprocess outputs
python postprocess.py
