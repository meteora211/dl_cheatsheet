from tvm.driver import tvmc

model = tvmc.load('../models/resnet50-v2-7.onnx')

model.summary()

# tune model
tvmc.tune(model, target="llvm")

# compile model
package = tvmc.compile(model, target="llvm")

result = tvmc.run(package, device="cpu")
