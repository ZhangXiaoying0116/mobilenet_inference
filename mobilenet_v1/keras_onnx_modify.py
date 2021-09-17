import onnx_graphsurgeon as gs
import numpy as np
import onnx

## pip install nvidia-pyindex
## pip install onnx-graphsurgeon

model = onnx.load("/home/devdata/xiaoying.zhang/DORADO/common/mobilenet_v1_keras-op13-fp32.onnx")
graph = gs.import_onnx(model)

tensors = graph.tensors()
## 1.generate a subgraph
graph.inputs = [tensors["input_1:01_permuted"].to_variable(dtype=np.float32,shape=(1,3,224,224))]
graph.outputs = [tensors["act_softmax"].to_variable(dtype=np.float32)]
## 2.Change the names of input and output
graph.inputs[0].name = 'input'
graph.outputs[0].name = 'output'

# Notice that we do not need to manually modify the rest of the graph. ONNX GraphSurgeon will
# take care of removing any unnecessary nodes or tensors, so that we are left with only the subgraph.
graph.cleanup()
onnx.save(gs.export_onnx(graph), "/home/devdata/xiaoying.zhang/DORADO/common/mobilenet_v1_kerassubgraph-op13-fp32.onnx")