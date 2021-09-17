export CUDA_VISIBLE_DEVICES=7

tfmodel=keras_mobilenetv1.pb
onnxmodel=mobilenet_v1_keras-op13-fp32.onnx
python -m tf2onnx.convert --input $tfmodel --output $onnxmodel \
    --fold_const --opset 13 --verbose \
    --inputs-as-nchw input:0[1,3,224,224] \
    --inputs input:0[1,3,224,224] \
    --outputs MobilenetV1/Predictions/Reshape_1:0
    # --outputs MobilenetV1/Logits/SpatialSqueeze:0
    
