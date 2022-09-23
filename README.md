本项目从dlpack的numpy [demo](https://github.com/dmlc/dlpack/tree/857f7dec9b6a2cc4a39c10bfa81d0315ef0f30d3/apps/numpy_dlpack)上进行了修改，目的是通过mindspore的`Tensor.asnumpy()`和`Tensor.from_numpy`实现DLPack协议，主要是演示用途。

通过DLPack协议，我们可以实现tensor在不同框架之间的数据转换。例如在julia中，通过[DLPack.jl](https://github.com/pabloferz/DLPack.jl)
由于是通过numpy支持的，应该只支持CPU？

```julia
using DLPack
using PythonCall

ms = pyimport("mindspore")
dl = pyimport("msdlpack")

v = rand(3, 2)
pyv = DLPack.share(v, dl.from_dlpack)

Bool(pyv.shape == pyconvert(Py,(2, 3)))  # the dimensions are reversed.

# v and pyv share same data
v[1, 1] = 0.0
(pyv[0, 0] == 0.0) # Tensor(True)

# change tensor could cause an unexpected error in julia Array
# see https://gitee.com/mindspore/mindspore/issues/I5SLJX

```

```julia
using DLPack
using PythonCall

ms = pyimport("mindspore")
dl = pyimport("msdlpack")

pyv = ms.numpy.arange(1, 5).reshape(2, 2)
v = DLPack.wrap(pyv, dl.to_dlpack)

Bool(v[2, 1] == 2 == pyv[0, 1])  # the dimensions are reversed

```

TODO: 在julia退出时发生了Segmentation fault，有些gc的地方还没有处理好