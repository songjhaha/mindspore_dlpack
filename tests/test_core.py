import numpy as np
import mindspore as ms
from msdlpack import to_dlpack, from_dlpack


def test_to_from_tensor_zero_copy():
    """Test the copy free conversion of tensor via DLPack."""
    np_ary = ms.Tensor(np.random.normal(size=[10, 10]))
    np_ary_big = ms.Tensor(np.random.normal(size=[12, 10]))
    dlpack_capsule = to_dlpack(np_ary_big)
    reconstructed_ary = from_dlpack(dlpack_capsule)
    del dlpack_capsule
    np_ary_big[1:11] = np_ary
    del np_ary_big
    # this test would fail, see https://gitee.com/mindspore/mindspore/issues/I5SLJX
    np.testing.assert_equal(actual=reconstructed_ary.asnumpy()[1:11], desired=np_ary.asnumpy())


def test_to_from_tensor_memory():
    """Test that DLPack capsule keeps the source array alive"""
    source_array = ms.Tensor(np.random.normal(size=[10, 10]))
    np_array_ref = source_array.copy()
    dlpack_capsule = to_dlpack(source_array)
    del source_array
    reconstructed_array = from_dlpack(dlpack_capsule)
    del dlpack_capsule
    np.testing.assert_equal(actual=reconstructed_array.asnumpy(), desired=np_array_ref.asnumpy())


if __name__ == "__main__":
    """
    Run both tests a bunch of times to make
    sure the conversions and memory management are stable.
    """
    print("### Running `test_to_from_tensor_zero_copy`")
    for i in range(10000):
        test_to_from_tensor_zero_copy()
    print("### Running `test_to_from_tensor_memory`")
    for i in range(10000):
        test_to_from_tensor_memory()
