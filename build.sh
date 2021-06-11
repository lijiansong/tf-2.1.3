#!/bin/bash
set -e
bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip uninstall tensorflow
# for python 3.5 the pkg will be tensorflow-2.1.3-cp35-cp35m-linux_x86_64.whl
#pip install /tmp/tensorflow_pkg/tensorflow-2.1.3-cp35-cp35m-linux_x86_64.whl
# for python 3.6 the pkg will be tensorflow-2.1.3-cp36-cp36m-linux_x86_64.whl
pip install /tmp/tensorflow_pkg/tensorflow-2.1.3-cp36-cp36m-linux_x86_64.whl
# for python 3.9 the pkg will be tensorflow-2.1.3-cp39-cp39m-linux_x86_64.whl
#pip install /tmp/tensorflow_pkg/tensorflow-2.1.3-cp39-cp39-linux_x86_64.whl
