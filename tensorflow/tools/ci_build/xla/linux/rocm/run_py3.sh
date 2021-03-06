#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

set -e
set -x

N_JOBS=$(grep -c ^processor /proc/cpuinfo)
N_GPUS=$(lspci|grep 'controller'|grep 'AMD/ATI'|wc -l)

echo ""
echo "Bazel will use ${N_JOBS} concurrent build job(s) and ${N_GPUS} concurrent test job(s)."
echo ""

# Run configure.
export PYTHON_BIN_PATH=`which python3`
export CC_OPT_FLAGS='-mavx'

export TF_NEED_ROCM=1
export ROCM_PATH=/opt/rocm-3.9.0
export TF_GPU_COUNT=${N_GPUS}

yes "" | $PYTHON_BIN_PATH configure.py
echo "build --distinct_host_configuration=false" >> .tf_configure.bazelrc

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test \
      --config=rocm \
      --config=xla \
      -k \
      --test_tag_filters=-no_oss,-oss_serial,-no_gpu,-no_rocm,-benchmark-test,-rocm_multi_gpu,-v1only \
      --jobs=${N_JOBS} \
      --local_test_jobs=${TF_GPU_COUNT} \
      --test_timeout 600,900,2400,7200 \
      --build_tests_only \
      --test_output=errors \
      --test_sharding_strategy=disabled \
      --test_size_filters=small,medium \
      --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
      -- \
      //tensorflow/compiler/... \
      -//tensorflow/compiler/tests:dense_layer_test \
      -//tensorflow/compiler/tests:dense_layer_test_gpu \
      -//tensorflow/compiler/tests:jit_test \
      -//tensorflow/compiler/tests:jit_test_gpu \
      -//tensorflow/compiler/tests:matrix_triangular_solve_op_test \
      -//tensorflow/compiler/tests:tensor_array_ops_test \
      -//tensorflow/compiler/tests:xla_ops_test \
      -//tensorflow/compiler/xla/client/lib:svd_test \
      -//tensorflow/compiler/tests:lstm_test \
&& bazel test \
      --config=rocm \
      --config=xla \
      -k \
      --test_tag_filters=-no_oss,-oss_serial,-no_gpu,-no_rocm,-benchmark-test,-rocm_multi_gpu,-v1only \
      --jobs=${N_JOBS} \
      --local_test_jobs=${TF_GPU_COUNT} \
      --test_timeout 600,900,2400,7200 \
      --build_tests_only \
      --test_output=errors \
      --test_sharding_strategy=disabled \
      --test_env=TF2_BEHAVIOR=0 \
      --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
      -- \
      //tensorflow/compiler/tests:dense_layer_test \
      //tensorflow/compiler/tests:dense_layer_test_gpu \
      //tensorflow/compiler/tests:jit_test \
      //tensorflow/compiler/tests:jit_test_gpu \
      //tensorflow/compiler/tests:matrix_triangular_solve_op_test \
      //tensorflow/compiler/tests:tensor_array_ops_test \
      //tensorflow/compiler/tests:xla_ops_test \
      //tensorflow/compiler/xla/client/lib:svd_test \
      //tensorflow/compiler/tests:lstm_test
