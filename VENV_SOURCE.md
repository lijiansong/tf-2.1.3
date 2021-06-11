# Virtual Env Source on the 180 Server

ICT 10.3.2.180 server venv backup list for ljs:

1. Apache Singa

```
$ cd ~/work-space
$ source anaconda2/bin/activate
```
Currently, Singa works on the `base` env.

2. TF-LMS

```
$ cd ~/work-space
$ source anaconda2/bin/activate
$ conda env list
```

then we have:

```
base                  *  /home/ljs/work-space/anaconda2
incubator-singa          /home/ljs/work-space/anaconda2/envs/incubator-singa
intel-caffe              /home/ljs/work-space/anaconda2/envs/intel-caffe
mega                     /home/ljs/work-space/anaconda2/envs/mega
mega-dtr                 /home/ljs/work-space/anaconda2/envs/mega-dtr
tensorflow-lms-v2        /home/ljs/work-space/anaconda2/envs/tensorflow-lms-v2
tf-lms                   /home/ljs/work-space/anaconda2/envs/tf-lms
tf2-source               /home/ljs/work-space/anaconda2/envs/tf2-source
wmlce_env-tf-lms         /home/ljs/work-space/anaconda2/envs/wmlce_env-tf-lms
```

TF-LMS works on the env `wmlce_env-tf-lms`,

```
$ conda activate wmlce_env-tf-lms
$ cd /home/ljs/work-space/memory/tensorflow-large-model-support/examples
$ ./run.sh
$ conda deactivate
```

3. MegEngine DTR

```
$ cd ~/work-space
$ source anaconda2/bin/activate
$ conda env list
```

then we have:

```
base                  *  /home/ljs/work-space/anaconda2
incubator-singa          /home/ljs/work-space/anaconda2/envs/incubator-singa
intel-caffe              /home/ljs/work-space/anaconda2/envs/intel-caffe
mega                     /home/ljs/work-space/anaconda2/envs/mega
mega-dtr                 /home/ljs/work-space/anaconda2/envs/mega-dtr
tensorflow-lms-v2        /home/ljs/work-space/anaconda2/envs/tensorflow-lms-v2
tf-lms                   /home/ljs/work-space/anaconda2/envs/tf-lms
tf2-source               /home/ljs/work-space/anaconda2/envs/tf2-source
wmlce_env-tf-lms         /home/ljs/work-space/anaconda2/envs/wmlce_env-tf-lms
```

MegEngine DTR works on the env `mega-dtr`,

```
$ conda activate mega-dtr
$ cd /home/ljs/work-space/memory/baselines/megengine/bert
$ ./dtr-run.sh
$ conda deactivate
```

4. tf-2.1.3

```
$ cd ~/work-space
$ source anaconda2/bin/activate
$ conda env list
```

then we have:

```
base                  *  /home/ljs/work-space/anaconda2
incubator-singa          /home/ljs/work-space/anaconda2/envs/incubator-singa
intel-caffe              /home/ljs/work-space/anaconda2/envs/intel-caffe
mega                     /home/ljs/work-space/anaconda2/envs/mega
mega-dtr                 /home/ljs/work-space/anaconda2/envs/mega-dtr
tensorflow-lms-v2        /home/ljs/work-space/anaconda2/envs/tensorflow-lms-v2
tf-lms                   /home/ljs/work-space/anaconda2/envs/tf-lms
tf2-source               /home/ljs/work-space/anaconda2/envs/tf2-source
wmlce_env-tf-lms         /home/ljs/work-space/anaconda2/envs/wmlce_env-tf-lms
```

tf-2.1.3 works on the env `tf2-source`,

```
$ conda activate tf2-source
$ cd /home/ljs/work-space/memory/tensorflow-2.1.3/ml-examples/dl/bert
$ ./py-run.sh
$ conda deactivate
```

