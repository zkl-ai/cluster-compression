#!/bin/bash

image="hampenv"
tag="nx1.0"
if [ -n "$1" ];then tag="v1";fi

curpath=`pwd`
exist=`docker ps -a |grep -v hamp-test |grep hamp|awk '{print $1}'`


if [ -z "${exist}" ];then
	docker run --runtime=nvidia -itd --name=hamp \
		-v /usr/bin/tegrastats:/usr/bin/tegrastats \
		-v /run/jtop.sock:/run/jtop.sock \
		-v /data:/data:rw \
		--network=host -w /workspace ${image}:${tag} \
		"echo 'Cloning cluster-compression...'; \
				echo 'installing dependencies...'; \
				pip install tqdm einops termcolor --index-url https://mirrors.sustech.edu.cn/pypi/web/simple; bash"
				# echo 'Running tp_worker_imagenet.py...'; \
				# python perf_measure.py --model vgg16 --data-path vgg16-5000-10.pkl --batch-size 16 --pretrained"
else
	echo "another hamp container exists, run failed."
	exit 1
fi

exist=`docker ps |grep -v hamp-test |grep hamp|awk '{print $1}'`

if [ -n "${exist}" ];then
	echo "worker start."
else
	echo "hamp doesn't exist. please start it first."
	exit 1
fi

