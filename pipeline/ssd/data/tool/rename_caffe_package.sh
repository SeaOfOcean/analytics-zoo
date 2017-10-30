#!/bin/bash

# protoc --proto_path=src/caffe/proto --java_out . src/caffe/proto/caffe.proto
sed -i 's/package caffe/package pipeline.caffe/g' src/main/java/pipeline/caffe/Caffe.java
sed -i 's/caffe.Caffe/pipeline.caffe.Caffe/g' src/main/java/pipeline/caffe/Caffe.java