/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.fasterrcnn

/**
 * Preprocess parameters
 * @param batchSize should be 1
 * @param scales Each scale is the pixel size of an image"s shortest side, can contain multiple
 * @param scaleMultipleOf Resize test images so that its width and height are multiples of ...
 * @param pixelMeanRGB mean value to be sub from
 * @param hasLabel whether data contains label, default is false
 */
case class PreProcessParam(batchSize: Int = 1,
  scales: Array[Int] = Array(600), scaleMultipleOf: Int = 1,
  pixelMeanRGB: (Float, Float, Float) = (122.7717f, 115.9465f, 102.9801f),
  hasLabel: Boolean = false
)

/**
 * post process parameters
 * @param nmsThresh Overlap threshold used for non-maximum suppression (suppress boxes with
 * IoU >= this threshold)
 * @param nClasses number of classes
 * @param bboxVote whether apply bounding box voting
 * @param maxPerImage
 * @param thresh
 */
case class PostProcessParam(nmsThresh: Float = 0.3f, nClasses: Int,
  bboxVote: Boolean, maxPerImage: Int = 100, thresh: Double = 0.05)