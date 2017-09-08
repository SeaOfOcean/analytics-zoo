/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.intel.analytics.bigdl.pipeline.ssd

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.common.IOUtils
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.{RecordToFeature, RoiImageToBatch, SSDMiniBatch}
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, MatToFloats, RandomTransformer}
import com.intel.analytics.zoo.transform.vision.image.augmentation.{ColorJitter, Expand, HFlip, Resize}
import com.intel.analytics.zoo.transform.vision.label.roi.{RandomSampler, RoiExpand, RoiHFlip, RoiNormalize}
import org.apache.spark.SparkContext


object Utils {

  def loadTrainSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int)
  : DataSet[SSDMiniBatch] = {
    val trainRdd = IOUtils.loadSeqFiles(Engine.nodeNumber, folder, sc)._1
    DataSet.rdd(trainRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      RoiNormalize() ->
      ColorJitter() ->
      RandomTransformer(Expand() -> RoiExpand(), 0.5) ->
      RandomSampler() ->
      Resize(resolution, resolution, -1) ->
      RandomTransformer(HFlip() -> RoiHFlip(), 0.5) ->
      MatToFloats(validHeight = resolution, validWidth = resolution,
        meanRGB = Some(123f, 117f, 104f)) ->
      RoiImageToBatch(batchSize)
  }

  def loadValSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int)
  : DataSet[SSDMiniBatch] = {
    val valRdd = IOUtils.loadSeqFiles(Engine.nodeNumber, folder, sc)._1

    DataSet.rdd(valRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      RoiNormalize() ->
      Resize(resolution, resolution) ->
      MatToFloats(validHeight = resolution, validWidth = resolution,
        meanRGB = Some(123f, 117f, 104f)) ->
      RoiImageToBatch(batchSize)
  }
}
