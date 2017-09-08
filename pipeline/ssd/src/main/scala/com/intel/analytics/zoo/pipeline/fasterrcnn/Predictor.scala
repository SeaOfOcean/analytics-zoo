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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.{RecordToFeature, RoiImageToBatch, SSDByteRecord, SSDMiniBatch}
import com.intel.analytics.zoo.transform.vision.image.augmentation.RandomResize
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, MatToFloats}
import com.intel.analytics.zoo.transform.vision.label.roi.RoiLabel
import org.apache.spark.rdd.RDD

class Predictor(
  model: Module[Float],
  preProcessParam: PreProcessParam,
  postProcessParam: PostProcessParam) {

  val preProcessor = RecordToFeature(true) ->
    BytesToMat() ->
    RandomResize(preProcessParam.scales, preProcessParam.scaleMultipleOf) ->
    MatToFloats(validHeight = 100, 100, meanRGB = Some(preProcessParam.pixelMeanRGB)) ->
    RoiImageToBatch(preProcessParam.batchSize, true, Some(preProcessParam.nPartition))

  val postProcessor = new Postprocessor(postProcessParam)

  def predict(rdd: RDD[SSDByteRecord]): RDD[Array[RoiLabel]] = {
    Predictor.predict(rdd, model, preProcessor, postProcessor)
  }
}

object Predictor {
  def predict(rdd: RDD[SSDByteRecord],
    model: Module[Float],
    preProcessor: Transformer[SSDByteRecord, SSDMiniBatch],
    postProcessor: Postprocessor
  ): RDD[Array[RoiLabel]] = {
    model.evaluate()
    val broadcastModel = ModelBroadcast().broadcast(rdd.sparkContext, model)
    val broadpostprocessor = rdd.sparkContext.broadcast(postProcessor)
    rdd.mapPartitions(preProcessor(_)).mapPartitions(dataIter => {
      val localModel = broadcastModel.value()
      val localPostProcessor = broadpostprocessor.value.clone()
      dataIter.map(batch => {
        val data = T()
        data.insert(batch.input)
        data.insert(batch.imInfo)
        val result = localModel.forward(data).toTable
        localPostProcessor.process(result, batch.imInfo)
      })
    })
  }
}
