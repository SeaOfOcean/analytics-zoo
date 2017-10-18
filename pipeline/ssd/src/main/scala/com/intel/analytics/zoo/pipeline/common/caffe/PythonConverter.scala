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

package com.intel.analytics.zoo.pipeline.common.caffe

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.common.nn.Proposal
import pipeline.caffe.Caffe.LayerParameter


class PythonConverter(implicit ev: TensorNumeric[Float]) extends Customizable[Float] {
  override def convertor(layer: GeneratedMessage): Seq[ModuleNode[Float]] = {
    val param = layer.asInstanceOf[LayerParameter].getPythonParam
    val layerName = param.getLayer
    layerName match {
      case "ProposalLayer" =>
        // for faster rcnn
        Seq(Proposal(preNmsTopN = 6000,
          postNmsTopN = 300, Array[Float](0.5f, 1.0f, 2.0f), Array[Float](8, 16, 32))
          .setName(getLayerName(layer)).inputs())
      case "AnchorTargetLayer" =>
        null
      case "ProposalTargetLayer" =>
        null
    }
  }

}
