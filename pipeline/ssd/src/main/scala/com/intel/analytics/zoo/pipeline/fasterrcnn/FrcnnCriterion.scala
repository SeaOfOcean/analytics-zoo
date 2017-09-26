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

package com.intel.analytics.zoo.pipeline.fasterrcnn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.nn.{ParallelCriterion, SmoothL1CriterionWithWeights, SoftmaxWithCriterion}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}


class FrcnnCriterion(rpnSigma: Float = 3, frcnnSigma: Float = 1,
  ignoreLabel: Option[Int] = Some(-1), rpnLossClsWeight: Float = 1,
  rpnLossBboxWeight: Float = 1,
  lossClsWeight: Float = 1,
  lossBboxWeight: Float = 1)(implicit ev: TensorNumeric[Float])
  extends AbstractCriterion[Table, Tensor[Float], Float] {

  val rpn_loss_bbox = SmoothL1CriterionWithWeights(rpnSigma)
  val rpn_loss_cls = SoftmaxWithCriterion(ignoreLabel = ignoreLabel)
  val loss_bbox = SmoothL1CriterionWithWeights(frcnnSigma)
  val loss_cls = SoftmaxWithCriterion()

  val data = T()
  val label = T()
  val criterion = ParallelCriterion()
  criterion.add(loss_cls, lossClsWeight)
  criterion.add(loss_bbox, lossBboxWeight)
  criterion.add(rpn_loss_cls, rpnLossClsWeight)
  criterion.add(rpn_loss_bbox, rpnLossBboxWeight)


  private val anchorBbox = T()
  private val proposalBbox = T()

  override def updateOutput(input: Table, target: Tensor[Float]): Float = {
    val cls_prob = input[Tensor[Float]](1)
    val bbox_pred = input[Tensor[Float]](2)
    val roi_data = input[Table](5)
    val rpn_cls_score_reshape = input[Tensor[Float]](3)
    val rpn_bbox_pred = input[Tensor[Float]](4)
    val rpn_data = input[Table](6)
    data.insert(1, cls_prob)
    label.insert(1, roi_data(2))
    data.insert(2, bbox_pred)
    label.insert(2, getSubTable(roi_data, proposalBbox, 3, 3))
    data.insert(3, rpn_cls_score_reshape)
    label.insert(3, rpn_data(1))
    data.insert(4, rpn_bbox_pred)
    label.insert(4, getSubTable(rpn_data, anchorBbox, 2, 3))
    output = criterion.updateOutput(data, label)
    output
  }

  private def getSubTable(src: Table, target: Table, startInd: Int, len: Int): Table = {
    var i = 1
    (startInd until startInd + len).foreach(j => {
      target.insert(i, src(j))
      i += 1
    })
    target
  }

  override def updateGradInput(input: Table, target: Tensor[Float]): Table = {
    gradInput = criterion.updateGradInput(data, label)
    gradInput.insert(5, T(Tensor(), Tensor(), Tensor(), Tensor(), Tensor()))
    gradInput.insert(6, T(Tensor(), Tensor(), Tensor(), Tensor()))
    gradInput
  }
}

