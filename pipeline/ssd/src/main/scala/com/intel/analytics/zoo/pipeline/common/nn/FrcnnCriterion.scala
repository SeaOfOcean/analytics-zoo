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

package com.intel.analytics.zoo.pipeline.common.nn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.nn.{ParallelCriterion, SoftmaxWithCriterion}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.pipeline.common.nn.FrcnnCriterion._
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.FasterRcnn
import org.apache.log4j.Logger

object FrcnnCriterion {
  val logger = Logger.getLogger(getClass)

  def apply(rpnSigma: Float = 3, frcnnSigma: Float = 1,
    ignoreLabel: Option[Int] = Some(-1), rpnLossClsWeight: Float = 1,
    rpnLossBboxWeight: Float = 1,
    lossClsWeight: Float = 1,
    lossBboxWeight: Float = 1)(implicit ev: TensorNumeric[Float]): FrcnnCriterion =
    new FrcnnCriterion(rpnSigma, frcnnSigma, ignoreLabel, rpnLossClsWeight, rpnLossBboxWeight,
      lossClsWeight)

}

class FrcnnCriterion(rpnSigma: Float = 3, frcnnSigma: Float = 1,
  ignoreLabel: Option[Int] = Some(-1), rpnLossClsWeight: Float = 1,
  rpnLossBboxWeight: Float = 1,
  lossClsWeight: Float = 1,
  lossBboxWeight: Float = 1)(implicit ev: TensorNumeric[Float])
  extends AbstractCriterion[Table, Tensor[Float], Float] {

  val rpn_loss_bbox = SmoothL1CriterionWithWeights2(rpnSigma)
  val rpn_loss_cls = SoftmaxWithCriterion(ignoreLabel = ignoreLabel)
  val loss_bbox = SmoothL1CriterionWithWeights2(frcnnSigma)
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
    val cls_prob = input[Tensor[Float]](FasterRcnn.clsProbIndex)
    val bbox_pred = input[Tensor[Float]](FasterRcnn.bboxPredIndex)
    val roiData = input[Table](FasterRcnn.roiDataIndex)
    val proposalTarget = roiData[Tensor[Float]](2)
    proposalTarget.apply1(x => {
      require(x >= 1, "proposal target is 1-based")
      x
    })
    val rpn_cls_score_reshape = input[Tensor[Float]](FasterRcnn.rpnClsScoreIndex)
    val rpn_bbox_pred = input[Tensor[Float]](FasterRcnn.rpnBboxPredIndex)
    val rpn_data = input[Table](FasterRcnn.rpnDataIndex)
    val rpnTarget = rpn_data[Tensor[Float]](1)
    rpnTarget.apply1(x => {
      require(x >= 1 || x == -1, "rpn target is 1-based, or -1 for ignored label")
      x
    })
    data.insert(1, cls_prob)
    label.insert(1, roiData(2))
    data.insert(2, bbox_pred)
    label.insert(2, getSubTable(roiData, proposalBbox, 3, 3))
    data.insert(3, rpn_cls_score_reshape)
    label.insert(3, rpn_data(1))
    data.insert(4, rpn_bbox_pred)
    label.insert(4, getSubTable(rpn_data, anchorBbox, 2, 3))
    output = criterion.updateOutput(data, label)
    logger.info(s"Train net output #0: loss_bbox = ${criterion.outputs[Float](2)}")
    logger.info(s"Train net output #1: loss_cls = ${criterion.outputs[Float](1)}")
    logger.info(s"Train net output #2: rpn_cls_loss = ${criterion.outputs[Float](3)}")
    logger.info(s"Train net output #3: rpn_loss_bbox = ${criterion.outputs[Float](4)}")
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

  @transient var roiDataGrad: Table = _
  @transient var imInfoGrad: Tensor[Float] = _
  @transient var rpnDataGrad: Table = _

  override def updateGradInput(input: Table, target: Tensor[Float]): Table = {
    if (null == roiDataGrad) {
      roiDataGrad = T(Tensor(), Tensor(), Tensor(), Tensor(), Tensor())
      imInfoGrad = Tensor()
      rpnDataGrad = T(Tensor(), Tensor(), Tensor(), Tensor())
      gradInput.insert(FasterRcnn.roiDataIndex, roiDataGrad)
      gradInput.insert(FasterRcnn.imInfoIndex, imInfoGrad)
      gradInput.insert(FasterRcnn.rpnDataIndex, roiDataGrad)
    }

    criterion.updateGradInput(data, label)
    gradInput.insert(FasterRcnn.clsProbIndex, criterion.gradInput(1))
    println("clsProbIndex: "
      + input[Tensor[Float]](FasterRcnn.clsProbIndex).size().mkString("x")
      + " " + gradInput[Tensor[Float]](FasterRcnn.clsProbIndex).size().mkString("x"))
    gradInput.insert(FasterRcnn.bboxPredIndex, criterion.gradInput(2))
    println("bboxPredIndex: "
      + input[Tensor[Float]](FasterRcnn.bboxPredIndex).size().mkString("x")
      + " " + gradInput[Tensor[Float]](FasterRcnn.bboxPredIndex).size().mkString("x"))
    gradInput.insert(FasterRcnn.rpnClsScoreIndex, criterion.gradInput(3))
    println("rpnClsScoreIndex: "
      + input[Tensor[Float]](FasterRcnn.rpnClsScoreIndex).size().mkString("x")
      + " " + gradInput[Tensor[Float]](FasterRcnn.rpnClsScoreIndex).size().mkString("x"))
    gradInput.insert(FasterRcnn.rpnBboxPredIndex, criterion.gradInput(4))
    println("rpnBboxPredIndex: "
      + input[Tensor[Float]](FasterRcnn.rpnBboxPredIndex).size().mkString("x")
      + " " + gradInput[Tensor[Float]](FasterRcnn.rpnBboxPredIndex).size().mkString("x"))
    gradInput
  }
}

