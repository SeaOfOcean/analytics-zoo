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

package com.intel.analytics.zoo.pipeline.fasterrcnn.model
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.{apply => _, _}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.pipeline.common.nn.Proposal
import com.intel.analytics.zoo.pipeline.fasterrcnn.{AnchorParam, AnchorTarget, Postprocessor, ProposalTarget, RoiPooling => RoiPoolingFrcnn}
import com.intel.analytics.zoo.pipeline.ssd.model.SSDGraph.{apply => _}
object VggFRcnn {

  private[pipeline] def addConvRelu(prevNodes: ModuleNode[Float], p: (Int, Int, Int, Int, Int),
    name: String, prefix: String = "conv", nGroup: Int = 1, propogateBack: Boolean = true)
  : ModuleNode[Float] = {
    val conv = SpatialConvolution(p._1, p._2, p._3, p._3, p._4, p._4,
      p._5, p._5, nGroup = nGroup, propagateBack = propogateBack)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"$prefix$name").inputs(prevNodes)
    ReLU(true).setName(s"relu$name").inputs(conv)
  }


  def vgg16(data: ModuleNode[Float]): ModuleNode[Float] = {
    val conv1_1 = SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, propagateBack = false)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"conv1_1").inputs(data)
    val relu1_1 = ReLU(true).setName(s"relu1_1").inputs(conv1_1)
    val relu1_2 = addConvRelu(relu1_1, (64, 64, 3, 1, 1), "1_2")
    val pool1 = SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool1").inputs(relu1_2)

    val relu2_1 = addConvRelu(pool1, (64, 128, 3, 1, 1), "2_1")
    val relu2_2 = addConvRelu(relu2_1, (128, 128, 3, 1, 1), "2_2")
    val pool2 = SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool2").inputs(relu2_2)

    val relu3_1 = addConvRelu(pool2, (128, 256, 3, 1, 1), "3_1")
    val relu3_2 = addConvRelu(relu3_1, (256, 256, 3, 1, 1), "3_2")
    val relu3_3 = addConvRelu(relu3_2, (256, 256, 3, 1, 1), "3_3")
    val pool3 = SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool3").inputs(relu3_3)

    val relu4_1 = addConvRelu(pool3, (256, 512, 3, 1, 1), "4_1")
    val relu4_2 = addConvRelu(relu4_1, (512, 512, 3, 1, 1), "4_2")
    val relu4_3 = addConvRelu(relu4_2, (512, 512, 3, 1, 1), "4_3")

    val pool4 = SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool4").inputs(relu4_3)
    val relu5_1 = addConvRelu(pool4, (512, 512, 3, 1, 1), "5_1")
    val relu5_2 = addConvRelu(relu5_1, (512, 512, 3, 1, 1), "5_2")
    val relu5_3 = addConvRelu(relu5_2, (512, 512, 3, 1, 1), "5_3")
    relu5_3
  }


  val anchorParam = AnchorParam(_scales = Array(8f, 16f, 32f), _ratios = Array(0.5f, 1.0f, 2.0f))
  val rpnPreNmsTopN = 6000
  val rpnPostNmsTopN = 300

  def apply(nClass: Int, param: PostProcessParam): Module[Float] = {
    val data = Input()
    val imInfo = Input()
    // for training only
    val gt = Input()
    val vgg = vgg16(data)
    // val rpnNet = rpn(vgg, imInfo)
    val rpn_conv_3x3 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
      .setName("rpn_conv/3x3").inputs(vgg)
    val relu3x3 = ReLU(true).setName("rpn_relu/3x3").inputs(rpn_conv_3x3)
    val rpn_cls_score = SpatialConvolution(512, 18, 1, 1, 1, 1)
      .setName("rpn_cls_score").inputs(relu3x3)
    val rpn_cls_score_reshape = InferReshape(Array(0, 2, -1, 0)).inputs(rpn_cls_score)
    val rpn_cls_prob = SoftMax().setName("rpn_cls_prob").inputs(rpn_cls_score_reshape)
    val rpn_cls_prob_reshape = InferReshape(Array(1, 2 * anchorParam.num, -1, 0))
      .setName("rpn_cls_prob_reshape").inputs(rpn_cls_prob)
    val rpn_bbox_pred = SpatialConvolution(512, 36, 1, 1, 1, 1).setName("rpn_bbox_pred")
      .inputs(relu3x3)
    val proposal = Proposal(preNmsTopN = rpnPreNmsTopN,
      postNmsTopN = rpnPostNmsTopN, anchorParam = anchorParam).setName("proposal")
      .inputs(rpn_cls_prob_reshape, rpn_bbox_pred, imInfo)


    val roi_data = ProposalTarget(VggParam(), nClass).setName("roi-data")
      .inputs(proposal, gt)
    val roi = SelectTable(1).setName("roi").inputs(roi_data)
    // val (clsProb, bboxPred) = fastRcnn(vgg, rpnNet)
    val pool = 7
    val roiPooling = RoiPoolingFrcnn(pool, pool, 0.0625f).setName("pool5").inputs(vgg, roi)
    val reshape = InferReshape(Array(-1, 512 * pool * pool)).inputs(roiPooling)
    val fc6 = Linear(512 * pool * pool, 4096).setName("fc6").inputs(reshape)
    val reLU6 = ReLU().inputs(fc6)
    val dropout6 = Dropout().setName("drop6").inputs(reLU6)
    val fc7 = Linear(4096, 4096).setName("fc7").inputs(dropout6)
    val reLU7 = ReLU().inputs(fc7)
    val dropout7 = Dropout().setName("drop7").inputs(reLU7)
    val cls_score = Linear(4096, 21).setName("cls_score").inputs(dropout7)
    val cls_prob = SoftMax().setName("cls_prob").inputs(cls_score)
    val bbox_pred = Linear(4096, 84).setName("bbox_pred").inputs(dropout7)

    // Training part
    val rpn_data = AnchorTarget(VggParam()).setName("rpn-data")
      .inputs(rpn_cls_score, gt, imInfo, data)

    val detectionOut = Postprocessor(param).inputs(cls_prob, bbox_pred, roi_data,
      rpn_cls_score_reshape, rpn_bbox_pred, rpn_data, imInfo)
    val model = Graph(Array(data, imInfo, gt), detectionOut)
    model.stopGradient(Array("rpn-data", "roi-data", "proposal", "roi", "relu2_2"))
  }
}
