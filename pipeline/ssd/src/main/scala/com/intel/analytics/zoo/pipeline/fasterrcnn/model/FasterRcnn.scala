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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.pipeline.common.ModuleUtil
import com.intel.analytics.zoo.pipeline.common.nn.Proposal
import com.intel.analytics.zoo.pipeline.fasterrcnn.AnchorParam

object FasterRcnn {

  /**
   *
   * @param p parameter: (nIn: Int, nOut: Int, ker: Int, stride: Int, pad: Int)
   * @param name name of layer
   * @return
   */
  def conv(p: (Int, Int, Int, Int, Int),
    name: String, withBias: Boolean = true): SpatialConvolution[Float] = {
    SpatialConvolution(p._1, p._2, p._3, p._3, p._4, p._4, p._5, p._5,
      withBias = withBias).setName(name)
  }


  def apply(nClass: Int, rpnPreNmsTopN: Int, rpnPostNmsTopN: Int, anchorParam: AnchorParam,
    rpn: Sequential[Float], fastRcnn: Sequential[Float]): Module[Float] = {
    val model = Sequential()
    val model1 = ParallelTable()
    model1.add(rpn)
    model1.add(new Identity())
    model.add(model1)
    // connect rpn and fast-rcnn
    val middle = ConcatTable()
    val left = Sequential()
    val left1 = ConcatTable()
    left1.add(selectTensor(1, 1, 1))
    left1.add(selectTensor(1, 1, 2))
    left1.add(selectTensor1(2))
    left.add(left1)
    left.add(Proposal(preNmsTopN = rpnPreNmsTopN,
      postNmsTopN = rpnPostNmsTopN, anchorParam = anchorParam))
    left.add(selectTensor1(1))
    // first add feature from feature net
    middle.add(selectTensor(1, 2))
    // then add rois from proposal
    middle.add(left)
    model.add(middle)
    // get the fast rcnn results and rois
    model.add(ConcatTable().add(fastRcnn).add(selectTensor(2)))
    ModuleUtil.shareMemory(model)
    model
  }


  /**
   * select tensor from nested tables
   * @param depths a serious of depth to use when fetching certain tensor
   * @return a wanted tensor
   */
  protected def selectTensor(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]()
    var i = 0
    while (i < depths.length - 1) {
      module.add(new SelectTable(depths(i)))
      i += 1
    }
    module.add(new SelectTable(depths(depths.length - 1)))
  }

  protected def selectTensor1(depth: Int): SelectTable[Float] = {
    new SelectTable(depth)
  }

}
