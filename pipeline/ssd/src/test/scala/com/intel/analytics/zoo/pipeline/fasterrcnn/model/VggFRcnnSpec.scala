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

package com.intel.analytics.zoo.pipeline.fasterrcnn.model

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.scalatest.{FlatSpec, Matchers}

class VggFRcnnSpec extends FlatSpec with Matchers {
  "faster rcnn graph" should "forward properly" in {
    val frcnnGraph = VggFRcnn(21).evaluate()
    val frcnn = VggFRcnnSeq(21).evaluate()
    frcnnGraph.loadModelWeights(frcnn)

    val data = Tensor[Float](1, 3, 300, 300).randn()
    val rois = Tensor[Float](T(300f, 300f, 1f, 1f)).resize(1, 4)
    val input = T(data, rois, null)
    val input2 = T(data, rois)
    frcnnGraph.forward(input)
    frcnn.forward(input2)

//    frcnnGraph.output.toTable.length() should equal(frcnn.output.toTable.length())

    val namedModule = Utils.getNamedModules(frcnnGraph)
    val namedModule2 = Utils.getNamedModules(frcnn)
    namedModule.keys.foreach(key => {
      if (namedModule.contains(key) && namedModule2.contains(key)) {
        val out1 = namedModule(key).output
        val out2 = namedModule2(key).output
        if (out1.isTensor) {
          if (pass(out1.toTensor, out2.toTensor)) {
//            println(s"${key} pass")
          } else {
            println(s"${key} not pass")
          }
        }
      }
    })
  }

  def pass(out1: Tensor[Float], out2: Tensor[Float]): Boolean = {
    out1.toTensor[Float].map(out2.toTensor[Float], (a, b) => {
      if (Math.abs(a - b) > 1e-6) {
        return false
      }
      a
    })
    true
  }
}
