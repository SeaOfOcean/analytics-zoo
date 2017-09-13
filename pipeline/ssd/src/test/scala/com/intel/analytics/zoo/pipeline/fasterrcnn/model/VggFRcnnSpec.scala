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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class VggFRcnnSpec extends FlatSpec with Matchers {
  "faster rcnn graph" should "forward properly" in {
    val frcnnGraph = VggFRcnn(21)
    val frcnn = VggFRcnnSeq(21)
    frcnnGraph.loadModelWeights(frcnn)

    val input = T(Tensor[Float](1, 3, 300, 300).randn(),
      Tensor[Float](T(300f, 300f, 1f, 1f)).resize(1, 4))
    frcnnGraph.forward(input)
    frcnn.forward(input)

    frcnnGraph.output.toTable.length() should equal(frcnn.output.toTable.length())
  }
}
