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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{File, Table}
import org.scalatest.{FlatSpec, Matchers}

class SmoothL1CriterionWithWeights2Spec extends FlatSpec with Matchers {
  "smoothl1 forward" should "work" in {
    val criterion = SmoothL1CriterionWithWeights2[Float](3)
    val input = File.load[Tensor[Float]]("/tmp/input_3104580143564290.bin")
    val target = File.load[Table]("/tmp/target_3104580143564290.bin")
    println(criterion.forward(input, target))
    println(criterion.output)
  }
}
