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
import com.intel.analytics.bigdl.utils.{File, T, Table}
import com.intel.analytics.zoo.pipeline.ssd.TestUtil
import org.scalatest.{FlatSpec, Matchers}

class SmoothL1CriterionWithWeights2Spec extends FlatSpec with Matchers {
  "smoothl1 forward" should "work" in {
    TestUtil.middleRoot = "/home/jxy/data/middle/vgg16/new"
    val criterion = SmoothL1CriterionWithWeights2[Float](1)
    val input = TestUtil.loadFeatures("bbox_pred")
    val bbox_targets = TestUtil.loadFeatures("bbox_targets")
    val bbox_inside_weights = TestUtil.loadFeatures("bbox_inside_weights")
    val bbox_outside_weights = TestUtil.loadFeatures("bbox_outside_weights")

    val target = T(bbox_targets, bbox_inside_weights, bbox_outside_weights)
    println(criterion.forward(input, target))
    println(criterion.output)
  }

}
