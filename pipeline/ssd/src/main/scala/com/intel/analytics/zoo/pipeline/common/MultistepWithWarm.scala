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
 *
 */

package com.intel.analytics.bigdl.pipeline.common

import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.optim.SGD.LearningRateSchedule

case class MultistepWithWarm(stepSizes: Array[Int], gamma: Double, warmUpIteration: Int = 0,
  minLr: Double)
  extends LearningRateSchedule {

  override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
    val lr = optimMethod.learningRate
    var clr = -lr
    val warmUpDelta = (lr - minLr) / warmUpIteration
    val nevals = optimMethod.state.get[Int]("evalCounter").getOrElse(0)
    if (nevals < warmUpIteration) {
      clr = -minLr - warmUpDelta * nevals
    } else {
      var currentStep = 0
      while (currentStep < stepSizes.length && nevals >= stepSizes(currentStep)) {
        clr *= gamma
        currentStep += 1
      }
    }
    optimMethod.state("evalCounter") = nevals + 1
    currentRate = clr
  }
}
