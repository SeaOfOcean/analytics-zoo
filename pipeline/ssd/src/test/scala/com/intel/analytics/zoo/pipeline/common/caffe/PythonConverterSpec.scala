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

import org.scalatest.{FlatSpec, Matchers}

import scala.util.parsing.json.JSON

class PythonConverterSpec extends FlatSpec with Matchers{
  "parse param" should "work properly" in {
    val param = "{'feat_stride': 16, 'ratios': [0.333, 0.5, 0.667, 1, 1.5, 2, 3], 'scales': [2, 3, 5, 9, 16, 32]}"
    val str = param.replaceAll("'", "\"")
    val result = JSON.parseFull(str)
    println(result)
    result match {
      case Some(map: Map[String, Any]) => {
        map("feat_stride") should be (16)
        map("ratios") should be (List(0.333, 0.5, 0.667, 1, 1.5, 2, 3))
        map("scales") should be (List(2, 3, 5, 9, 16, 32))
        val ratios = map("ratios").asInstanceOf[List[Double]].toArray
        val scales = map("scales").asInstanceOf[List[Double]].toArray
      }
    }
  }
}
