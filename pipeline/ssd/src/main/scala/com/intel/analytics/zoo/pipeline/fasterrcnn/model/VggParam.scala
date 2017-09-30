/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.fasterrcnn.model

import com.intel.analytics.zoo.pipeline.fasterrcnn.model.Model.ModelType

case class VggParam() extends FasterRcnnParam {
//  val anchorParam = AnchorParam(_scales = Array[Float](8, 16, 32),
//    _ratios = Array[Float](0.5f, 1.0f, 2.0f))
  val ratios = Array[Float](0.5f, 1.0f, 2.0f)
  val scales = Array[Float](8, 16, 32)
  override val BG_THRESH_LO = 0.0
  override val BATCH_SIZE = 128
  override val modelType: ModelType = Model.VGG16

  RPN_PRE_NMS_TOP_N = 12000
  RPN_POST_NMS_TOP_N = 2000
}


