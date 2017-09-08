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

package com.intel.analytics.zoo.pipeline.fasterrcnn.example

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.pipeline.common.ParamUtil._
import com.intel.analytics.bigdl.pipeline.common.{IOUtils, PascalVocEvaluator}
import com.intel.analytics.bigdl.pipeline.fasterrcnn.model.{PvanetFRcnn, VggFRcnn}
import com.intel.analytics.bigdl.pipeline.fasterrcnn.{PostProcessParam, PreProcessParam, Predictor, Validator}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object Test {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.pipeline.fasterrcnn").setLevel(Level.INFO)


  def main(args: Array[String]) {
    testParamParser.parse(args, TestParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("Spark-DL Faster RCNN Test")
      val sc = new SparkContext(conf)
      Engine.init

      val evaluator = new PascalVocEvaluator(params.imageSet)
      val rdd = IOUtils.loadSeqFiles(Engine.nodeNumber * Engine.coreNumber, params.folder, sc)
      val (model, preParam, postParam) = params.modelType.toLowerCase() match {
        case "vgg16" =>
          (Module.loadCaffe(VggFRcnn(params.nClass),
            params.caffeDefPath, params.caffeModelPath),
            PreProcessParam(),
            PostProcessParam(0.3f, params.nClass, false, 100, 0.05))
        case "pvanet" =>
          (Module.loadCaffe(PvanetFRcnn(params.nClass),
            params.caffeDefPath, params.caffeModelPath),
            PreProcessParam(1, Array(640), 32),
            PostProcessParam(0.4f, params.nClass, true, 100, 0.05))
        case _ =>
          throw new Exception("unsupport network")
      }

      val validator = new Validator(model, preParam, postParam, evaluator)

      validator.test(rdd)
    }
  }
}
