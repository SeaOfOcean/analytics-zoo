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

package com.intel.analytics.zoo.pipeline.ssd.example

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.pipeline.common.{MultistepWithWarm, PlateauWithWarm}
import com.intel.analytics.bigdl.pipeline.ssd.IOUtils
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.pipeline.common.MeanAveragePrecision
import com.intel.analytics.zoo.pipeline.common.caffe.CaffeLoader
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.SSDMiniBatch
import com.intel.analytics.zoo.pipeline.common.nn.{MultiBoxLoss, MultiBoxLossParam}
import com.intel.analytics.zoo.pipeline.ssd.model.SSDVgg
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

object Option {

  case class TrainParams(
    trainFolder: String = "./",
    valFolder: String = "./",
    modelType: String = "vgg16",
    caffeDefPath: Option[String] = None,
    caffeModelPath: Option[String] = None,
    resolution: Int = 300,
    checkpoint: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    classNumber: Int = 21,
    batchSize: Int = -1,
    warmLearningRate: Double = 0.0001,
    learningRate: Double = 0.001,
    sgdWarmEpoch: Int = 5,
    schedule: String = "multistep",
    learningRateDecay: Double = 0.1,
    learningRateSteps: Option[Array[Int]] = None,
    patience: Int = 10,
    warmUpMap: Option[Double] = None,
    overWriteCheckpoint: Boolean = false,
    resumeEpoch: Option[Int] = None,
    maxEpoch: Option[Int] = None,
    weights: Option[String] = None,
    jobName: String = "BigDL SSD Train Example",
    summaryDir: Option[String] = None,
    gradientClipMax: Double = 0
  )

  val trainParser = new OptionParser[TrainParams]("BigDL SSD Example") {
    opt[String]('f', "trainFolder")
      .text("url of hdfs folder store the train hadoop sequence files")
      .action((x, c) => c.copy(trainFolder = x))
    opt[String]('v', "valFolder")
      .text("url of hdfs folder store the validation hadoop sequence files")
      .action((x, c) => c.copy(valFolder = x))
    opt[String]('t', "modelType")
      .text("net type : vgg16")
      .action((x, c) => c.copy(modelType = x))
      .required()
    opt[String]("caffeDefPath")
      .text("caffe prototxt")
      .action((x, c) => c.copy(caffeDefPath = Some(x)))
    opt[String]("caffeModelPath")
      .text("caffe model path")
      .action((x, c) => c.copy(caffeModelPath = Some(x)))
    opt[Int]('r', "resolution")
      .text("input resolution 300 or 512")
      .action((x, c) => c.copy(resolution = x))
      .required()
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("weights")
      .text("pretrained weights")
      .action((x, c) => c.copy(weights = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Int]("patience")
      .text("epoch to wait")
      .action((x, c) => c.copy(patience = x))
    opt[Int]("warmEpoch")
      .text("epoch for sgd to warm")
      .action((x, c) => c.copy(sgdWarmEpoch = x))
    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = Some(x)))
    opt[Double]('l', "learningRate")
      .text("inital learning rate")
      .action((x, c) => c.copy(learningRate = x))
      .required()
    opt[Double]("warmLearningRate")
      .text("warm up learning rate")
      .action((x, c) => c.copy(warmLearningRate = x))
    opt[String]("schedule")
      .text("learning rate schedule")
      .action((x, c) => c.copy(schedule = x))
      .required()
    opt[Double]('d', "learningRateDecay")
      .text("learning rate decay")
      .action((x, c) => c.copy(learningRateDecay = x))
    opt[String]("step")
      .text("learning rate steps, split by ,")
      .action((x, c) => c.copy(learningRateSteps = Some(x.split(",").map(_.toInt))))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
      .required()
    opt[Int]("classNum")
      .text("class number")
      .action((x, c) => c.copy(classNumber = x))
    opt[Unit]("overWrite")
      .text("overwrite checkpoint files")
      .action((_, c) => c.copy(overWriteCheckpoint = true))
    opt[Double]("warm")
      .text("warm up map")
      .action((x, c) => c.copy(warmUpMap = Some(x)))
    opt[String]("name")
      .text("job name")
      .action((x, c) => c.copy(jobName = x))
    opt[String]("summary")
      .text("train validate summary")
      .action((x, c) => c.copy(summaryDir = Some(x)))
    opt[Int]("resumeEpoch")
      .text("which epoch to start")
      .action((x, c) => c.copy(resumeEpoch = Some(x)))
    opt[Double]("gradientClipMax")
      .text("max norm2 of gradient clipping")
      .action((x, c) => c.copy(gradientClipMax = x))
  }
}


object Train {

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.bigdl.pipeline").setLevel(Level.INFO)

  import Option._

  val logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName(param.jobName)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val trainSet = IOUtils.loadTrainSet(param.trainFolder, sc, param.resolution, param.batchSize)

      val valSet = IOUtils.loadValSet(param.valFolder, sc, param.resolution, param.batchSize)

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        param.modelType match {
          case "vgg16" =>
            val model = SSDVgg(param.classNumber, param.resolution)
            if (param.weights.isDefined) {
              model.loadWeights(param.weights.get)
            } else if (param.caffeDefPath.isDefined && param.caffeModelPath.isDefined) {
              CaffeLoader.load[Float](model,
                param.caffeDefPath.get, param.caffeModelPath.get, matchAll = false)
            }
            model
          case _ => throw new Exception("currently only test over vgg ssd model")
        }
      }

      val warmUpModel = if (param.warmUpMap.isDefined) {
        val optimMethod = new Adam[Float](
          learningRate = param.warmLearningRate,
          learningRateDecay = 0.0005
        )
        optimize(model, trainSet, valSet, param, optimMethod,
          Trigger.maxScore(param.warmUpMap.get.toFloat))
      } else {
        model
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        val iterationsPerEpoch = math.ceil(1281167.toFloat / param.batchSize).toInt
        val learningRateSchedule = param.schedule match {
          case "multistep" =>
            val steps = if (param.learningRateSteps.isDefined) {
              param.learningRateSteps.get
            } else {
              Array[Int](80000 * 32 / param.batchSize, 100000 * 32 / param.batchSize,
                120000 * 32 / param.batchSize)
            }
            MultistepWithWarm(steps, param.learningRateDecay, 16551 * param.sgdWarmEpoch / param.batchSize, 0.001)
          case "plateau" =>
            PlateauWithWarm(monitor = "score",
              factor = param.learningRateDecay.toFloat,
              patience = param.patience, minLr = 1e-5f, mode = "max",
              warmUpIteration = 16551 * 5 / param.batchSize,
              startWarnLr = 0.001)
        }
        val sgd = new SGD[Float](
          learningRate = param.learningRate,
          momentum = 0.9,
          dampening = 0.0,
          learningRateSchedule = learningRateSchedule,
          gradientClipMax = param.gradientClipMax)
        if (param.resumeEpoch.isDefined) {
          val resumeEpoch = param.resumeEpoch.get
          val neval = (resumeEpoch - 1) * iterationsPerEpoch + 1
          sgd.setState(T(
            "epoch" -> resumeEpoch,
            "neval" -> neval,
            "evalCounter" -> (neval - 1)
          ))
        }
        sgd
      }

      optimize(warmUpModel, trainSet, valSet, param, optimMethod,
        Trigger.maxEpoch(param.maxEpoch.get))

    })
  }

  private def optimize(model: Module[Float],
    trainSet: DataSet[SSDMiniBatch],
    valSet: DataSet[SSDMiniBatch], param: TrainParams, optimMethod: OptimMethod[Float],
    endTrigger: Trigger): Module[Float] = {
    val optimizer = Optimizer(
      model = model,
      dataset = trainSet,
      criterion = new MultiBoxLoss[Float](MultiBoxLossParam())
    )

    if (param.checkpoint.isDefined) {
      optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
    }

    if (param.overWriteCheckpoint) {
      optimizer.overWriteCheckpoint()
    }

    if (param.summaryDir.isDefined) {
      val trainSummary = TrainSummary(param.summaryDir.get, param.jobName)
      val validationSummary = ValidationSummary(param.summaryDir.get, param.jobName)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
//      trainSummary.setSummaryTrigger("gradientNorm2", Trigger.severalIteration(1))
//      trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(10))
      optimizer.setTrainSummary(trainSummary)
      optimizer.setValidationSummary(validationSummary)
    }
    optimizer
      .setOptimMethod(optimMethod)
      .setValidation(Trigger.everyEpoch,
        valSet.asInstanceOf[DataSet[MiniBatch[Float]]],
        Array(new MeanAveragePrecision(true, normalized = true,
          nClass = param.classNumber)))
      .setEndWhen(endTrigger)
      .optimize()
  }
}
