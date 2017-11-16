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

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.pipeline.ssd.IOUtils
import com.intel.analytics.zoo.pipeline.common.BboxUtil
import com.intel.analytics.zoo.pipeline.common.caffe.SSDCaffeLoader
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.Visualizer
import com.intel.analytics.zoo.pipeline.ssd._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.ssd.model.PreProcessParam
import com.intel.analytics.zoo.transform.vision.image.{Image, ImageFeature}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import scopt.OptionParser

import scala.io.Source

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.pipeline.ssd").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PascolVocDemoParam(imageLoc: String = "",
    outputFolder: String = "data/demo",
    folderType: String = "local",
    modelType: String = "vgg16",
    model: Option[String] = None,
    caffeDefPath: Option[String] = None,
    caffeModelPath: Option[String] = None,
    batch: Int = 8,
    savetxt: Boolean = true,
    vis: Boolean = true,
    classname: String = "",
    resolution: Int = 300,
    topK: Option[Int] = None,
    nPartition: Int = 1,
    sql: String = "")

  val parser = new OptionParser[PascolVocDemoParam]("BigDL SSD Demo") {
    head("BigDL SSD Demo")
    opt[String]('f', "folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(imageLoc = x))
      .required()
    opt[String]("folderType")
      .text("local image folder or hdfs sequence folder")
      .action((x, c) => c.copy(folderType = x))
      .required()
      .validate(x => {
        if (Set("local", "seq").contains(x.toLowerCase)) {
          success
        } else {
          failure("folderType only support local|seq")
        }
      })
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
      .required()
    opt[String]('t', "modelType")
      .text("net type : vgg16 | alexnet")
      .action((x, c) => c.copy(modelType = x))
      .required()
    opt[String]("model")
      .text("BigDL model")
      .action((x, c) => c.copy(model = Some(x)))
    opt[String]("caffeDefPath")
      .text("caffe prototxt")
      .action((x, c) => c.copy(caffeDefPath = Some(x)))
    opt[String]("caffeModelPath")
      .text("caffe model path")
      .action((x, c) => c.copy(caffeModelPath = Some(x)))
    opt[Int]('b', "batch")
      .text("batch number")
      .action((x, c) => c.copy(batch = x))
    opt[Boolean]('s', "savetxt")
      .text("whether to save detection results")
      .action((x, c) => c.copy(savetxt = x))
    opt[Boolean]('v', "vis")
      .text("whether to visualize the detections")
      .action((x, c) => c.copy(vis = x))
    opt[String]("classname")
      .text("file store class name")
      .action((x, c) => c.copy(classname = x))
      .required()
    opt[Int]('r', "resolution")
      .text("input resolution 300 or 512")
      .action((x, c) => c.copy(resolution = x))
      .required()
    opt[Int]('k', "topk")
      .text("return topk results")
      .action((x, c) => c.copy(topK = Some(x)))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
      .required()
    opt[String]("sql")
      .text("sql to query url")
      .action((x, c) => c.copy(sql = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PascolVocDemoParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("BigDL SSD Demo")
      val sc = new SparkContext(conf)
      val ss = SparkSession.builder().enableHiveSupport().getOrCreate()
      Engine.init

      val classNames = Source.fromFile(params.classname).getLines().toArray

      val model = if (params.model.isDefined) {
        // load BigDL model
        Module.loadModule[Float](params.model.get)
      } else if (params.caffeDefPath.isDefined && params.caffeModelPath.isDefined) {
        // load caffe dynamically
        SSDCaffeLoader.loadCaffe(params.caffeDefPath.get, params.caffeModelPath.get)
      } else {
        throw new IllegalArgumentException(s"currently only support" +
          s" loading BigDL model or caffe model")
      }

      val imageFrame = params.folderType match {
        case "local" => Image.read(params.imageLoc, sc)
        case "seq" => IOUtils.loadImageFrameFromSeq(params.nPartition, params.imageLoc, sc)
        case "hbase" => IOUtils.loadImageFrameFromHbase(params.nPartition, params.imageLoc, params.sql, ss)
        case _ => throw new IllegalArgumentException(s"invalid folder name ${params.folderType}")
      }

      val predictor = new SSDPredictor(model,
        PreProcessParam(params.batch, params.resolution, (123f, 117f, 104f), false, params.nPartition))

      val start = System.nanoTime()
      val featureKey = "roi"
      val output = predictor.predictWithFeature(imageFrame, featureKey)

      if (params.savetxt) {
        output.rdd.map { feature => BboxUtil.featureToString(feature, featureKey)
        }.saveAsTextFile(params.outputFolder)
      } else {
        output.rdd.count()
      }

      val totalTime = (System.nanoTime() - start) / 1e9
      logger.info(s"[Prediction] total time: $totalTime seconds")

      if (params.vis) {
        if (params.folderType == "seq") {
          logger.warn("currently only support visualize local folder in Predict")
          return
        }

        output.rdd.foreach(feature => {
          val decoded = BboxUtil.decodeRois(feature[Tensor[Float]](featureKey))
          Visualizer.visDetection(feature[String](ImageFeature.uri), decoded, classNames, outPath = params.outputFolder)
        })
        logger.info(s"labeled images are saved to ${params.outputFolder}")
      }
    }
  }
}
