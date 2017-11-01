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

package com.intel.analytics.zoo.pipeline.common

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.pipeline.common.Compress
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.scalatest.{FlatSpec, Matchers}


class CompressSpec extends FlatSpec with Matchers {
  "svd" should "work properly" in {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("com.intel.analytics.bigdl.pipeline.fasterrcnn").setLevel(Level.INFO)
    val conf = Engine.createSparkConf().setMaster("local[*]").setAppName("test compress")
    val sc = new SparkContext(conf)
    Engine.init
    val input = Tensor[Double](T(2, 4, 1, 3, 0, 0, 0, 0)).resize(4, 2)
    val out = Compress.compressWeight(sc, input, 2)
    val expectedU = Tensor[Double](T(-0.81741556, -0.57604844, -0.57604844, 0.81741556, 0,
      0, 0, 0)).resize(4, 2)
    val expectedV = Tensor[Double](T(-2.21087956, -4.99780755, -0.33468131, 0.14805293)).resize(2, 2)

    out._1.map(expectedU, (a, b) => {
      assert((a - b).abs < 1e-6); a
    })
    out._2.map(expectedV, (a, b) => {
      assert((a - b).abs < 1e-6); a
    })
  }

  "svd2" should "work properly" in {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("com.intel.analytics.bigdl.pipeline.fasterrcnn").setLevel(Level.INFO)
    val conf = Engine.createSparkConf().setMaster("local[*]").setAppName("test compress")
    val sc = new SparkContext(conf)
    Engine.init
    val input = Tensor[Double](512 * 7 * 7, 4096).randn()
    val start = System.nanoTime()
    val out = Compress.compressWeight(sc, input, 1024)
    println(s" takes ${(System.nanoTime() - start) / 1e9} s")

    val input2 = Tensor[Double](1024, 1024).randn()
    val start2 = System.nanoTime()
    val out2 = Compress.compressWeight(sc, input, 256)
    println(s" takes ${(System.nanoTime() - start2) / 1e9} s")
  }

  "svd3" should "work properly" in {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("com.intel.analytics.bigdl.pipeline.fasterrcnn").setLevel(Level.INFO)
    val conf = Engine.createSparkConf().setMaster("local[*]").setAppName("test compress")
    val sc = new SparkContext(conf)
    Engine.init
    val input = Tensor[Double](4096, 84).randn()
    val start = System.nanoTime()
    val out = Compress.compressWeight2(sc, input, 1024)
    println(s" takes ${(System.nanoTime() - start) / 1e9} s")
  }

  "svd4" should "work properly" in {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("com.intel.analytics.bigdl.pipeline.fasterrcnn").setLevel(Level.INFO)
    val conf = Engine.createSparkConf().setMaster("local[*]").setAppName("test compress")
    val sc = new SparkContext(conf)
    Engine.init
    val input = Tensor[Double](4096, 4096).randn()
    val start = System.nanoTime()
    val out = Compress.compressWeight(sc, input, 256)
    println(s" takes ${(System.nanoTime() - start) / 1e9} s")
  }

  "compress frcnn" should "work" in {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("com.intel.analytics.bigdl.pipeline.fasterrcnn").setLevel(Level.INFO)
    val conf = Engine.createSparkConf().setMaster("local[*]").setAppName("test compress")
    val sc = new SparkContext(conf)
    Engine.init
    val model = Module.loadModule[Float](
      "/home/jxy/code/analytics-zoo/pipeline/ssd/data/models/bigdl_frcnn_vgg_voc.model")
    println("load model done ...")
    val linears = Map("fc6" -> 1024, "fc7" -> 256)
    val compressed = Compress.compress[Float](model, linears, sc)
    println("compress done ...")
    compressed.saveModule("/tmp/vgg_compress.model", true)
  }
}
