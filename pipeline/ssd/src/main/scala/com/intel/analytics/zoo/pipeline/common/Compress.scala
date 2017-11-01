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

package com.intel.analytics.bigdl.pipeline.common

import breeze.linalg.{diag, svd, DenseMatrix => BrzDenseMatrx, Matrix => BrzMatrx}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
//import org.apache.spark.mllib.linalg.DenseMatrix
//import org.apache.spark.mllib.linalg.distributed.RowMatrix
import scopt.OptionParser

import scala.reflect._


object Compress {
  val logger = Logger.getLogger(getClass)

  /**
   * Compress the weight matrix W of an inner product (fully connected) layer
   * using truncated SVD.
   *
   * @param weight N x M weights matrix
   * @param l number of singular values to retain
   * @return Ul, L: matrices such that W \approx Ul*L
   */
  def compressWeight[T: ClassTag](sc: SparkContext, weight: Tensor[T], l: Int): (Tensor[T], Tensor[T]) = {
    // MLlibMatrix only accept Double type
    val doubleWeight = if (classTag[T] == classTag[Float]) {
      toDoubleTensor(weight.asInstanceOf[Tensor[Float]])
    } else weight
    val array = doubleWeight.toMLlibMatrix().rowIter.toArray
    val rows = sc.parallelize(array)
    logger.info(s"start computing svd for weight ${weight.size().mkString("x")}.. with s ${l}")
//    val dim = Math.min(weight.size(1), weight.size(2))
    val svd = new RowMatrix(rows).computeSVD(l, true)
    logger.info("compute svd done ..")
    // Create the inv diagonal matrix from S
    val invS = DenseMatrix.diag(svd.s)
    val U = Tensor(Storage(svd.U.rows.collect().flatMap(x => x.toArray)))
      .resize(svd.U.numRows().toInt, svd.U.numCols().toInt)
    val L = svd.V match {
      case dense: org.apache.spark.mllib.linalg.DenseMatrix =>
        invS.multiply(dense.transpose)
      case sparse: org.apache.spark.mllib.linalg.SparseMatrix =>
        invS.multiply(sparse.toDense.transpose)
    }
    // data type from svd is Double, need to convert to Float if T == Float
    val (tu, tl) = if (classTag[T] == classTag[Float]) {
      (toFloatTensor(U), toFloatTensor(Tensor(L)))
    } else {
      (U, Tensor(L))
    }

    logger.info(s"compress linear  weight from ${weight.size().mkString("x")} with l $l to" +
      s" ${tu.size().mkString("x")} and ${tl.size().mkString("x")}")
    (tu.asInstanceOf[Tensor[T]], tl.asInstanceOf[Tensor[T]])
  }

  def compressWeight2[T: ClassTag](sc: SparkContext, weight: Tensor[T], l: Int): (Tensor[T], Tensor[T]) = {
    val (u, v) = if (classTag[T] == classTag[Float]) {
      compressWeigtFloat(sc, weight.asInstanceOf[Tensor[Float]], l)
    } else {
      compressWeightDouble(sc, weight.asInstanceOf[Tensor[Double]], l)
    }
    (u.asInstanceOf[Tensor[T]], v.asInstanceOf[Tensor[T]])
  }

  /**
   * Compress the weight matrix W of an inner product (fully connected) layer
   * using truncated SVD.
   *
   * @param weight N x M weights matrix
   * @param l number of singular values to retain
   * @return Ul, L: matrices such that W \approx Ul*L
   */
  def compressWeigtFloat(sc: SparkContext, weight: Tensor[Float], l: Int): (Tensor[Float], Tensor[Float]) = {
    logger.info(s"start computing svd for weight ${weight}.. with s ${l}")
    val svd.SVD(u, s, v) = svd.reduced(weight.toBreezeMatrix())
    val L = diag(s) * v
    logger.info("compute svd done ..")
    (Tensor(u), Tensor(L))
  }

  def compressWeightDouble(sc: SparkContext, weight: Tensor[Double], l: Int): (Tensor[Double], Tensor[Double]) = {
    logger.info(s"start computing svd for weight ${weight.size().mkString("x")}.. with s ${l}")
    val svd.SVD(u, s, v) = svd.reduced(weight.toBreezeMatrix())
    val L = diag(s) * v
    logger.info("compute svd done ..")
    (Tensor(u), Tensor(L))
  }


  def toFloatTensor(tensor: Tensor[Double]): Tensor[Float] = {
    val arr = tensor.storage().array().map(_.toFloat)
    Tensor(Storage(arr), tensor.storageOffset(),
      tensor.size(), tensor.stride())
  }

  def toDoubleTensor(tensor: Tensor[Float]): Tensor[Double] = {
    val arr = tensor.storage().array().map(_.toDouble)
    Tensor(Storage(arr), tensor.storageOffset(),
      tensor.size(), tensor.stride())
  }

  def compressLinear[T: ClassTag](linear: Linear[T], l: Int, sc: SparkContext)(
    implicit ev: TensorNumeric[T]): Module[T] = {
    require(linear.weight != null && linear.bias != null)
    val (ul, lL) = compressWeight(sc, linear.weight, l)
    val linearL = Linear[T](linear.inputSize, l, withBias = false)
    val linearU = Linear[T](l, linear.outputSize)
    linearL.weight.copy(lL)
    linearU.weight.copy(ul)
    linearU.bias.copy(linear.bias)
    Sequential[T].add(linearL).add(linearU)
  }

  def compressGraph[T: ClassTag](module: Module[T], linears: Map[String, Int],
    sc: SparkContext)(
    implicit ev: TensorNumeric[T]): Module[T] = {
    val graph = module.asInstanceOf[Graph[T]]
    val sortedNodes = graph.getForwardExecutions

    for (i <- sortedNodes.indices) {
      val currNode = sortedNodes(i)
      val currModule = currNode.element
      val waitedModule = compress(currNode.element, linears, sc)
      if (waitedModule != currModule) {
        currNode.setElement(waitedModule)
      }
    }

    // modules in container need to rebuild
    graph.resetModules()
    // nodes in backward executions need to rebuild
    graph.build()
  }

  def compressContainer[T: ClassTag](model: Container[Activity, Activity, T],
    linears: Map[String, Int],
    sc: SparkContext)(
    implicit ev: TensorNumeric[T]): Module[T] = {
    for (i <- model.modules.indices) {
      val curModule = model.modules(i)
      model.modules(i) = compress(curModule, linears, sc)
    }
    model
  }

  def compressCell[T: ClassTag](model: Cell[T], linears: Map[String, Int],
    sc: SparkContext)(
    implicit ev: TensorNumeric[T]): Module[T] = {
    model.cell = compress(model.cell, linears, sc)
    model
  }

  def compress[T: ClassTag](model: Module[T], linears: Map[String, Int],
    sc: SparkContext)(
    implicit ev: TensorNumeric[T]): Module[T] = {
    if (model.isInstanceOf[Linear[T]] && linears.contains(model.getName())) {
      compressLinear[T](model.asInstanceOf[Linear[T]], linears(model.getName()), sc)
    } else {
      model match {
        case container: Container[Activity, Activity, T] =>
          container match {
            case graph: Graph[T] => compressGraph(graph, linears, sc)
            case _ => compressContainer(container, linears, sc)
          }
        case cell if cell.isInstanceOf[Cell[T]] =>
          // because Cell[T] extends AbstractModule[Table, Table, T], and the Table is a class,
          // which is not as same as trait Tensor. So if we use this form:
          //   case cell: Cell[T] => CellQuantizer.quantize(cell)
          // scalac will throw an compiler error.
          compressCell(cell.asInstanceOf[Cell[T]], linears, sc)
        case _ => model
      }
    }
  }

  case class CompressParam(
    model: String = "",
    output: String = "",
    params: String = "")

  val parser = new OptionParser[CompressParam]("BigDL Compress") {
    head("BigDL Compress")
    opt[String]('m', "model")
      .text("bigdl model")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[String]('o', "output")
      .text("output bigDL model")
      .action((x, c) => c.copy(output = x))
      .required()
    opt[String]('p', "param")
      .text("linear name and param")
      .action((x, c) => c.copy(params = x))
      .required()
  }


  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)

  def main(args: Array[String]) {
    parser.parse(args, CompressParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("BigDL Compress")
      val sc = new SparkContext(conf)
      Engine.init
      val linears = params.params.split("\\s").grouped(2).map(x => (x(0), x(1).toInt)).toMap
      val model = Module.loadModule[Float](params.model)
      logger.info(s"load model ${model.getName()} done ...")
      val compressed = Compress.compress[Float](model, linears, sc)
      compressed.saveModule(params.output, true)
    }
  }
}
