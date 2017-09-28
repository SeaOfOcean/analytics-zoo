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

package com.intel.analytics.zoo.pipeline.fasterrcnn

import com.intel.analytics.bigdl.nn.Nms
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.zoo.pipeline.common.BboxUtil
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.PostProcessParam
import com.intel.analytics.zoo.transform.vision.label.roi.RoiLabel
import org.apache.commons.lang3.SerializationUtils
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer


object Postprocessor {
  val logger = Logger.getLogger(this.getClass)

  def apply(param: PostProcessParam): Postprocessor = new Postprocessor(param)
}

class Postprocessor(param: PostProcessParam) extends AbstractModule[Table, Activity, Float] {

  @transient var nmsTool: Nms = _

  /**
   *
   * @param scores (N, clsNum)
   * @param boxes (N, 4 * clsNum)
   * @return
   */
  private def postProcess(scores: Tensor[Float], boxes: Tensor[Float])
  : Array[RoiLabel] = {
    require(scores.size(1) == boxes.size(1))
    val results = new Array[RoiLabel](param.nClasses)
    // skip j = 0, because it's the background class
    var clsInd = 1
    while (clsInd < param.nClasses) {
      results(clsInd) = postProcessOneClass(scores, boxes, clsInd)
      clsInd += 1
    }

    // Limit to max_per_image detections *over all classes*
    if (param.maxPerImage > 0) {
      limitMaxPerImage(results)
    }
    results
  }

  private def resultToTensor(results: Array[RoiLabel]): Tensor[Float] = {
    var maxDetection = 0
    results.foreach(res => {
      if (null != res) {
        maxDetection += res.size()
      }
    })
    val out = Tensor[Float](1, 1 + maxDetection * 6)
    val outi = out(1)

    outi.setValue(1, maxDetection)
    var offset = 2
    var c = 0
    results.filter(_ != null).foreach(label => {
      (1 to label.size()).foreach(j => {
        outi.setValue(offset, c)
        outi.setValue(offset + 1, label.classes.valueAt(j))
        outi.setValue(offset + 2, label.bboxes.valueAt(j, 1))
        outi.setValue(offset + 3, label.bboxes.valueAt(j, 2))
        outi.setValue(offset + 4, label.bboxes.valueAt(j, 3))
        outi.setValue(offset + 5, label.bboxes.valueAt(j, 4))
        offset += 6
      })
      c += 1
    })
    out
//    if (maxDetection > 0) {
//      var i = 0
//      while (i < batch) {
//        val outi = out(i + 1)
//        var c = 0
//        outi.setValue(1, maxDetection)
//        var offset = 2
//        while (c < allIndices(i).length) {
//          val indices = allIndices(i)(c)
//          if (indices != null) {
//            val indicesNum = allIndicesNum(i)(c)
//            val locLabel = if (param.shareLocation) allDecodedBboxes(i).length - 1 else c
//            val bboxes = allDecodedBboxes(i)(locLabel)
//            var bboxesOffset = allDecodedBboxes(i)(locLabel).storageOffset() - 1
//            var j = 0
//            while (j < indicesNum) {
//              val idx = indices(j)
//              outi.setValue(offset, c)
//              outi.setValue(offset + 1, allConfScores(i)(c).valueAt(idx))
//              outi.setValue(offset + 2, bboxes.valueAt(idx, 1))
//              outi.setValue(offset + 3, bboxes.valueAt(idx, 2))
//              outi.setValue(offset + 4, bboxes.valueAt(idx, 3))
//              outi.setValue(offset + 5, bboxes.valueAt(idx, 4))
//              offset += 6
//              j += 1
//            }
//          }
//          c += 1
//        }
//        i += 1
//      }
//    }
  }

  @transient private var areas: Tensor[Float] = _

  private def postProcessOneClass(scores: Tensor[Float], boxes: Tensor[Float],
    clsInd: Int): RoiLabel = {
    val inds = (1 to scores.size(1)).filter(ind =>
      scores.valueAt(ind, clsInd + 1) > param.thresh).toArray
    if (inds.length == 0) return null
    val clsScores = selectTensor(scores.select(2, clsInd + 1), inds, 1)
    val clsBoxes = selectTensor(boxes.narrow(2, clsInd * 4 + 1, 4), inds, 1)

    val keepN = nmsTool.nms(clsScores, clsBoxes, param.nmsThresh, inds)

    val bboxNms = selectTensor(clsBoxes, inds, 1, keepN)
    val scoresNms = selectTensor(clsScores, inds, 1, keepN)
    if (param.bboxVote) {
      if (areas == null) areas = Tensor[Float]
      BboxUtil.bboxVote(scoresNms, bboxNms, clsScores, clsBoxes, areas)
    } else {
      RoiLabel(scoresNms, bboxNms)
    }
  }

  private def selectTensor(matrix: Tensor[Float], indices: Array[Int],
    dim: Int, indiceLen: Int = -1, out: Tensor[Float] = null): Tensor[Float] = {
    assert(dim == 1 || dim == 2)
    var i = 1
    val n = if (indiceLen == -1) indices.length else indiceLen
    if (matrix.nDimension() == 1) {
      val res = if (out == null) {
        Tensor[Float](n)
      } else {
        out.resize(n)
      }
      while (i <= n) {
        res.update(i, matrix.valueAt(indices(i - 1)))
        i += 1
      }
      return res
    }
    // select rows
    if (dim == 1) {
      val res = if (out == null) {
        Tensor[Float](n, matrix.size(2))
      } else {
        out.resize(n, matrix.size(2))
      }
      while (i <= n) {
        res.update(i, matrix(indices(i - 1)))
        i += 1
      }
      res
    } else {
      val res = if (out == null) {
        Tensor[Float](matrix.size(1), n)
      } else {
        out.resize(matrix.size(1), n)
      }
      while (i <= n) {
        var rid = 1
        val value = matrix.select(2, indices(i - 1))
        while (rid <= res.size(1)) {
          res.setValue(rid, i, value.valueAt(rid))
          rid += 1
        }
        i += 1
      }
      res
    }
  }

  def limitMaxPerImage(results: Array[RoiLabel]): Unit = {
    val nImageScores = (1 until param.nClasses).map(j => if (results(j) == null) 0
    else results(j).classes.size(1)).sum
    if (nImageScores > param.maxPerImage) {
      val imageScores = ArrayBuffer[Float]()
      var j = 1
      while (j < param.nClasses) {
        val res = results(j).classes
        if (res.nElement() > 0) {
          res.apply1(x => {
            imageScores.append(x)
            x
          })
        }
        j += 1
      }
      val imageThresh = imageScores.sortWith(_ < _)(imageScores.length - param.maxPerImage)
      j = 1
      while (j < param.nClasses) {
        val box = results(j).bboxes
        val keep = (1 to box.size(1)).filter(x =>
          box.valueAt(x, box.size(2)) >= imageThresh).toArray
        val selectedScores = selectTensor(results(j).classes, keep, 1)
        val selectedBoxes = selectTensor(results(j).bboxes, keep, 1)
        results(j).classes.resizeAs(selectedScores).copy(selectedScores)
        results(j).bboxes.resizeAs(selectedBoxes).copy(selectedBoxes)
        j += 1
      }
    }
  }

  @transient var boxesBuf: Tensor[Float] = _

  def process(scores: Tensor[Float],
    boxDeltas: Tensor[Float],
    rois: Tensor[Float],
    imInfo: Tensor[Float]): Array[RoiLabel] = {
    if (nmsTool == null) nmsTool = new Nms
    // post process
    // unscale back to raw image space
    if (boxesBuf == null) boxesBuf = Tensor[Float]
    boxesBuf.resize(rois.size(1), 4).copy(rois.narrow(2, 2, 4)).div(imInfo.valueAt(1, 3))
    // Apply bounding-box regression deltas
    val predBoxes = BboxUtil.bboxTransformInv(boxesBuf, boxDeltas)
    BboxUtil.clipBoxes(predBoxes, imInfo.valueAt(1, 1) / imInfo.valueAt(1, 3),
      imInfo.valueAt(1, 2) / imInfo.valueAt(1, 4))
    val res = postProcess(scores, predBoxes)
    res
  }


  override def clone(): Postprocessor = {
    SerializationUtils.clone(this)
  }

  override def updateOutput(input: Table): Activity = {
    if (isTraining()) {
      output = input
      return output
    }
    val scores = input[Tensor[Float]](1)
    val boxDeltas = input[Tensor[Float]](2)
    val rois = input[Table](3)[Tensor[Float]](1)
    val imInfo = input[Tensor[Float]](7)
    output = resultToTensor(process(scores, boxDeltas, rois, imInfo))
    output
  }

  override def updateGradInput(input: Table, gradOutput: Activity): Table = {
    gradInput = null
    gradInput
  }
}