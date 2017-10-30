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

package com.intel.analytics.zoo.pipeline.common

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.transform.vision.util.NormalizedBox
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag

class BboxUtilSpec extends FlatSpec with Matchers {
  "jaccardOverlap partial overlap" should "work properly" in {
    val bbox1 = NormalizedBox(0.2f, 0.3f, 0.3f, 0.5f)
    val bbox2 = NormalizedBox(0.1f, 0.1f, 0.3f, 0.4f)
    val overlap = BboxUtil.jaccardOverlap(bbox1, bbox2)
    assert(Math.abs(overlap - 1.0 / 7) < 1e-6)
  }

  "jaccardOverlap fully contain" should "work properly" in {
    val bbox1 = NormalizedBox(0.2f, 0.3f, 0.3f, 0.5f)
    val bbox2 = NormalizedBox(0.1f, 0.1f, 0.4f, 0.6f)
    val overlap = BboxUtil.jaccardOverlap(bbox1, bbox2)
    assert(Math.abs(overlap - 2.0 / 15) < 1e-6)
  }

  "jaccardOverlap outside" should "work properly" in {
    val bbox1 = NormalizedBox(0.2f, 0.3f, 0.3f, 0.5f)
    val bbox2 = NormalizedBox(0f, 0f, 0.1f, 0.1f)
    val overlap = BboxUtil.jaccardOverlap(bbox1, bbox2)
    assert(Math.abs(overlap - 0) < 1e-6)
  }

  "projectBbox" should "work properly" in {
    val box1 = NormalizedBox(0.222159f, 0.427017f, 0.606492f, 0.679355f)
    val box2 = NormalizedBox(0.418f, 0.396396f, 0.55f, 0.666667f)
    val projBox = new NormalizedBox()

    val state = BboxUtil.projectBbox(box1, box2, projBox)
    state should be(true)
    assert(Math.abs(projBox.x1 - 0.509561f) < 1e-5)
    assert(Math.abs(projBox.y1 - 0f) < 1e-5)
    assert(Math.abs(projBox.x2 - 0.853014f) < 1e-5)
    assert(Math.abs(projBox.y2 - 0.949717f) < 1e-5)
  }

  "meetEmitCenterConstraint true" should "work properly" in {
    val box1 = NormalizedBox(0.222159f, 0.427017f, 0.606492f, 0.679355f)
    val box2 = NormalizedBox(0.418f, 0.396396f, 0.55f, 0.666667f)

    val state = BboxUtil.meetEmitCenterConstraint(box1, box2)

    state should be(true)
  }

  "meetEmitCenterConstraint false" should "work properly" in {
    val box1 = NormalizedBox(0.0268208f, 0.388175f, 0.394421f, 0.916685f)
    val box2 = NormalizedBox(0.418f, 0.396396f, 0.55f, 0.666667f)

    val state = BboxUtil.meetEmitCenterConstraint(box1, box2)

    state should be(false)
  }

  "getLocPredictions shared" should "work properly" in {
    val num = 2
    val numPredsPerClass = 2
    val numLocClasses = 1
    val shareLoc = true
    val dim = numPredsPerClass * numLocClasses * 4
    val loc = Tensor[Float](num, dim, 1, 1)

    val locData = loc.storage().array()
    (0 until num).foreach(i => {
      (0 until numPredsPerClass).foreach(j => {
        val idx = i * dim + j * 4
        locData(idx) = i * numPredsPerClass * 0.1f + j * 0.1f
        locData(idx + 1) = i * numPredsPerClass * 0.1f + j * 0.1f
        locData(idx + 2) = i * numPredsPerClass * 0.1f + j * 0.1f + 0.2f
        locData(idx + 3) = i * numPredsPerClass * 0.1f + j * 0.1f + 0.2f
      })
    })

    val out = BboxUtil.getLocPredictions(loc, numPredsPerClass, numLocClasses, shareLoc)

    assert(out.length == num)

    (0 until num).foreach(i => {
      assert(out(i).length == 1)
      val bboxes = out(i)(0)
      assert(bboxes.size(1) == numPredsPerClass)
      val startValue = i * numPredsPerClass * 0.1f
      var j = 0
      while (j < numPredsPerClass) {
        expectNear(bboxes(j + 1).valueAt(1), startValue + j * 0.1, 1e-6)
        expectNear(bboxes(j + 1).valueAt(2), startValue + j * 0.1, 1e-6)
        expectNear(bboxes(j + 1).valueAt(3), startValue + j * 0.1 + 0.2, 1e-6)
        expectNear(bboxes(j + 1).valueAt(4), startValue + j * 0.1 + 0.2, 1e-6)
        j += 1
      }
    })
  }

  def expectNear(v1: Float, v2: Double, eps: Double): Unit = {
    assert(Math.abs(v1 - v2) < eps)
  }

  "decodeBoxes" should "work properly" in {
    val priorBoxes = Tensor[Float](4, 4)
    val priorVariances = Tensor[Float](4, 4)
    val bboxes = Tensor[Float](4, 4)
    var i = 1
    while (i < 5) {
      priorBoxes.setValue(i, 1, 0.1f * i)
      priorBoxes.setValue(i, 2, 0.1f * i)
      priorBoxes.setValue(i, 3, 0.1f * i + 0.2f)
      priorBoxes.setValue(i, 4, 0.1f * i + 0.2f)

      priorVariances.setValue(i, 1, 0.1f)
      priorVariances.setValue(i, 2, 0.1f)
      priorVariances.setValue(i, 3, 0.2f)
      priorVariances.setValue(i, 4, 0.2f)

      bboxes.setValue(i, 1, 0f)
      bboxes.setValue(i, 2, 0.75f)
      bboxes.setValue(i, 3, Math.log(2).toFloat)
      bboxes.setValue(i, 4, Math.log(3f / 2).toFloat)
      i += 1
    }

    val decodedBboxes = BboxUtil.decodeBoxes(priorBoxes, priorVariances, false, bboxes, true)

    assert(decodedBboxes.size(1) == 4)

    i = 1
    while (i < 5) {
      expectNear(decodedBboxes.valueAt(i, 1), 0 + (i - 1) * 0.1, 1e-5)
      expectNear(decodedBboxes.valueAt(i, 2), 0.2 + (i - 1) * 0.1, 1e-5)
      expectNear(decodedBboxes.valueAt(i, 3), 0.4 + (i - 1) * 0.1, 1e-5)
      expectNear(decodedBboxes.valueAt(i, 4), 0.5 + (i - 1) * 0.1, 1e-5)
      i += 1
    }

  }

  //    "decodeSingleBbox" should "work properly" in {
  //      val priorBox = Tensor[Float](4)
  //      priorBox.setValue(1, 0.1f)
  //      priorBox.setValue(2, 0.1f)
  //      priorBox.setValue(3, 0.3f)
  //      priorBox.setValue(4, 0.3f)
  //
  //      val priorVariance = Tensor[Float](4)
  //      priorVariance.setValue(1, 0.1f)
  //      priorVariance.setValue(2, 0.1f)
  //      priorVariance.setValue(3, 0.2f)
  //      priorVariance.setValue(4, 0.2f)
  //
  //      val bbox = Tensor[Float](4)
  //      bbox.setValue(1, 0f)
  //      bbox.setValue(2, 0.75f)
  //      bbox.setValue(3, Math.log(2).toFloat)
  //      bbox.setValue(4, Math.log(3f / 2).toFloat)
  //
  //      val decodedBox = BboxUtil.decodeSingleBbox(priorBox, priorVariance, false, bbox, true)
  //
  //      expectNear(decodedBox.valueAt(1), 0, 1e-5)
  //      expectNear(decodedBox.valueAt(2), 0.2, 1e-5)
  //      expectNear(decodedBox.valueAt(3), 0.4, 1e-5)
  //      expectNear(decodedBox.valueAt(4), 0.5, 1e-5)
  //    }

  "getPriorVariance" should "work properly" in {
    val num_channels = 2
    val num_priors = 2
    val dim = num_priors * 4
    val prior = Tensor[Float](1, num_channels, dim, 1)
    val prior_data = prior.storage().array()
    for (i <- 0 until num_priors) {
      prior_data(i * 4) = i * 0.1f
      prior_data(i * 4 + 1) = i * 0.1f
      prior_data(i * 4 + 2) = i * 0.1f + 0.2f
      prior_data(i * 4 + 3) = i * 0.1f + 0.1f
      for (j <- 0 until 4) {
        prior_data(dim + i * 4 + j) = 0.1f
      }
    }

    val (boxes, variances) = BboxUtil.getPriorBboxes(prior, num_priors)
    assert(boxes.size(1) == num_priors)
    assert(variances.size(1) == num_priors)
    for (i <- 0 until num_priors) {
      expectNear(boxes.valueAt(i + 1, 1), i * 0.1, 1e-5)
      expectNear(boxes.valueAt(i + 1, 2), i * 0.1, 1e-5)
      expectNear(boxes.valueAt(i + 1, 3), i * 0.1 + 0.2, 1e-5)
      expectNear(boxes.valueAt(i + 1, 4), i * 0.1 + 0.1, 1e-5)
      expectNear(variances.valueAt(i + 1, 1), 0.1, 1e-5)
      expectNear(variances.valueAt(i + 1, 2), 0.1, 1e-5)
      expectNear(variances.valueAt(i + 1, 3), 0.1, 1e-5)
      expectNear(variances.valueAt(i + 1, 4), 0.1, 1e-5)
    }
  }

  "getGroundTruths" should "work properly" in {
    val input = Tensor(Storage(Array(
      0.0f, 1.0f, 0.14285715f, 0.1904762f, 0.23809524f, 0.2857143f, 0.33333334f,
      0.0f, 1.0f, 0.47619048f, 0.52380955f, 0.5714286f, 0.61904764f, 0.6666667f,
      1.0f, 3.0f, 0.8095238f, 0.85714287f, 0.9047619f, 0.95238096f, 1.0f
    ))).resize(3, 7)

    val gt0 = Tensor(Storage(Array(
      0.0f, 1.0f, 0.14285715f, 0.1904762f, 0.23809524f, 0.2857143f, 0.33333334f,
      0.0f, 1.0f, 0.47619048f, 0.52380955f, 0.5714286f, 0.61904764f, 0.6666667f
    ))).resize(2, 7)

    val gt1 = Tensor(Storage(Array(
      1.0f, 3.0f, 0.8095238f, 0.85714287f, 0.9047619f, 0.95238096f, 1.0f
    ))).resize(1, 7)

    val gts = BboxUtil.getGroundTruths(input)

    gts(0) should be(gt0)
    gts(1) should be(gt1)

    val gts2 = BboxUtil.getGroundTruths(gt1)

    gts2(0) should be(gt1)

    val label = Tensor(Storage(Array(
      3.0, 8.0, 0.0, 0.241746, 0.322738, 0.447184, 0.478388,
      3.0, 8.0, 0.0, 0.318659, 0.336546, 0.661729, 0.675461,
      3.0, 8.0, 0.0, 0.56154, 0.300144, 0.699173, 0.708098,
      3.0, 8.0, 0.0, 0.220494, 0.327759, 0.327767, 0.396797,
      3.0, 8.0, 0.0, 0.194182, 0.317717, 0.279191, 0.389266,
      4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
      5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
      6.0, 10.0, 0.0, 0.67894, 0.471823, 0.929308, 0.632044,
      6.0, 10.0, 0.0, 0.381443, 0.572376, 0.892489, 0.691713,
      7.0, 9.0, 0.0, 0.0, 0.0620616, 0.667269, 1.0
    ).map(_.toFloat))).resize(10, 7)


    val labelgt = BboxUtil.getGroundTruths(label)

    labelgt.size should be(3)
    labelgt(0).size(1) should be(5)
    labelgt(0).valueAt(1, 1) should be(3)
    labelgt(3).size(1) should be(2)
    labelgt(3).valueAt(1, 1) should be(6)
    labelgt(4).size(1) should be(1)
    labelgt(4).valueAt(1, 1) should be(7)
  }

  "decodeRoi" should "work properly" in {
    val tensor = Tensor[Float](T(0))
    println(tensor)
    BboxUtil.decodeRois(tensor, 21)
  }

  "vertex" should "work" in {
    def vertcat[T: ClassTag](tensors: Tensor[T]*)(implicit ev: TensorNumeric[T]): Tensor[T] = {
      require(tensors(0).dim() == 2, "currently only support 2D")

      def getRowCol(tensor: Tensor[T]): (Int, Int) = {
        if (tensors(0).nDimension() == 2) {
          (tensor.size(1), tensor.size(2))
        } else {
          (1, tensor.size(1))
        }
      }

      var nRows = getRowCol(tensors(0))._1
      val nCols = getRowCol(tensors(0))._2
      for (i <- 1 until tensors.length) {
        require(getRowCol(tensors(i))._2 == nCols, "the cols length must be equal")
        nRows += getRowCol(tensors(i))._1
      }
      val resData = Tensor[T](nRows, nCols)
      var id = 0
      tensors.foreach { tensor =>
        if (tensor.nDimension() == 1) {
          id = id + 1
          resData.update(id, tensor)
        } else {
          (1 to getRowCol(tensor)._1).foreach(rid => {
            id = id + 1
            resData.update(id, tensor(rid))
          })
        }
      }
      resData
    }

    val tensor = Tensor[Float](2, 3).randn()
    val tensor2 = Tensor[Float](4, 3).randn()
    val tensor3 = Tensor[Float](3, 3).randn()

    val res = vertcat[Float](tensor, tensor2, tensor3)
    val res2 = BboxUtil.vertcat2D[Float](tensor, tensor2, tensor3)

    res should equal(res2)
  }

  "bboxTransform" should "work" in {

    def selectCol(mat: Tensor[Float], cid: Int): Tensor[Float] = {
      if (mat.nElement() == 0) return Tensor[Float](0)
      mat.select(2, cid)
    }
    def bboxTransform(sampleRois: Tensor[Float], gtRois: Tensor[Float]): Tensor[Float] = {
      val exWidths = sampleRois.select(2, 3) - sampleRois.select(2, 1) + 1.0f
      val exHeights = sampleRois.select(2, 4) - sampleRois.select(2, 2) + 1.0f
      val exCtrX = sampleRois.select(2, 1) + exWidths * 0.5f
      val exCtrY = sampleRois.select(2, 2) + exHeights * 0.5f

      val gtWidths = selectCol(gtRois, 3) - selectCol(gtRois, 1) + 1.0f
      val gtHeights = selectCol(gtRois, 4) - selectCol(gtRois, 2) + 1.0f
      val gtCtrX = selectCol(gtRois, 1) + gtWidths * 0.5f
      val gtCtrY = selectCol(gtRois, 2) + gtHeights * 0.5f

      val targetsDx = (gtCtrX - exCtrX) / exWidths
      val targetsDy = (gtCtrY - exCtrY) / exHeights
      val targetsDw = gtWidths.cdiv(exWidths).log()
      val targetsDh = gtHeights.cdiv(exHeights).log()

      val res = BboxUtil.vertcat1D(targetsDx, targetsDy, targetsDw, targetsDh)
      res.t().contiguous()
    }

    val t1 = Tensor[Float](8, 4).rand()
    val t2 = Tensor[Float](8, 4).rand()

    val res = bboxTransform(t1.clone(), t2.clone())
    val res2 = BboxUtil.bboxTransform(t1.clone(), t2.clone())
    res should be(res2)
  }

  "getBboxRegressionLabels" should "work" in {
    def getBboxRegressionLabels(bboxTargetData: Tensor[Float],
      numClasses: Int): (Tensor[Float], Tensor[Float]) = {

      // Deprecated (inside weights)
      val BBOX_INSIDE_WEIGHTS = Tensor(Storage(Array(1.0f, 1.0f, 1.0f, 1.0f)))
      val label = bboxTargetData.select(2, 1).clone().storage().array()
      val bbox_targets = Tensor[Float](label.length, 4 * numClasses)
      val bbox_inside_weights = Tensor[Float]().resizeAs(bbox_targets)
      val inds = label.zipWithIndex.filter(x => x._1 > 0).map(x => x._2)
      inds.foreach(ind => {
        val cls = label(ind)
        val start = 4 * cls
        (2 to bboxTargetData.size(2)).foreach(x => {
          bbox_targets.setValue(ind + 1, x + start.toInt - 1, bboxTargetData.valueAt(ind + 1, x))
          bbox_inside_weights.setValue(ind + 1, x + start.toInt - 1,
            BBOX_INSIDE_WEIGHTS.valueAt(x - 1))
        })
        println()
      })
      (bbox_targets, bbox_inside_weights)
    }

    val input = Tensor[Float](200, 5).randn()

    // Deprecated (inside weights)
//    val BBOX_INSIDE_WEIGHTS = Tensor(Storage(Array(1.0f, 1.0f, 1.0f, 1.0f)))
    (1 to 10).foreach(i => input.setValue(i, 1, i - 1))
    val res = getBboxRegressionLabels(input, 21)
    val res2 = BboxUtil.getBboxRegressionLabels(input, 21)

    res should be(res2)
  }

  "apply1" should "work" in {
    def a1(bboxInsideWeights: Tensor[Float]): Tensor[Float] = {
      for (r <- 1 to bboxInsideWeights.size(1)) {
        for (c <- 1 to bboxInsideWeights.size(2)) {
          if (bboxInsideWeights.valueAt(r, c) > 0) {
            bboxInsideWeights.setValue(r, c, 1f)
          } else {
            bboxInsideWeights.setValue(r, c, 0f)
          }
        }
      }
      bboxInsideWeights
    }
    def a2(bboxInsideWeights: Tensor[Float]): Tensor[Float] = {
      bboxInsideWeights.apply1(x => {
        if (x > 0) 1f else 0f
      })
    }
    val tensor = Tensor[Float](30, 40).rand(-1, 1)
    val tensor2 = tensor.clone()

    val out1 = a1(tensor)
    val out2 = a2(tensor2)
    out1 should be (out2)
  }
}
