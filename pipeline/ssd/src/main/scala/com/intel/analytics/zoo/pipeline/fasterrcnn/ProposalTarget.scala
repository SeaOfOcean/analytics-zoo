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

package com.intel.analytics.zoo.pipeline.fasterrcnn

import breeze.numerics.round
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.pipeline.common.BboxUtil
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.FasterRcnnParam
import org.apache.log4j.Logger

import scala.util.Random

object ProposalTarget {
  val logger = Logger.getLogger(getClass)
  def apply(param: FasterRcnnParam, numClasses: Int)
    (implicit ev: TensorNumeric[Float]): ProposalTarget = new ProposalTarget(param, numClasses)
}

/**
 * Assign object detection proposals to ground-truth targets. Produces proposal
 * classification labels and bounding-box regression targets.
 */
class ProposalTarget(param: FasterRcnnParam, numClasses: Int)
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Table, Float] {

  //  val labelTarget = T()
//  val target = T()
//  @transient var bboxTool: Bbox = new Bbox

  /**
   * Compute bounding-box regression targets for an image.
   *
   */
  def computeTargets(sampledRois: Tensor[Float],
    gtRois: Tensor[Float],
    labels: Tensor[Float]): Tensor[Float] = {

    val targets = BboxUtil.bboxTransform(sampledRois, gtRois)

    if (param.BBOX_NORMALIZE_TARGETS_PRECOMPUTED) {
      // Optionally normalize targets by a precomputed mean and stdev
      for (r <- 1 to targets.size(1)) {
        targets(r).add(-1, param.BBOX_NORMALIZE_MEANS)
        targets(r).cdiv(param.BBOX_NORMALIZE_STDS)
      }
    }
    BboxUtil.horzcat(labels.resize(labels.nElement(), 1), targets)
  }




  // Fraction of minibatch that is labeled foreground (i.e. class > 0)
  val FG_FRACTION = 0.25
  val rois_per_image = param.BATCH_SIZE
  val fgRoisPerImage = round(FG_FRACTION * param.BATCH_SIZE).toInt

  var fgRoisPerThisImage = 0
  var bg_rois_per_this_image = 0

  // Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
  val FG_THRESH = 0.5f
  private def selectForeGroundRois(maxOverlaps: Tensor[Float]): Array[Int] = {
    // Select foreground RoIs as those with >= FG_THRESH overlap
    var fgInds = maxOverlaps.storage().array().zip(Stream from 1)
      .filter(x => x._1 >= FG_THRESH).map(x => x._2)
    // Guard against the case when an image has fewer than fg_rois_per_image
    // foreground RoIs
    fgRoisPerThisImage = Math.min(fgRoisPerImage, fgInds.length)
    // Sample foreground regions without replacement
    if (fgInds.length > 0) {
      fgInds = Random.shuffle(fgInds.toList).slice(0, fgRoisPerThisImage).toArray
    }
    fgInds
  }

  def selectBackgroundRois(max_overlaps: Tensor[Float]): Array[Int] = {
    // Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    var bg_inds = max_overlaps.storage().array().zip(Stream from 1)
      .filter(x => (x._1 < param.BG_THRESH_HI) && (x._1 >= param.BG_THRESH_LO))
      .map(x => x._2)
    // Compute number of background RoIs to take from this image (guarding
    // against there being fewer than desired)
    bg_rois_per_this_image = Math.min(rois_per_image - fgRoisPerThisImage, bg_inds.length)
    // Sample background regions without replacement
    if (bg_inds.length > 0) {
      bg_inds = Random.shuffle(bg_inds.toList).slice(0, bg_rois_per_this_image).toArray
    }
    bg_inds
  }


  /**
   * Generate a random sample of RoIs comprising foreground and background examples.
   * @param roisPlusGts (0, x1, y1, x2, y2)
   * @param gts GT boxes (index, label, difficult, x1, y1, x2, y2)
   * @return
   */
  def sampleRois(roisPlusGts: Tensor[Float],
    gts: Tensor[Float])
  : (Tensor[Float], Tensor[Float], Tensor[Float], Tensor[Float]) = {
    // overlaps: (rois x gt_boxes)
    val overlaps = BboxUtil.bboxOverlap(roisPlusGts.narrow(2, 2, 4), FrcnnMiniBatch.getBboxes(gts))

//    val overlaps2 = BboxUtil.bboxOverlap(BboxUtil.selectMatrix(roisPlusGts, Array.range(2, 6), 2),
//      BboxUtil.selectMatrix(gts, Array.range(4, 8), 2))

    // for each roi, get the gt with max overlap with it
    val (maxOverlaps, gtIndices) = overlaps.max(2)
    // todo: the last few overlap should be 1, they are gt overlap gt

//    var labels = BboxUtil.selectMatrix2(gts, gtIndices.storage().array()
//      .map(x => x.toInt),
//      Array(FrcnnMiniBatch.labelIndex)).squeeze().clone()

    // labels for rois
    var labels = Tensor[Float](gtIndices.nElement())
    (1 to gtIndices.nElement()).foreach(i => {
//      println("proposal target " + i, gts.size().mkString("x"), FrcnnMiniBatch.labelIndex,
//        gtIndices.size().mkString("x"))
      labels.setValue(i, gts.valueAt(gtIndices.valueAt(i, 1).toInt, FrcnnMiniBatch.labelIndex))
    })

//    // todo: optimize this
//    var labels2 = BboxUtil.selectMatrix2(gts, gtIndices.storage().array()
//      .map(x => x.toInt),
//      Array(FrcnnMiniBatch.labelIndex)).squeeze().clone()


    // from max overlaps, select foreground and background
    val fgInds = selectForeGroundRois(maxOverlaps)
    val bg_inds = selectBackgroundRois(maxOverlaps)
    // for test usage
    // fg_inds = FileUtil.loadFeatures("fg_inds_choice").storage().array().map(x => x.toInt + 1)
    // bg_inds = FileUtil.loadFeatures("bg_inds_choice").storage().array().map(x => x.toInt + 1)

    // The indices that we're selecting (both fg and bg)
    val keepInds = fgInds ++ bg_inds

    // Select sampled values from various arrays:
    labels = BboxUtil.selectMatrix(labels, keepInds, 1)
    // Clamp labels for the background RoIs to 0
    (fgRoisPerThisImage + 1 to labels.nElement()).foreach(i => labels(i) = 0)

    val sampledRois = BboxUtil.selectMatrix(roisPlusGts, keepInds, 1)
    val keepInds2 = keepInds.map(x => gtIndices.valueAt(x, 1).toInt)

    val bboxTargetData = computeTargets(
      sampledRois.narrow(2, 2, 4),
      BboxUtil.selectMatrix(gts, keepInds2, 1)
        .narrow(2, FrcnnMiniBatch.x1Index, 4),
      labels)

    val (bboxTarget, bboxInsideWeights) =
      BboxUtil.getBboxRegressionLabels(bboxTargetData, numClasses)
    (labels.squeeze(), sampledRois, bboxTarget, bboxInsideWeights)
  }


  override def updateOutput(input: Table): Table = {
    if (!isTraining()) {
      output.insert(1, input(1))
      output.insert(2, input(2))
      return output
    }

    // Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    val proposalRois = input[Tensor[Float]](1)
    // GT boxes (index, label, difficult, x1, y1, x2, y2)
    val gts = input[Tensor[Float]](2)

    // Include ground-truth boxes in the set of candidate rois
    val roisPlusGts = BboxUtil.vertcat2D(proposalRois, gts.narrow(2, 3, 5))
    // in case gts has difficult (1)
    roisPlusGts.select(2, 1).fill(0)

    // Sample rois with classification labels and bounding box regression
    // targets
    val (labels, rois, bbox_targets, bboxInsideWeights) = sampleRois(roisPlusGts, gts)

    labels.apply1(x => if (x == -1) -1 else x + 1f)
    if (output.length() == 0) {
      // bbox_targets (1, numClasses * 4) + bbox_inside_weights (1, numClasses * 4)
      // + bbox_outside_weights (1, numClasses * 4)

//      for (r <- 1 to bboxInsideWeights.size(1)) {
//        for (c <- 1 to bboxInsideWeights.size(2)) {
//          if (bboxInsideWeights.valueAt(r, c) > 0) {
//            bboxInsideWeights.setValue(r, c, 1f)
//          } else {
//            bboxInsideWeights.setValue(r, c, 0f)
//          }
//        }
//      }
      bboxInsideWeights.apply1(x => {
        if (x > 0) 1f else 0f
      })
    }

    // sampled rois (0, x1, y1, x2, y2) (1,5)
    output.insert(1, rois)
//    println(rois)
    // labels (1,1)
    output.insert(2, labels)
    output.insert(3, bbox_targets)
    output.insert(4, bboxInsideWeights)
    output.insert(5, bboxInsideWeights)
    output
  }


//  def matrix2Table(mat1: Tensor[Float], mat2: Tensor[Float],
//    mat3: Tensor[Float]): Table = {
//    target.insert(1, mat1)
//    target.insert(2, mat2)
//    target.insert(3, mat3)
//    target
//  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = null
    gradInput
  }
}
