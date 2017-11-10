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

package com.intel.analytics.zoo.pipeline.common.nn

import breeze.numerics.round
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.common.BboxUtil
import com.intel.analytics.zoo.pipeline.fasterrcnn.FrcnnMiniBatch
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


  var debug = false
  // Fraction of minibatch that is labeled foreground (i.e. class > 0)
  val FG_FRACTION = 0.25
  val rois_per_image = param.BATCH_SIZE
  val fgRoisPerImage = round(FG_FRACTION * param.BATCH_SIZE).toInt

  var fgRoisPerThisImage = 0
  var bgRoisPerThisImage = 0

  // Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
  val FG_THRESH = 0.5f

  private def selectForeGroundRois(maxOverlaps: Tensor[Float]): Array[Int] = {
    // Select foreground RoIs as those with >= FG_THRESH overlap
    var fgInds = Array.range(1, maxOverlaps.nElement() + 1)
      .filter(i => maxOverlaps.valueAt(i, 1) >= FG_THRESH)
//    var fgInds = maxOverlaps.storage().array().zip(Stream from 1)
//      .filter(x => x._1 >= FG_THRESH).map(x => x._2)
    // Guard against the case when an image has fewer than fg_rois_per_image
    // foreground RoIs
    fgRoisPerThisImage = Math.min(fgRoisPerImage, fgInds.length)
    // Sample foreground regions without replacement
    if (fgInds.length > 0) {
      fgInds = if (debug) {
        fgInds.toList.slice(0, fgRoisPerThisImage).toArray
      } else {
        Random.shuffle(fgInds.toList).slice(0, fgRoisPerThisImage).toArray
      }
    }
    fgInds
  }

  def selectBackgroundRois(maxOverlaps: Tensor[Float]): Array[Int] = {
    // Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    var bgInds = Array.range(1, maxOverlaps.nElement() + 1)
      .filter(i => (maxOverlaps.valueAt(i, 1) < param.BG_THRESH_HI) &&
        (maxOverlaps.valueAt(i, 1) >= param.BG_THRESH_LO))
//    var bgInds = maxOverlaps.storage().array().zip(Stream from 1)
//      .filter(x => (x._1 < param.BG_THRESH_HI) && (x._1 >= param.BG_THRESH_LO))
//      .map(x => x._2)
    // Compute number of background RoIs to take from this image (guarding
    // against there being fewer than desired)
    bgRoisPerThisImage = Math.min(rois_per_image - fgRoisPerThisImage, bgInds.length)
    // Sample background regions without replacement
    if (bgInds.length > 0) {
      bgInds = if (debug) {
        bgInds.toList.slice(0, bgRoisPerThisImage).toArray
      } else {
        Random.shuffle(bgInds.toList).slice(0, bgRoisPerThisImage).toArray
      }
    }
    bgInds
  }


  /**
   * Generate a random sample of RoIs comprising foreground and background examples.
   *
   * @param roisPlusGts (0, x1, y1, x2, y2)
   * @param gts GT boxes (index, label, difficult, x1, y1, x2, y2)
   * @return
   */
  def sampleRois(roisPlusGts: Tensor[Float],
    gts: Tensor[Float])
  : (Tensor[Float], Tensor[Float], Tensor[Float], Tensor[Float]) = {
    // overlaps: (rois x gt_boxes)
    val overlaps = BboxUtil.bboxOverlap(roisPlusGts.narrow(2, 2, 4), FrcnnMiniBatch.getBboxes(gts))

    // for each roi, get the gt with max overlap with it
    val (maxOverlaps, gtIndices) = overlaps.max(2)
    // todo: the last few overlap should be 1, they are gt overlap gt

    // labels for rois
    var labels = Tensor[Float](gtIndices.nElement())
    (1 to gtIndices.nElement()).foreach(i => {
      val cls = gts.valueAt(gtIndices.valueAt(i, 1).toInt, FrcnnMiniBatch.labelIndex)
      require(cls >= 1 && cls <= numClasses, s"$cls is not in range [1, $numClasses]")
      labels.setValue(i, cls)
    })



    // from max overlaps, select foreground and background
    val fgInds = selectForeGroundRois(maxOverlaps)
    val bg_inds = selectBackgroundRois(maxOverlaps)
    // The indices that we're selecting (both fg and bg)
    val keepInds = fgInds ++ bg_inds

    // Select sampled values from various arrays:
    labels = BboxUtil.selectMatrix(labels, keepInds, 1)
    // Clamp labels for the background RoIs to 1 (1-based)
    (fgRoisPerThisImage + 1 to labels.nElement()).foreach(i => labels(i) = 1)

    val sampledRois = BboxUtil.selectMatrix(roisPlusGts, keepInds, 1)
    val keepInds2 = keepInds.map(x => gtIndices.valueAt(x, 1).toInt)

    val bboxTargetData = computeTargets(
      sampledRois.narrow(2, 2, 4),
      BboxUtil.selectMatrix(gts, keepInds2, 1)
        .narrow(2, FrcnnMiniBatch.x1Index, 4),
      labels)

//    println("bboxTargetData =============", Tensor.sparse(bboxTargetData))

    val (bboxTarget, bboxInsideWeights) =
      BboxUtil.getBboxRegressionLabels(bboxTargetData, numClasses)
//    printSparse(bboxTarget)
//    println(Tensor.sparse(bboxTarget))
//    println(Tensor.sparse(bboxInsideWeights))
    (labels.squeeze(), sampledRois, bboxTarget, bboxInsideWeights)
  }

  override def updateOutput(input: Table): Table = {
    if (!isTraining()) {
      output(1) = input(1)
      output(2) = input(2)
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

    if (output.length() == 0) {
      // bbox_targets (1, numClasses * 4) + bbox_inside_weights (1, numClasses * 4)
      // + bbox_outside_weights (1, numClasses * 4)
      bboxInsideWeights.apply1(x => {
        if (x > 0) 1f else 0f
      })
    }

    // sampled rois (0, x1, y1, x2, y2) (1,5)
    output.update(1, rois)
    // labels (1,1)
    output.update(2, labels)
    output.update(3, bbox_targets)
    output.update(4, bboxInsideWeights)
    output.update(5, bboxInsideWeights)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = null
    gradInput
  }
}
