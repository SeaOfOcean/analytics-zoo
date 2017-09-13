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

import scala.util.Random

object ProposalTarget {
  def apply(param: FasterRcnnParam, numClasses: Int)
    (implicit ev: TensorNumeric[Float]): ProposalTarget = new ProposalTarget(param, numClasses)
}

/**
 * Assign object detection proposals to ground-truth targets. Produces proposal
 * classification labels and bounding-box regression targets.
 */
class ProposalTarget(param: FasterRcnnParam, numClasses: Int)
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Table, Float] {

  @transient val labelTarget = T()
  @transient val target = T()
//  @transient var bboxTool: Bbox = new Bbox

  /**
   * Compute bounding-box regression targets for an image.
   *
   */
  def computeTargets(ex_rois: Tensor[Float],
    gt_rois: Tensor[Float],
    labels: Tensor[Float]): Tensor[Float] = {
    require(ex_rois.size(1) == gt_rois.size(1))
    require(ex_rois.size(2) == 4)
    require(gt_rois.size(2) == 4)

    val targets = BboxUtil.bboxTransform(ex_rois, gt_rois)

    if (param.BBOX_NORMALIZE_TARGETS_PRECOMPUTED) {
      // Optionally normalize targets by a precomputed mean and stdev
      for (r <- 1 to targets.size(1)) {
        targets(r).add(-1, param.BBOX_NORMALIZE_MEANS)
        targets(r).cdiv(param.BBOX_NORMALIZE_STDS)
      }
    }
    BboxUtil.horzcat(labels.resize(labels.nElement(), 1), targets)
  }

  /**
   * Bounding-box regression targets (bbox_target_data) are stored in a
   * compact form N x (class, tx, ty, tw, th)
   * *
   * This function expands those targets into the 4-of-4*K representation used
   * by the network (i.e. only one class has non-zero targets).
   * *
   * Returns:
   * bbox_target (ndarray): N x 4K blob of regression targets
   * bbox_inside_weights (ndarray): N x 4K blob of loss weights
   *
   */
  def getBboxRegressionLabels(bbox_target_data: Tensor[Float],
    numClasses: Int): (Tensor[Float], Tensor[Float]) = {
    val clss = BboxUtil.selectCol(bbox_target_data, 1).clone().storage().array()
    val bbox_targets = Tensor[Float](clss.length, 4 * numClasses)
    val bbox_inside_weights = Tensor[Float]().resizeAs(bbox_targets)
    val inds = clss.zipWithIndex.filter(x => x._1 > 0).map(x => x._2)
    inds.foreach(ind => {
      val cls = clss(ind)
      val start = 4 * cls
      (2 to bbox_target_data.size(2)).foreach(x => {
        bbox_targets.setValue(ind + 1, x + start.toInt - 1, bbox_target_data.valueAt(ind + 1, x))
        bbox_inside_weights.setValue(ind + 1, x + start.toInt - 1,
          param.BBOX_INSIDE_WEIGHTS.valueAt(x - 1))
      })
    })
    (bbox_targets, bbox_inside_weights)
  }

  val rois_per_image = param.BATCH_SIZE
  val fg_rois_per_image = round(param.FG_FRACTION * param.BATCH_SIZE).toInt

  var fg_rois_per_this_image = 0
  var bg_rois_per_this_image = 0

  def selectForeGroundRois(max_overlaps: Tensor[Float]): Array[Int] = {
    // Select foreground RoIs as those with >= FG_THRESH overlap
    var fg_inds = findAllGeInds(max_overlaps, param.FG_THRESH)
    // Guard against the case when an image has fewer than fg_rois_per_image
    // foreground RoIs
    fg_rois_per_this_image = Math.min(fg_rois_per_image, fg_inds.length)
    // Sample foreground regions without replacement
    if (fg_inds.length > 0) {
      fg_inds = Random.shuffle(fg_inds.toList).slice(0, fg_rois_per_this_image).toArray
    }
    fg_inds
  }

  def selectBackgroundRois(max_overlaps: Tensor[Float]): Array[Int] = {
    // Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    var bg_inds = max_overlaps.storage().array().zip(Stream from 1)
      .filter(x => (x._1 < param.BG_THRESH_HI) && (x._1 >= param.BG_THRESH_LO))
      .map(x => x._2)
    // Compute number of background RoIs to take from this image (guarding
    // against there being fewer than desired)
    bg_rois_per_this_image = Math.min(rois_per_image - fg_rois_per_this_image, bg_inds.length)
    // Sample background regions without replacement
    if (bg_inds.length > 0) {
      bg_inds = Random.shuffle(bg_inds.toList).slice(0, bg_rois_per_this_image).toArray
    }
    bg_inds
  }


  // Generate a random sample of RoIs comprising foreground and background examples.
  def sampleRois(all_rois: Tensor[Float],
    gt_boxes: Tensor[Float])
  : (Tensor[Float], Tensor[Float], Tensor[Float], Tensor[Float]) = {
    // overlaps: (rois x gt_boxes)
    val overlaps = BboxUtil.bboxOverlap(BboxUtil.selectMatrix(all_rois, Array.range(2, 6), 2),
      BboxUtil.selectMatrix(gt_boxes, Array.range(1, 5), 2))

    val (max_overlaps, gt_assignment) = overlaps.max(2)

    var labels = BboxUtil.selectMatrix2(gt_boxes, gt_assignment.storage().array()
      .map(x => x.toInt),
      Array(5)).squeeze().clone()

    val fg_inds = selectForeGroundRois(max_overlaps)
    val bg_inds = selectBackgroundRois(max_overlaps)
    // for test usage
    // fg_inds = FileUtil.loadFeatures("fg_inds_choice").storage().array().map(x => x.toInt + 1)
    // bg_inds = FileUtil.loadFeatures("bg_inds_choice").storage().array().map(x => x.toInt + 1)
    // The indices that we're selecting (both fg and bg)
    val keep_inds = fg_inds ++ bg_inds

    // Select sampled values from various arrays:
    labels = BboxUtil.selectMatrix(labels, keep_inds, 1)
    // Clamp labels for the background RoIs to 0
    (fg_rois_per_this_image + 1 to labels.nElement()).foreach(i => labels(i) = 0)

    val rois = BboxUtil.selectMatrix(all_rois, keep_inds, 1)
    val keepInds2 = keep_inds.map(x => gt_assignment.valueAt(x, 1).toInt)
    val bbox_target_data = computeTargets(
      BboxUtil.selectMatrix(rois, Array.range(2, 6), 2),
      BboxUtil.selectMatrix(BboxUtil.selectMatrix(gt_boxes, keepInds2, 1),
        Array.range(1, 5), 2), labels)


    val (bbox_targets, bbox_inside_weights) =
      getBboxRegressionLabels(bbox_target_data, numClasses)
    (labels.squeeze(), rois, bbox_targets, bbox_inside_weights)
  }

  def findAllGeInds(tensor: Tensor[Float], thresh: Float): Array[Int] = {
    tensor.storage().array().zip(Stream from 1)
      .filter(x => x._1 >= param.FG_THRESH).map(x => x._2)
  }

  override def updateOutput(input: Table): Table = {
    if (!isTraining()) {
      output.insert(1, input(1))
      output.insert(2, input(2))
      return output
    }

    // Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    // (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    val all_roisTen = input[Tensor[Float]](1)
    // GT boxes (x1, y1, x2, y2, label)
    // and other times after box coordinates -- normalize to one format
    val gt_boxes = input[Tensor[Float]](2)

    // Include ground-truth boxes in the set of candidate rois
    val zeros = Tensor[Float](gt_boxes.size(1), 1)
    val all_rois = BboxUtil.vertcat(all_roisTen,
      BboxUtil.horzcat(zeros, BboxUtil.selectMatrix(gt_boxes,
        Array.range(1, gt_boxes.size(2)), 2)))

    // Sample rois with classification labels and bounding box regression
    // targets
    val (labels, rois, bbox_targets, bbox_inside_weights) = sampleRois(all_rois, gt_boxes)

    labels.apply1(x => if (x == -1) -1 else x + 1f)
    if (output.length() == 0) {
      // sampled rois (0, x1, y1, x2, y2) (1,5)
      output.insert(rois)
      // labels (1,1)
      labelTarget.insert(labels)
      // bbox_targets (1, numClasses * 4) + bbox_inside_weights (1, numClasses * 4)
      // + bbox_outside_weights (1, numClasses * 4)

      for (r <- 1 to bbox_inside_weights.size(1)) {
        for (c <- 1 to bbox_inside_weights.size(2)) {
          if (bbox_inside_weights.valueAt(r, c) > 0) {
            bbox_inside_weights.setValue(r, c, 1f)
          } else {
            bbox_inside_weights.setValue(r, c, 0f)
          }
        }
      }

      labelTarget.insert(matrix2Table(bbox_targets, bbox_inside_weights,
        bbox_inside_weights))
      output.insert(labelTarget)
    } else {
      output.update(1, rois)
      labelTarget.update(1, labels)
      labelTarget.update(2, matrix2Table(bbox_targets, bbox_inside_weights,
        bbox_inside_weights))
    }
    output
  }


  def matrix2Table(mat1: Tensor[Float], mat2: Tensor[Float],
    mat3: Tensor[Float]): Table = {
    if (target.length() == 0) {
      target.insert(mat1)
      target.insert(mat2)
      target.insert(mat3)
    } else {
      target.update(1, mat1)
      target.update(2, mat2)
      target.update(3, mat3)
    }
    target
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = gradOutput
    gradInput
  }
}
