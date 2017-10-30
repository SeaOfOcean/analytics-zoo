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

package com.intel.analytics.zoo.pipeline.fasterrcnn

import org.scalatest.{FlatSpec, Matchers}

class AnchorTargetSpec extends FlatSpec with Matchers {
//  TestUtil.middleRoot = "/home/jxy/data/middle/vgg16/step1/"
//  val param = new VggParam()
//  val imdb = Imdb.getImdb("voc_2007_testcode1", "$home/data/VOCdevkit")
//  val roidb = imdb.loadAnnotation("000014")
//  val reader = LocalByteRoiImageReader()
//  val resizer = RoiImageResizer(param.SCALES, param.SCALE_MULTIPLE_OF, true)
//  val normalizer = RoiImageNormalizer((122.7717f, 115.9465f, 102.9801f))
//  val img = normalizer.transform(resizer.transform(reader.transform(roidb)))
//  println(img.target)
//  val anchorTarget = new AnchorTarget[Float](param)
//  val insideAnchorsGtOverlaps =
//    TestUtil.loadFeatures("insideAnchorsGtOverlaps", "$home/data/middle/vgg16/step1")
//  val featureW = 57
//  val featureH = 38
//  val (indsInside, insideAnchors, totalAnchors) =
//    anchorTarget.getAnchors(featureW, featureH, 901, 600)
//
//  "getAllLabels" should "work properly" in {
//    val labels = anchorTarget.getAllLabels(indsInside, insideAnchorsGtOverlaps)
//    compare("labelbeforesample", labels, 1e-6)
//  }
//
//  "computeTargets" should "work properly" in {
//    val gtBoxes = img.target.bboxes
//    val targets = anchorTarget.computeTargets(insideAnchors, gtBoxes, insideAnchorsGtOverlaps)
//    // todo: precision may not be enough
//    compare("targetBefore3354", targets, 0.01)
//  }
//
//  "get weights" should "work properly" in {
//    val labels = TestUtil.loadFeatures("labelsBefore3354")
//    val bboxInsideWeights = anchorTarget.getBboxInsideWeights(indsInside, labels)
//    compare("inwBefore3354", bboxInsideWeights, 1e-6)
//    val bboxOutSideWeights = anchorTarget.getBboxOutsideWeights(indsInside, labels)
//    compare("outWBefore3354", bboxOutSideWeights, 1e-6)
//  }
//
//  "unmap" should "work properly" in {
//    var labels = TestUtil.loadFeatures("labelsBefore3354")
//    labels = anchorTarget.unmap(labels, totalAnchors, indsInside, -1)
//    compare("labelUnmap", labels, 1e-6)
//  }
//
//  def compare(name: String, vec: Tensor[Float], prec: Double): Unit = {
//    val exp = TestUtil.loadFeatures(name, "$home/data/middle/vgg16/step1")
//    TestUtil.assertEqualIgnoreSize(exp, vec, name, prec)
//  }
}
