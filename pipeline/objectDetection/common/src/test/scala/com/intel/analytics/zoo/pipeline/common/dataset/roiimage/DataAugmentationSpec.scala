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

package com.intel.analytics.zoo.pipeline.common.dataset.roiimage

import java.io.File

import com.intel.analytics.zoo.pipeline.common.dataset.{Imdb, LocalByteRoiimageReader}
import com.intel.analytics.zoo.transform.vision.image.augmentation._
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, MatToFloats, RandomTransformer}
import com.intel.analytics.zoo.transform.vision.label.roi._
import org.opencv.core.{Mat, Point, Scalar}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class DataAugmentationSpec extends FlatSpec with Matchers with BeforeAndAfter {
  def visulize(label: RoiLabel, mat: Mat, normalized: Boolean = true): Unit = {
    var i = 1
    val scaleW = if (normalized) mat.width() else 1
    val scaleH = if (normalized) mat.height() else 1
    while (label.bboxes.nElement() > 0 && i <= label.bboxes.size(1)) {
      Imgproc.rectangle(mat, new Point(label.bboxes.valueAt(i, 1) * scaleW,
        label.bboxes.valueAt(i, 2) * scaleH),
        new Point(label.bboxes.valueAt(i, 3) * scaleW,
          label.bboxes.valueAt(i, 4) * scaleH),
        new Scalar(0, 255, 0))
      i += 1
    }
  }

  "ImageAugmentation" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("VOCdevkit")
    val voc = Imdb.getImdb("voc_2007_testcode", resource.getPath)
    val roidb = voc.getRoidb().toIterator
    val imgAug = LocalByteRoiimageReader() -> RecordToFeature(true) ->
      BytesToMat() ->
      RoiNormalize() ->
      ColorJitter() ->
      RandomTransformer(Expand() -> RoiExpand(), 0.5) ->
      RandomSampler() ->
      Resize(300, 300, -1) ->
      RandomTransformer(HFlip() -> RoiHFlip(), 0.5) ->
      MatToFloats(validHeight = 300, validWidth = 300, meanRGB = Some(123f, 117f, 104f))
    val out = imgAug(roidb)
    out.foreach(img => {
      val tmpFile = java.io.File.createTempFile("module", ".jpg")
      val mat = OpenCVMat.floatToMat(img.getFloats(), img.getHeight(), img.getWidth())
      visulize(img.getLabel[RoiLabel], mat)
      Imgcodecs.imwrite(tmpFile.getAbsolutePath, mat)
      println(s"save to ${tmpFile.getAbsolutePath}, "
        + new File(tmpFile.getAbsolutePath).length())
    })
  }

  "faster rcnn preprocess" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("VOCdevkit")
    val voc = Imdb.getImdb("voc_2007_testcode", resource.getPath)
    val roidb = voc.getRoidb().toIterator
    val imgAug = LocalByteRoiimageReader() -> RecordToFeature(true) ->
      BytesToMat() ->
      RandomAspectScale(Array(400, 500, 600, 700), 1) ->
      RoiResize() ->
      RandomTransformer(HFlip() -> RoiHFlip(false), 0.5) ->
      MatToFloats(validHeight = 300, validWidth = 300) //, meanRGB = Some(123f, 117f, 104f)
    val out = imgAug(roidb)
    out.foreach(img => {
      val tmpFile = java.io.File.createTempFile("module", ".jpg")
      val mat = OpenCVMat.floatToMat(img.getFloats(), img.getHeight(), img.getWidth())
      visulize(img.getLabel[RoiLabel], mat, false)
      Imgcodecs.imwrite(tmpFile.getAbsolutePath, mat)
      println(s"save to ${tmpFile.getAbsolutePath}, "
        + new File(tmpFile.getAbsolutePath).length())
    })
  }
}
