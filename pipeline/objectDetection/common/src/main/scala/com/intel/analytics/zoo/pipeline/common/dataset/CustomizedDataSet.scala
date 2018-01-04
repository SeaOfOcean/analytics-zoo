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

package com.intel.analytics.zoo.pipeline.common.dataset

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.RoiImagePath

import scala.io.Source


class CustomizedDataSet(val imageSet: String, devkitPath: String) extends Imdb {

  def loadRoidb(): Array[RoiImagePath] = {
    val classFile = new File(devkitPath + "/" + "classname.txt")
    require(classFile.exists(), s"if labelMap is null," +
      s" there should be a classname.txt in $devkitPath")
    val labelMap = getLabelMap(classFile.getAbsolutePath)
    val imageSetFile = Paths.get(devkitPath, "ImageSets", s"$imageSet.txt").toFile
    assert(imageSetFile.exists(), "Path does not exist " + imageSetFile.getAbsolutePath)
    val roidb = Source.fromFile(imageSetFile).getLines()
      .map(line => line.trim.split("\\s")).toArray.map(x => {
      RoiImagePath(devkitPath + "/" + x(0),
        PascalVoc.loadAnnotation(devkitPath + "/" + x(1), labelMap))
    })
    roidb
  }

  def getLabelMap(labelFile: String): Map[String, Float] = {
    val classes = Source.fromFile(labelFile).getLines().toArray
    (classes zip (Stream from 1).map(_.toFloat)).toMap
  }
}
