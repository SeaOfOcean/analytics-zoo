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

package com.intel.analytics.zoo.transform.vision.image

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat

import scala.collection.{Iterator, mutable}
import scala.reflect.ClassTag

class ImageFeature extends Serializable {
  def this(bytes: Array[Byte], label: Option[Any] = None, path: Option[String] = None) {
    this
    state(ImageFeature.bytes) = bytes
    if (path.isDefined) {
      state(ImageFeature.path) = path.get
    }
    if (label.isDefined) {
      state(ImageFeature.label) = label.get
    }
  }

  private val state = new mutable.HashMap[String, Any]()


  def apply[T](key: String): T = state(key).asInstanceOf[T]

  def update(key: String, value: Any): Unit = state(key) = value

  def contains(key: String): Boolean = state.contains(key)

  def opencvMat(): OpenCVMat = state(ImageFeature.mat).asInstanceOf[OpenCVMat]

  def hasLabel(): Boolean = state.contains(ImageFeature.label)

  def getFloats(): Array[Float] = {
    state(ImageFeature.floats).asInstanceOf[Array[Float]]
  }

  def getWidth(): Int = {
    if (state.contains(ImageFeature.width)) state(ImageFeature.width).asInstanceOf[Int]
    else opencvMat().width()
  }

  def getHeight(): Int = {
    if (state.contains(ImageFeature.height)) state(ImageFeature.height).asInstanceOf[Int]
    else opencvMat().height()
  }

  def getOriginalWidth: Int = state(ImageFeature.originalW).asInstanceOf[Int]

  def getOriginalHeight: Int = state(ImageFeature.originalH).asInstanceOf[Int]

  def getLabel[T: ClassTag]: T = {
    if (hasLabel()) this (ImageFeature.label).asInstanceOf[T] else null.asInstanceOf[T]
  }

  def clear(): Unit = {
    state.clear()
  }


  def copyTo(storage: Array[Float], offset: Int,
             toRGB: Boolean = true): Unit = {
    require(contains(ImageFeature.floats), "there should be floats in ImageFeature")
    val data = getFloats()
    require(data.length >= getWidth() * getHeight() * 3,
      "float array length should be larger than 3 * ${getWidth()} * ${getHeight()}")
    val frameLength = getWidth() * getHeight()
    require(frameLength * 3 + offset <= storage.length)
    var j = 0
    if (toRGB) {
      while (j < frameLength) {
        storage(offset + j) = data(j * 3 + 2)
        storage(offset + j + frameLength) = data(j * 3 + 1)
        storage(offset + j + frameLength * 2) = data(j * 3)
        j += 1
      }
    } else {
      while (j < frameLength) {
        storage(offset + j) = data(j * 3)
        storage(offset + j + frameLength) = data(j * 3 + 1)
        storage(offset + j + frameLength * 2) = data(j * 3 + 2)
        j += 1
      }
    }
  }
}

object ImageFeature {
  val label = "label"
  val path = "path"
  val mat = "mat"
  val bytes = "bytes"
  val floats = "floats"
  val width = "width"
  val height = "height"
  // original image width
  val originalW = "originalW"
  val originalH = "originalH"
  val cropBbox = "cropBbox"
  val expandBbox = "expandBbox"

  def apply(bytes: Array[Byte], path: Option[String] = None, label: Option[Any] = None)
  : ImageFeature = new ImageFeature(bytes, label, path)

  def apply(): ImageFeature = new ImageFeature()
}

abstract class FeatureTransformer() extends Serializable {
  private var outKey: Option[String] = None

  def setOutKey(key: String): this.type = {
    outKey = Some(key)
    this
  }

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> [C](other: FeatureTransformer): FeatureTransformer = {
    new SingleChainedTransformer(this, other)
  }

  def transform(feature: ImageFeature): Unit = {}

  def apply(feature: ImageFeature): ImageFeature = {
    try {
      transform(feature)
    } catch {
      case e: Exception =>
        e.printStackTrace()
    }
    if (outKey.isDefined) {
      require(outKey.get != ImageFeature.mat, s"the output key should not equal to" +
        s" ${ImageFeature.mat}, please give another name")
      if (feature.contains(outKey.get)) {
        val mat = feature(outKey.get).asInstanceOf[OpenCVMat]
        feature.opencvMat().copyTo(mat)
      } else {
        feature(outKey.get) = feature.opencvMat().clone()
      }
    }
    feature
  }

  def toIterator: Transformer[ImageFeature, ImageFeature] = {
    new TransfomerIterator(this)
  }
}


class SingleChainedTransformer
(first: FeatureTransformer, last: FeatureTransformer)
  extends FeatureTransformer {
  override def apply(prev: ImageFeature): ImageFeature = {
    last(first(prev))
  }
}


class TransfomerIterator(singleTransformer: FeatureTransformer)
  extends Transformer[ImageFeature, ImageFeature] {
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    prev.map(singleTransformer(_))
  }
}

class RandomTransformer(transformer: FeatureTransformer, maxProb: Double)
  extends FeatureTransformer {
  override def apply(prev: ImageFeature): ImageFeature = {
    if (RNG.uniform(0, 1) < maxProb) {
      transformer(prev)
    }
    prev
  }

  override def toString: String = {
    s"Random[${transformer.getClass.getCanonicalName}, $maxProb]"
  }
}

object RandomTransformer {
  def apply(transformer: FeatureTransformer, maxProb: Double): RandomTransformer =
    new RandomTransformer(transformer, maxProb)
}