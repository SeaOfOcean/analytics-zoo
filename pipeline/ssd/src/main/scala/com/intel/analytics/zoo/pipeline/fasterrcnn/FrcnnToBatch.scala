package com.intel.analytics.zoo.pipeline.fasterrcnn

import com.intel.analytics.bigdl.dataset.{Transformer, Utils}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.transform.vision.label.roi.RoiLabel

import scala.collection.Iterator
import scala.collection.mutable.ArrayBuffer

object FrcnnToBatch {
  def apply(batchSize: Int, convertLabel: Boolean = true,
    partitionNum: Option[Int] = None, keepImageFeature: Boolean = true,
    inputKey: String = ImageFeature.floats): FrcnnToBatch
  = new FrcnnToBatch(batchSize, convertLabel, partitionNum, keepImageFeature, inputKey)
}

/**
 * Convert a batch of labeled BGR images into a Mini-batch.
 *
 * Notice: The totalBatch means a total batch size. In distributed environment, the batch should be
 * divided by total core number
 * @param totalBatch
 */
class FrcnnToBatch(totalBatch: Int,
  convertLabel: Boolean = true,
  partitionNum: Option[Int] = None, val keepImageFeature: Boolean = true,
  inputKey: String = ImageFeature.floats)
  extends Transformer[ImageFeature, FrcnnMiniBatch] {

  private val batchPerPartition = Utils.getBatchSize(totalBatch, partitionNum)

  override def apply(prev: Iterator[ImageFeature]): Iterator[FrcnnMiniBatch] = {
    val batchSizePerPartition = batchPerPartition
    println("batchSizePerPartition: " + batchSizePerPartition)
    new Iterator[FrcnnMiniBatch] {
      private val inputBatch: Table = T()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private val imInfoTensor: Tensor[Float] = Tensor[Float]()
      private var labelData: ArrayBuffer[Float] = null
      private var imInfoData: Array[Float] = null
      private var maps: Array[ImageFeature] = null
      private var width = 0
      private var height = 0
      private val batchSize = batchSizePerPartition

      override def hasNext: Boolean = prev.hasNext

      override def next(): FrcnnMiniBatch = {
        if (prev.hasNext) {
          var i = 0
          inputBatch.clear()
          if (labelData != null) labelData.clear()
          while (i < batchSize && prev.hasNext) {
            val input = T()
            inputBatch.insert(input)
            val feature = prev.next()
            height = feature.getHeight()
            width = feature.getWidth()
            println(height, width)
            if (imInfoData == null) {
              imInfoData = new Array[Float](batchSize * 4)
              maps = new Array[ImageFeature](batchSize)
              if (convertLabel) {
                labelData = new ArrayBuffer[Float]()
              }
            }
            require(feature.contains(inputKey), s"there should be ${inputKey} in ImageFeature")
            val data = feature.getFloats(inputKey)
            // hwc to chw
            val featureTensor = Tensor(Storage(data))
              .resize(1, feature.getHeight(), feature.getWidth(), 3).transpose(2, 4).transpose(3, 4)
              .contiguous()
            input.insert(featureTensor)
            input.insert(imInfoTensor)
            input.insert(labelTensor)
            imInfoData(i * 4) = height
            imInfoData(i * 4 + 1) = width
            imInfoData(i * 4 + 2) = height.toFloat / feature.getOriginalHeight
            imInfoData(i * 4 + 3) = width.toFloat / feature.getOriginalWidth
            if (convertLabel) {
              require(feature.hasLabel(), "if convert label, there should be label")
              val target = feature.getLabel[RoiLabel]
              if (target.classes.nElement() > 0) {
                var r = 0
                while (r < target.classes.size(2)) {
                  labelData.append(i)
                  labelData.append(target.classes.valueAt(1, r + 1))
                  // difficult
                  labelData.append(target.classes.valueAt(2, r + 1))
                  labelData.append(target.bboxes.valueAt(r + 1, 1))
                  labelData.append(target.bboxes.valueAt(r + 1, 2))
                  labelData.append(target.bboxes.valueAt(r + 1, 3))
                  labelData.append(target.bboxes.valueAt(r + 1, 4))
                  r += 1
                }
              } else {
                labelData.append(i)
                labelData.append(-1)
                // difficult
                labelData.append(-1)
                labelData.append(-1)
                labelData.append(-1)
                labelData.append(-1)
                labelData.append(-1)
              }
            }
            maps(i) = feature
            i += 1
          }

          imInfoTensor.set(Storage[Float](imInfoData), storageOffset = 1, sizes = Array(i, 4))
          val batch = if (convertLabel) {
            labelTensor.set(Storage[Float](labelData.toArray),
              storageOffset = 1, sizes = Array(labelData.length / 7, 7))
            FrcnnMiniBatch(inputBatch, labelTensor, imInfoTensor)
          } else {
            FrcnnMiniBatch(inputBatch, null, imInfoTensor)
          }
          batch
        } else {
          null
        }
      }
    }
  }
}
