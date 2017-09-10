package com.intel.analytics.bigdl.pipeline.common

import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.optim.SGD.LearningRateSchedule

case class PlateauWithWarm(monitor: String, factor: Float = 0.1f,
  patience: Int = 10, mode: String = "min", epsilon: Float = 1e-4f,
  cooldown: Int = 0, minLr: Float = 0, warmUpIteration: Int = 0,
  startWarnLr: Double) extends LearningRateSchedule {
  require(factor < 1, "Plateau does not support a factor >= 1.0")
  require(mode == "min" || mode == "max",
    s"Learning Rate Plateau Reducing mode ${ mode } is unknown, please use min | max")
  var (monitorOp, best) = if (mode == "min") {
    ((a: Float, b: Float) => a < b - epsilon, Float.PositiveInfinity)
  } else {
    ((a: Float, b: Float) => a > b + epsilon, Float.NegativeInfinity)
  }
  private var cooldownCounter: Int = 0
  private var waitCounter: Int = 0
  private val lrEpsilon: Float = minLr * 1e-4f
  private var curEpoch = 1


  /**
   * update learning rate by config table and state table
   * @param optimMethod init optiMethod.
   */
  override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
    val nevals = optimMethod.state.get[Int]("evalCounter").getOrElse(0)
    val epoch = optimMethod.state[Int]("epoch")
    if (nevals < warmUpIteration) {
      val lr = optimMethod.learningRate
      val warmUpDelta = (lr - minLr) / warmUpIteration
      currentRate = -minLr - warmUpDelta * nevals
      optimMethod.state("evalCounter") = nevals + 1
      return
    }
    if (epoch == 1) currentRate = -optimMethod.learningRate
    if (epoch == curEpoch) return
    curEpoch = epoch
    val current = optimMethod.state.get[Float](monitor)
    require(current.isDefined, s"Learning Rate Plateau Reducing requires ${monitor} available!")
    if (cooldownCounter > 0) {
      cooldownCounter -= 1
      waitCounter = 0
    }
    if (monitorOp(current.get, best)) {
      best = current.get
      waitCounter = 0
    } else if (cooldownCounter <= 0) {
      if (waitCounter >= patience) {
        if (currentRate.abs > minLr + lrEpsilon) {
          currentRate = - Math.max(currentRate.abs * factor, minLr)
          cooldownCounter = cooldown
          waitCounter = 0
        }
      }
      waitCounter += 1
    }
  }
}

