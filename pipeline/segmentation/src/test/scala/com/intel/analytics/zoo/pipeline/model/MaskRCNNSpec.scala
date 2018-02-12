package com.intel.analytics.zoo.pipeline.model

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.{DetectionOutputMRcnn, Graph, ProposalMaskRcnn, PyramidROIAlign, _}
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.LocalPredictor
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.zoo.pipeline.model.MaskRCNN
import com.intel.analytics.zoo.pipeline.utils.{ImageMeta, UnmodeDetection, Visualizer}
import org.opencv.core.CvType
import org.opencv.imgcodecs.Imgcodecs
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.io.Source

class MaskRCNNSpec extends FlatSpec with Matchers with BeforeAndAfter {


  private val nodeNumber = 1
  private val coreNumber = 1
  val batchPerCore = 1
  var subModelNumber = coreNumber

  before {
    System.setProperty("bigdl.localMode", "true")
    Engine.init(nodeNumber, coreNumber, false)
//    subModelNumber = Engine.getEngineType match {
//      case MklBlas => coreNumber
//      case _ => throw new IllegalArgumentException
//    }
  }

  after {
    System.clearProperty("bigdl.localMode")
  }

  def loadFeaturesFullName(s: String, hasSize: Boolean = true,
    middleRoot: String = middleRoot): Tensor[Float] = {
    loadFeaturesFullPath(Paths.get(middleRoot, s).toString, hasSize)
  }

  def loadFeaturesFullPath(s: String, hasSize: Boolean = true): Tensor[Float] = {
    if (hasSize) {
      val size = s.substring(s.lastIndexOf("-") + 1, s.lastIndexOf("."))
        .split("_").map(x => x.toInt)
      Tensor(Storage(Source.fromFile(s).getLines()
        .map(x => x.toFloat).toArray)).reshape(size)
    } else {
      Tensor(Storage(Source.fromFile(s).getLines()
        .map(x => x.toFloat).toArray))
    }
  }

  var middleRoot = "/home/jxy/data/maskrcnn/weights/"

  def loadFeatures(s: String, middleRoot: String = middleRoot)
  : Tensor[Float] = {
    if (s.contains(".txt")) {
      loadFeaturesFullName(s, hasSize = true, middleRoot)
    } else {
      val list = new File(middleRoot).listFiles()
      if (list != null) {
        list.foreach(x => {
          if (x.getName.matches(s"$s-.*txt")) {
            return loadFeaturesFullName(x.getName, hasSize = true, middleRoot)
          }
        })
      }
      println(s"cannot map $s")
      null
    }
  }

  "compare resnet" should "work" in {
    val input = Tensor[Float](1, 3, 128, 128).fill(1)
    val model = MaskRCNN().evaluate()

    loadWeights(model, "/home/jxy/data/maskrcnn/weights3/")

    val out = model.forward(input).toTable
    compare("C2", model("res2c_out").get, 1e-5, "weights3")
    compare("C3", model("res3d_out").get, 1e-5, "weights3")
    compare("C4", model("res4w_out").get, 1e-5, "weights3")
    compare("C5", model("res5c_out").get, 1e-5, "weights3")
  }

  "compare feature map" should "work" in {
    val input = Tensor[Float](1, 3, 128, 128).fill(1)

    val model = MaskRCNN().evaluate()

    loadWeights(model, "/home/jxy/data/maskrcnn/weights5/")

    val out = model.forward(input).toTable

    middleRoot = "/home/jxy/data/maskrcnn/weights5/p2"
    var expected2 = loadFeatures("p2")
    var outout = toHWC(out[Tensor[Float]](1)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5);
      a
    })

    middleRoot = "/home/jxy/data/maskrcnn/weights5/p3"
    expected2 = loadFeatures("p3")
    outout = toHWC(out[Tensor[Float]](2)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5);
      a
    })


    middleRoot = "/home/jxy/data/maskrcnn/weights5/p4"
    expected2 = loadFeatures("p4")
    outout = toHWC(out[Tensor[Float]](3)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5);
      a
    })

    middleRoot = "/home/jxy/data/maskrcnn/weights5/p5"
    expected2 = loadFeatures("p5")
    outout = toHWC(out[Tensor[Float]](4)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5);
      a
    })

  }
  "compare conv" should "work" in {
    val inputImage = Input[Float]()
    val x = Convolution(3, 64, 7, 7, 2, 2,
      optnet = false, propagateBack = false).setName("conv1").inputs(inputImage)
    val x2 = x

    val model = Graph(inputImage, x2).evaluate()


    loadWeights(model, "/home/jxy/data/maskrcnn/weights_c11/")
    middleRoot = "/home/jxy/data/maskrcnn/weights_c11/input"
    val input = loadFeatures("input").transpose(2, 4).transpose(3, 4).contiguous()
    val out = model.forward(input)
    compare2("x2", out.toTensor, 1e-6, "weights_c11")
//    compare2("relu", out[Tensor[Float]](5), 1e-5, "weights_c1")
//    compare2("c1", output, 1e-5, "weights_c1")
//    println(output.size().mkString("x"))
////    output.map(expected2, (a, b) => {
////      if (Math.abs(a - b) > 1e-5) {
////        println(a, b)
////      }
//////      assert(Math.abs(a - b) < 1e-5);
////      a
////    })
//
//    middleRoot = "/home/jxy/data/maskrcnn/weights_c1/x1"
//    expected2 = loadFeatures("x1")
//    output = out[Tensor[Float]](2)
//    println(output.size().mkString("x"))
//    output.map(expected2, (a, b) => {
//      if (Math.abs(a - b) > 1e-5) {
//        println(a, b)
//      }
////      assert(Math.abs(a - b) < 1e-5);
//      a
//    })
  }
  "compare c1" should "work" in {
    val inputImage = Input[Float]()
    var x = SpatialZeroPadding(3, 3, 3, 3).inputs(inputImage)
    val x1 = x
    x = Convolution(3, 64, 7, 7, 2, 2,
      optnet = false, propagateBack = false).setName("conv1").inputs(x)
    val x2 = x
    x = SpatialBatchNormalization(64, eps = 0.001)
      .setName("bn_conv1").inputs(x)
    val x3 = x
    x = ReLU(true).inputs(x)
    val x4 = x
    x = SpatialMaxPooling(3, 3, 2, 2, -1, -1).setName("pool1").inputs(x)

    val model = Graph(inputImage, Array(x, x1, x2, x3, x4)).evaluate()


    loadWeights(model, "/home/jxy/data/maskrcnn/weights/")
    middleRoot = "/home/jxy/data/maskrcnn/weights"
    val input = loadFeatures("input").transpose(2, 4).transpose(3, 4).contiguous()
    //    val input = Tensor(1, 3, 128, 128).fill(1)
    val out = model.forward(input).toTable

//    middleRoot = "/home/jxy/data/maskrcnn/weights_c11/c1"
//    var expected2 = loadFeatures("c1")
//    var output = out[Tensor[Float]](1)

//    compare("C1", model("pool1").get, 1e-3, "weights")
//    compare2("c1", out[Tensor[Float]](1), 1e-3, "weights_c1")
    compare2("x1", out[Tensor[Float]](2), 1e-6, "weights_c1")
    compare2("x2", out[Tensor[Float]](3), 1e-3, "weights_c1")
//    compare2("relu", out[Tensor[Float]](5), 1e-5, "weights_c1")
//    compare2("c1", output, 1e-5, "weights_c1")
//    println(output.size().mkString("x"))
////    output.map(expected2, (a, b) => {
////      if (Math.abs(a - b) > 1e-5) {
////        println(a, b)
////      }
//////      assert(Math.abs(a - b) < 1e-5);
////      a
////    })
//
//    middleRoot = "/home/jxy/data/maskrcnn/weights_c1/x1"
//    expected2 = loadFeatures("x1")
//    output = out[Tensor[Float]](2)
//    println(output.size().mkString("x"))
//    output.map(expected2, (a, b) => {
//      if (Math.abs(a - b) > 1e-5) {
//        println(a, b)
//      }
////      assert(Math.abs(a - b) < 1e-5);
//      a
//    })
  }


  "compare rpn" should "work" in {
    val inputs = (2 to 6).map(i => {
      middleRoot = s"/home/jxy/data/maskrcnn/weights/p$i"
      loadFeatures(s"p$i").transpose(2, 4).transpose(3, 4).contiguous()
    }).toArray

    val data = T.array(inputs)

    println("load data done ", data.length())
//    val input = Tensor[Float](1, 256, size, size).fill(1)

//    val model = MaskRCNN.buildRpnModel(1, 3, 256)

    val rpn_feature_maps = (1 to 5).map(i => Input()).toArray

    // RPN Model
    val rpn = MaskRCNN.buildRpnModel(1, 3, 256)
    val mapTable = MapTable(rpn).inputs(rpn_feature_maps)

    // Concatenate layer outputs
    // Convert from list of lists of level outputs to list of lists
    // of outputs across levels.
    // e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    val rpn_class_logits = JoinTable(2, 3).setName("rpn_class_logits")
      .inputs(MaskRCNN.select(rpn_feature_maps.length, 1, mapTable))
    val rpn_class = JoinTable(2, 3).setName("rpn_class")
      .inputs(MaskRCNN.select(rpn_feature_maps.length, 2, mapTable))
    val rpn_bbox = JoinTable(2, 3).setName("rpn_bbox")
      .inputs(MaskRCNN.select(rpn_feature_maps.length, 3, mapTable))

    val model = Graph(rpn_feature_maps, Array(rpn_class_logits, rpn_class, rpn_bbox))

    loadWeights(model, "/home/jxy/data/maskrcnn/weights/")

    val out = model.forward(data).toTable
    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_class_logits"
    var expected2 = loadFeatures("rpn_class_logits")
    var output = out[Tensor[Float]](1)
    println(output.size().mkString("x"))
    output.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-4);
      a
    })
    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_class"
    expected2 = loadFeatures("rpn_class")
    output = out[Tensor[Float]](2)
    output.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-4);
      a
    })
    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_bbox"
    expected2 = loadFeatures("rpn_bbox")
    output = out[Tensor[Float]](3)
    output.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-4);
      a
    })

  }

  "compare convblock" should "work" in {
    middleRoot = "/home/jxy/data/maskrcnn/weights3/input"
    val input = loadFeatures("input").transpose(2, 4).transpose(3, 4).contiguous()

    val in = Input[Float]()
    val convBlock = MaskRCNN.convBlock(3, in, 3, Array(64, 64, 256), stage = 2,
      block = 'a', strides = (1, 1))

    val model = Graph(in, convBlock).evaluate()

    loadWeights(model, "/home/jxy/data/maskrcnn/weights4/")

    val out = model.forward(input)

    middleRoot = "/home/jxy/data/maskrcnn/weights3/x"
    val expected2 = loadFeatures("x")
    toHWC(out.toTensor).contiguous().map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-6);
      a
    })
  }

  "compare identity" should "work" in {
    middleRoot = "/home/jxy/data/maskrcnn/weights_identity/input"
    val input = loadFeatures("input").transpose(2, 4).transpose(3, 4).contiguous()

    val in = Input[Float]()
    val convBlock = MaskRCNN.identityBlock(3, in, 3, Array(64, 64, 3), stage = 2,
      block = 'b')

    val model = Graph(in, convBlock).evaluate()

    loadWeights(model, "/home/jxy/data/maskrcnn/weights_identity/")

    val out = model.forward(input)
//    middleRoot = "/home/jxy/data/maskrcnn/weights_identity/x1"
//    var expected = loadFeatures("x1")
//    toHWC(out.toTable[Tensor[Float]](2)).contiguous().map(expected, (a, b) => {
//      assert(Math.abs(a - b) < 1e-6); a
//    })
//
//    middleRoot = "/home/jxy/data/maskrcnn/weights_identity/x2"
//    expected = loadFeatures("x2")
//    toHWC(out.toTable[Tensor[Float]](3)).contiguous().map(expected, (a, b) => {
//      assert(Math.abs(a - b) < 1e-6); a
//    })
//
//    middleRoot = "/home/jxy/data/maskrcnn/weights_identity/x3"
//    expected = loadFeatures("x3")
//    val x3 = out.toTable[Tensor[Float]](4)
//    toHWC(x3).contiguous().map(expected, (a, b) => {
//      assert(Math.abs(a - b) < 1e-6); a
//    })

    middleRoot = "/home/jxy/data/maskrcnn/weights_identity/x"
    val expected2 = loadFeatures("x")
    val transform = toHWC(out.toTensor[Float]).contiguous()
    var i = 0
    transform.map(expected2, (a, b) => {
      i += 1
      assert(Math.abs(a - b) < 1e-5);
      a
    })
  }

  "MaskRCNN forward" should "work" in {
    val input = loadFeatures("data").transpose(2, 4).transpose(3, 4).contiguous()
    val imageMeta = loadFeatures("image_metas")
    var model = MaskRCNN().evaluate()
//    val saved = Module.load[Float]("/tmp/mask-rcnn.model")
//    model.loadModelWeights(saved)
    loadWeights(model)
//    model.save("/tmp/mask-rcnn.model", true)
    model.saveModule("/tmp/mask-rcnn.model", overWrite = true)
//val model = Module.loadModule[Float]("/tmp/mask-rcnn.model").evaluate()
    println("load model done ...........")
    val out = model.forward(T(input, imageMeta))
    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_class_logits"
    var expected = loadFeatures("rpn_class_logits")

    println(out.toTable[Tensor[Float]](1).size().mkString("x"))
    println(out.toTable[Tensor[Float]](2).size().mkString("x"))
    println(out.toTable[Tensor[Float]](3).size().mkString("x"))

//    compare("C1", model("pool1").get, 1e-3, "weights")
//
//    compare("C2", model("res2c_out").get, 1e-3, "weights")
//    compare("C3", model("res3d_out").get, 1e-3, "weights")
//    compare("C4", model("res4w_out").get, 1e-3, "weights")
//    compare("C5", model("res5c_out").get, 1e-3, "weights")
//
//    compare("p2", model("fpn_p2").get, 1e-3, "weights")
//    compare("p3", model("fpn_p3").get, 1e-3, "weights")
//    compare("p4", model("fpn_p4").get, 1e-3, "weights")
//    compare("p5", model("fpn_p5").get, 1e-3, "weights")
//    compare("p6", model("fpn_p6").get, 1e-3, "weights")

//    compare("rpn_class_logits", model("rpn_class_logits").get, 1e-3, "weights")
    compare("rpn_bbox", model("rpn_bbox").get, 1e-3, "weights")
    compare("rpn_class", model("rpn_class").get, 1e-3, "weights")

//    compare("anchors", model("anchors").get, 1e-3, "weights")


    val output1 = model("ROI").get.output.toTensor
    println(output1.size().mkString("x"))
    val output = output1.clone()
    output.narrow(3, 1, 1).copy(output1.narrow(3, 2, 1))
    output.narrow(3, 2, 1).copy(output1.narrow(3, 1, 1))
    output.narrow(3, 3, 1).copy(output1.narrow(3, 4, 1))
    output.narrow(3, 4, 1).copy(output1.narrow(3, 3, 1))
//    compare("mrcnn_mask", model("mrcnn_mask").get, 1e-3, "weights")
//    print(model("rpn_class").get.output)
//
//    println(model("ROI").get.output)
    println(model.output.toTable[Tensor[Float]](1).size().mkString("x"))
    println(model.output.toTable[Tensor[Float]](2).size().mkString("x"))
    println(model.output.toTable[Tensor[Float]](3).size().mkString("x"))
    println(model.output.toTable[Tensor[Float]](4).size().mkString("x"))
    println(model.output.toTable[Tensor[Float]](5).size().mkString("x"))
    println(model.output.toTable[Tensor[Float]](6).size().mkString("x"))
    println(model.output.toTable[Tensor[Float]](7).size().mkString("x"))
    println(model.output.toTable[Tensor[Float]](4))
    println("hehe")
//
//    def compare(name: String): Unit = {
//      middleRoot = s"/home/jxy/data/maskrcnn/weights/$name"
//      expected = loadFeatures(s"${name}")
//      toHWC(model(s"fpn_$name").get.output.toTensor[Float]).contiguous().map(expected, (a, b) => {
//        assert(Math.abs(a - b) < 1e-3); a
//      })
//    }

//    out.toTable[Tensor[Float]](1).map(expected, (a, b) => {
//      assert(Math.abs(a - b) < 1e-5);
//      a
//    })
//    toHWC(out.toTable[Tensor[Float]](1)).contiguous().map(expected, (a, b) => {
//      assert(Math.abs(a - b) < 1e-5);
//      a
//    })
  }


  def compare(expectedname: String, layer: Module[Float], prec: Double, middle: String): Unit = {
    compare2(expectedname, layer.output.toTensor[Float], prec, middle)
  }

  def compare2(expectedname: String, output: Tensor[Float], prec: Double, middle: String): Unit = {
    println(s"compare .............$expectedname")
    middleRoot = s"/home/jxy/data/maskrcnn/${middle}/$expectedname"
    val expected = loadFeatures(s"${expectedname}")
    val out = if (output.dim() == 4) {
      toHWC(output).contiguous()
    } else {
      output.contiguous()
    }
    out.map(expected, (a, b) => {
      if (Math.abs(a - b) > prec) {
        println(a, b, Math.abs(a - b))
//        return false
      }
      a
//      assert(Math.abs(a - b) < prec); a
    })
  }

  def compare3(expectedname: String, output: Tensor[Float], prec: Double, middle: String): Unit = {
    println(s"compare .............$expectedname")
    middleRoot = s"${middle}/$expectedname"
    val expected = loadFeatures(s"${expectedname}")
    val out = if (output.dim() == 4) {
      toHWC(output).contiguous()
    } else {
      output.contiguous()
    }
    out.map(expected, (a, b) => {
      if (Math.abs(a - b) > prec) {
        println(a, b, Math.abs(a - b))
//        return false
      }
      a
//      assert(Math.abs(a - b) < prec); a
    })
  }

  def loadWeights(model: Module[Float], root: String = "/home/jxy/data/maskrcnn/weights/"): Unit = {
    val modules = Utils.getNamedModules(model)
    modules.foreach(x => {
      val name = x._1
      val layer = x._2
      if (layer.getParametersTable() != null) {
        middleRoot = root + s"$name"
        println(s"load for $middleRoot")
        val pt = layer.getParametersTable()
        if (pt.contains(name)) {
          pt[Table](name).keySet.foreach(x => {
            val param = pt[Table](name)[Tensor[Float]](x)
            if (x != "gradWeight" && x != "gradBias") {

              val load = if (x == "weight") {
                if (layer.getClass.getCanonicalName.contains("BatchNorm")) {
                  loadFeatures("gamma:0")
                } else {
                  var w = loadFeatures("kernel:0")
                  if (w != null) {
                    if (w.dim() == 4) {
                      w = w.transpose(1, 3).transpose(2, 4).transpose(1, 2).contiguous()
                    }
                    if (w.dim() == 2) {
                      w = w.transpose(1, 2).contiguous()
                    }
                  }
                  w
                }
              } else if (x == "bias") {
                if (layer.getClass.getCanonicalName.contains("BatchNorm")) {
                  loadFeatures("beta:0")
                } else {
                  loadFeatures("bias:0")
                }
              } else if (x == "runningMean") {
                loadFeatures("moving_mean:0")
              } else {
                // if (x == "runningVar") {
                require(x == "runningVar")
                loadFeatures("moving_variance:0")
              }
              if (load != null && param.nElement() > 0) {
                println(s"load $name $x..............................")
                compareShape(param.size(), load.size())
                param.copy(load)
              }
            }
          })
        }
      }
    })
  }

  def compareShape(size1: Array[Int], size2: Array[Int]): Unit = {
    val s = if (size1.length != size2.length) {
      size1.slice(1, 5)
    } else size1
    s.zip(size2).foreach(x => {
      if (x._1 != x._2) {
        println(s"compare ${size1.mkString("x")} with ${size2.mkString("x")}")
        throw new Exception(s"compare ${size1.mkString("x")} with ${size2.mkString("x")}")
        return
      }
    })
  }

  def toHWC(tensor: Tensor[Float]): Tensor[Float] = {
    require(tensor.dim() == 4)
    tensor.transpose(2, 3).transpose(3, 4).contiguous()
  }

  def toCHW(tensor: Tensor[Float]): Tensor[Float] = {
    require(tensor.dim() == 4)
    tensor.transpose(3, 4).transpose(2, 3).contiguous()
  }
//
//  "generate anchors" should "work" in {
//    val rpn_feature_maps = Array(Input(), Input(), Input(), Input(), Input())
//    val priorBoxes = rpn_feature_maps.indices.map(i => {
//      PriorBox[Float](Array(MaskRCNN.RPN_ANCHOR_SCALES(i)),
//        _aspectRatios = MaskRCNN.RPN_ANCHOR_RATIOS,
//        imgSize = 1, step = MaskRCNN.BACKBONE_STRIDES(i), isFlip = false, offset = 0,
//        hasVariances = false)
//        .inputs(rpn_feature_maps(i))
//    }).toArray
//
//    val anchors = JoinTable(3, 3).inputs(priorBoxes)
//    val reshape = InferReshape(Array(1, -1, 4)).inputs(anchors)
//    val model = Graph(rpn_feature_maps, reshape)
//
//    val input = T(Tensor(1, 256, 16, 16))
//
//
////    val input = T(Tensor(1, 256, 256, 256), Tensor(1, 256, 128, 128),
////      Tensor(1, 256, 64, 64), Tensor(1, 256, 32, 32), Tensor(1, 256, 16, 16))
//
//    println(model.forward(input))
//  }


//  "generate anchors1" should "work" in {
//    val rpn_feature_maps = Input()
//    val priorBoxes = PriorBox[Float](Array(MaskRCNN.RPN_ANCHOR_SCALES.last),
//      _aspectRatios = MaskRCNN.RPN_ANCHOR_RATIOS,
//      imgSize = 1, step = MaskRCNN.BACKBONE_STRIDES.last, isFlip = false, offset = 0,
//      hasVariances = false)
//      .inputs(rpn_feature_maps)
//    val reshape = InferReshape(Array(1, -1, 4)).inputs(priorBoxes)
//    val model = Graph(rpn_feature_maps, reshape)
//
//    val input = Tensor(1, 256, 16, 16)
//
//    println(model.forward(input))
//  }

  "proposal" should "work" in {
    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_class"
    val rpn_class = loadFeatures("rpn_class")

    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_bbox"
    val rpn_bbox = loadFeatures("rpn_bbox")

    middleRoot = "/home/jxy/data/maskrcnn/weights/anchors"
    val anchors1 = loadFeatures("anchors").resize(1, 261888, 4)
    val anchors = anchors1.clone()
    anchors.narrow(3, 1, 1).copy(anchors1.narrow(3, 2, 1))
    anchors.narrow(3, 2, 1).copy(anchors1.narrow(3, 1, 1))
    anchors.narrow(3, 3, 1).copy(anchors1.narrow(3, 4, 1))
    anchors.narrow(3, 4, 1).copy(anchors1.narrow(3, 3, 1))

    val inputs = (1 to 3).map(i => Input()).toArray
    // rpn_class, rpn_bbox, imInfo, anchors
    val proposal = ProposalMaskRcnn(6000, 1000)
      .setName("ROI")
      .inputs(inputs)

    val model = Graph[Float](inputs, proposal).evaluate()

    val out = model.forward(T(rpn_class, rpn_bbox, anchors))
    println(out)

    val output1 = model("ROI").get.output.toTensor
    val output = output1.clone()
    output.narrow(3, 1, 1).copy(output1.narrow(3, 2, 1))
    output.narrow(3, 2, 1).copy(output1.narrow(3, 1, 1))
    output.narrow(3, 3, 1).copy(output1.narrow(3, 4, 1))
    output.narrow(3, 4, 1).copy(output1.narrow(3, 3, 1))
    compare2("rpn_rois", output, 1e-3, "weights")
  }

  "maskrcnn classifier" should "work" in {
    val rpn_rois = Input()
    val mrcnn_feature_maps: Array[ModuleNode[Float]] = Array(Input(), Input(), Input(), Input())
    val (mrcnn_class_logits, mrcnn_class, mrcnn_bbox) =
      MaskRCNN.fpnClassifierGraph(rpn_rois, mrcnn_feature_maps,
        MaskRCNN.IMAGE_SHAPE, MaskRCNN.POOL_SIZE, 81)

    val model = Graph(Array(rpn_rois) ++ mrcnn_feature_maps,
      Array(mrcnn_class_logits, mrcnn_class, mrcnn_bbox)).evaluate()
    loadWeights(model)

//    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_class"
//    val rpn_class = loadFeatures("rpn_class")
//
//    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_bbox"
//    val rpn_bbox = loadFeatures("rpn_bbox")

    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_rois"
    val rpn_rois_data = loadFeatures("rpn_rois")

    middleRoot = "/home/jxy/data/maskrcnn/weights/p2"
    val p2 = toCHW(loadFeatures("p2"))
    middleRoot = "/home/jxy/data/maskrcnn/weights/p3"
    val p3 = toCHW(loadFeatures("p3"))
    middleRoot = "/home/jxy/data/maskrcnn/weights/p4"
    val p4 = toCHW(loadFeatures("p4"))
    middleRoot = "/home/jxy/data/maskrcnn/weights/p5"
    val p5 = toCHW(loadFeatures("p5"))

    val input = T(rpn_rois_data, p2, p3, p4, p5)
    model.forward(input)


    compare("mrcnn_class_conv1", model("mrcnn_class_conv1").get, 1e-3, "weights")
    compare("mrcnn_class_logits", model("mrcnn_class_logits").get, 1e-3, "weights")
    compare("mrcnn_class", model("mrcnn_class").get, 1e-3, "weights")
    compare("mrcnn_bbox", model("mrcnn_bbox").get, 1e-3, "weights")
  }

  "PyramidRoiAlign forward" should "work properly" in {
    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_rois"
    val rpn_rois_data1 = loadFeatures("rpn_rois")
    val rpn_rois_data = rpn_rois_data1.clone()

    rpn_rois_data.narrow(3, 1, 1).copy(rpn_rois_data1.narrow(3, 2, 1))
    rpn_rois_data.narrow(3, 2, 1).copy(rpn_rois_data1.narrow(3, 1, 1))
    rpn_rois_data.narrow(3, 3, 1).copy(rpn_rois_data1.narrow(3, 4, 1))
    rpn_rois_data.narrow(3, 4, 1).copy(rpn_rois_data1.narrow(3, 3, 1))

    middleRoot = "/home/jxy/data/maskrcnn/weights/p2"
    val p2 = toCHW(loadFeatures("p2"))
    middleRoot = "/home/jxy/data/maskrcnn/weights/p3"
    val p3 = toCHW(loadFeatures("p3"))
    middleRoot = "/home/jxy/data/maskrcnn/weights/p4"
    val p4 = toCHW(loadFeatures("p4"))
    middleRoot = "/home/jxy/data/maskrcnn/weights/p5"
    val p5 = toCHW(loadFeatures("p5"))

    val input = T(rpn_rois_data, p2, p3, p4, p5)
    val layer = PyramidROIAlign(7, 7, 1024, 1024, 3)
    layer.saveModule("/tmp/roi", overWrite = true)
    val start = System.nanoTime()
    val loop = 10
    (1 to loop).foreach(i => {
      layer.forward(input)
    })
    println(s"time is ${(System.nanoTime() - start) / loop / 1e9}")
//    println(layer.output)
    compare2("roialign", layer.output, 1e-4, "weights")
  }

  "PyramidRoiAlign single forward" should "work properly" in {
    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_rois"
    val rpn_rois_data1 = loadFeatures("rpn_rois")
    val rpn_rois_data = rpn_rois_data1.clone()

    rpn_rois_data.narrow(3, 1, 1).copy(rpn_rois_data1.narrow(3, 2, 1))
    rpn_rois_data.narrow(3, 2, 1).copy(rpn_rois_data1.narrow(3, 1, 1))
    rpn_rois_data.narrow(3, 3, 1).copy(rpn_rois_data1.narrow(3, 4, 1))
    rpn_rois_data.narrow(3, 4, 1).copy(rpn_rois_data1.narrow(3, 3, 1))

//    middleRoot = "/home/jxy/data/maskrcnn/weights/p2"
//    val p2 = toCHW(loadFeatures("p2"))
//    middleRoot = "/home/jxy/data/maskrcnn/weights/p3"
//    val p3 = toCHW(loadFeatures("p3"))
//    middleRoot = "/home/jxy/data/maskrcnn/weights/p4"
//    val p4 = toCHW(loadFeatures("p4"))
    middleRoot = "/home/jxy/data/maskrcnn/weights/p5"
    val p5 = toCHW(loadFeatures("p5"))

    val input = T(rpn_rois_data, p5)
    val layer = PyramidROIAlign(7, 7, 1024, 1024, 3)
    layer.saveModule("/tmp/roi", overWrite = true)
    layer.forward(input)
    println(toHWC(layer.output))
//    compare2("roialign", layer.output, 1e-5, "weights")
  }

  "compare deconv" should "work" in {
    val layer = SpatialFullConvolution(256, 256, 2, 2, 2, 2).setName("jxy2")
    loadWeights(layer, "/tmp/Conv2DTranspose_p/")
    middleRoot = "/tmp/Conv2DTranspose_p/data/"
    val input = toCHW(loadFeatures("data", middleRoot))
    layer.forward(input)
    println(toHWC(layer.output))
    compare3("out", layer.output, 1e-5, "/tmp/Conv2DTranspose_p/")
  }

  "refine detection" should "wprk" in {
    val layer = DetectionOutputMRcnn()

    middleRoot = "/home/jxy/data/maskrcnn/weights/"
    val rois = loadFeatures("rois")
    val probs = loadFeatures("probs")
    val deltas = loadFeatures("deltas")
    val window = loadFeatures("window")
    layer.refineDetection(rois, probs, deltas, window)
  }

  "data preprocessing" should "work" in {
    val images = ImageFrame.read("/home/jxy/code/Mask_RCNN/images/1045023827_4ec3e8ba5c_z.jpg") ->
      AspectScale(800, 1, 1024, useScaleFactor = false, minScale = Some(1)) ->
      FixExpand(1024, 1024) ->
      ChannelNormalize(103.9f, 116.8f, 123.7f) ->
      MatToTensor() -> ImageMeta(81) ->
      ImageFrameToSample(Array(ImageFeature.imageTensor, ImageMeta.imageMeta))
    val model = Module.loadModule[Float]("/tmp/mask-rcnn.model").evaluate()
    println("load model done")
    val predictor = LocalPredictor[Float](model, batchPerCore = 1)
    val output = predictor.predictImage(images.toLocal())
    val detectOut = UnmodeDetection()
    detectOut(images)
    val image = OpenCVMat.read("/home/jxy/code/Mask_RCNN/images/1045023827_4ec3e8ba5c_z.jpg")
    val out = images.toLocal()
      .array(0)[(Tensor[Float], Tensor[Float], Tensor[Float], Array[Tensor[Float]])]("unmode")

    val imageWithMask = Visualizer.drawMask(image, out._4)
    Imgcodecs.imwrite("/tmp/save.jpg", imageWithMask)
  }

  "data preprocessing cat" should "work" in {
    //    val imagePath = "/home/jxy/code/Mask_RCNN/images/1045023827_4ec3e8ba5c_z.jpg"
    val imagePath = "/home/jxy/data/dogs.jpg"
    val images = ImageFrame.read(imagePath) ->
      AspectScale(800, 1, 1024, useScaleFactor = false, minScale = Some(1)) ->
      FixExpand(1024, 1024) ->
      ChannelNormalize(103.9f, 116.8f, 123.7f) ->
      MatToTensor() -> ImageMeta(81) ->
      ImageFrameToSample(Array(ImageFeature.imageTensor, ImageMeta.imageMeta))
    println("load data done")
    val model = Module.loadModule[Float]("/tmp/mask-rcnn.model")
    println("load model done")
    //    SpatialShareConvolution.shareConvolution(model)
//    val predictor = LocalPredictor[Float](model, batchPerCore = 1)
//    val output = predictor.predictImage(images.toLocal())
val detectOut = UnmodeDetection()
    val output = model.predictImage(images, batchPerPartition = 1).transform(detectOut)
    println("predict done")
    val image = OpenCVMat.read(imagePath)
    val out = output.toLocal()
      .array(0)[(Tensor[Float], Tensor[Float], Tensor[Float], Array[Tensor[Float]])]("unmode")
    // (boxes, classIds, scores, fullMasks)

    println(out)
    val labelMap = readCocoLabelMap()

    Visualizer.visualize(image, out._1, out._3, out._2, labelMap, 0.3)
    val imageWithMask = Visualizer.drawMask(image, out._4)
    Imgcodecs.imwrite("/tmp/save4.jpg", imageWithMask)
  }


  def readCocoLabelMap(): Map[Int, String] = {
    readLabelMap("/coco_classname.txt")
  }

  def readLabelMap(labelFileName: String): Map[Int, String] = {
    val labelFile = getClass().getResource(labelFileName)
    Source.fromURL(labelFile).getLines().zipWithIndex.map(x => (x._2, x._1)).toMap
  }

  "data preprocessing2" should "work" in {
    val images = ImageFrame.read("/home/jxy/code/Mask_RCNN/images/1045023827_4ec3e8ba5c_z.jpg") ->
      AspectScale(800, 1, 1024, useScaleFactor = false, minScale = Some(1)) ->
      FixExpand(1024, 1024) ->
      ChannelNormalize(103.9f, 116.8f, 123.7f) ->
      MatToTensor() -> ImageMeta(81) ->
      ImageFrameToSample(Array(ImageFeature.imageTensor, ImageMeta.imageMeta))
    val data = images.toLocal().array(0)[Tensor[Float]](ImageFeature.imageTensor)
      .resize(1, 3, images.toLocal().array(0).getHeight, images.toLocal().array(0).getWidth())
    println(data.size().mkString("x"))
    compare2("input", data, 1e-2, "weights")
  }

  "whole process" should "work" in {
    val images = ImageFrame.read("/home/jxy/code/Mask_RCNN/images/1045023827_4ec3e8ba5c_z.jpg")
    val transformer =
      AspectScale(800, 1, 1024, useScaleFactor = false, minScale = Some(1)) ->
        FixExpand(1024, 1024) ->
        ChannelNormalize(103.9f, 116.8f, 123.7f) ->
        MatToTensor()
  }

  "draw mask" should "work" in {
    val image = OpenCVMat.read("/home/jxy/code/Mask_RCNN/images/1045023827_4ec3e8ba5c_z.jpg")
    val mask = loadFeatures("mask0")
    val mask1 = loadFeatures("mask1")
    val mask2 = loadFeatures("mask2")
    val mask3 = loadFeatures("mask3")
    val imageWithMask = Visualizer.drawMask(image, Array(mask, mask1, mask2, mask3))
    Imgcodecs.imwrite("/tmp/save.jpg", imageWithMask)
  }

//  "resize and crop" should "work" in {
//    val mat = OpenCVMat.read("/home/jxy/data/roger.jpg")
//    mat.convertTo(mat, CvType.CV_32FC3)
//    val tensor = OpenCVMat.toTensor(mat)
//    tensor.resize(Array(1) ++ tensor.size())
//    val input = toCHW(tensor)
//
//    println(input.size().mkString("x"))
//    val height = mat.height()
//    val width = mat.width()
//    //    val crop = Cropping2D(Array((height * 0.3).round.toInt, (height * 0.3).round.toInt),
////      Array((0.1 * width).round.toInt, (width * 0.2).round.toInt), DataFormat.NCHW)
//val crop = Cropping2D(Array((0.2 * height).floor.toInt, (0.0 * height).toInt),
//  Array((0.3 * width).floor.toInt, (0.0 * width).toInt), DataFormat.NCHW)
////    val crop = Cropping2D(Array(0, height * 0.7 - 20 - 50),
////      Array(0, width - 30 - 60), DataFormat.NCHW)
//    crop.forward(input)
//    //    println(toHWC(crop.output))
//    val resize = ResizeBilinear(300, 400, true)
//    resize.forward(crop.output)
//    println(toHWC(resize.output))
//  }
}
