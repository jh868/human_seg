package com.msai.myportraitseg

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.nio.ByteBuffer

class SegmentationModule(private val module: Module) {
    private val tagName: String = javaClass.simpleName

    init {
        Log.i(tagName, "SegmentationModule is instantiate")
    }

    private fun matToBitmap(inputMat: Mat): Bitmap {
        val bitmap = Bitmap.createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(inputMat, bitmap)
        return bitmap
    }

    private fun bitmapToTensor(inputBitmap: Bitmap): Tensor {
        return TensorImageUtils.bitmapToFloat32Tensor(
            inputBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
    }

    private fun bitmapToMat(inputBitmap: Bitmap): Mat {
        val mat = Mat()
        Utils.bitmapToMat(inputBitmap, mat)
        return mat
    }

    private fun floatArrayToGrayscaleBitmap (
        floatArray: FloatArray,
        width: Int,
        height: Int,
        alpha :Byte = (255).toByte(),
        reverseScale :Boolean = false
    ) : Bitmap {

        // Create empty bitmap in RGBA format (even though it says ARGB but channels are RGBA)
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val byteBuffer = ByteBuffer.allocate(width * height * 4)

        // mapping smallest value to 0 and largest value to 255
        val maxValue = floatArray.maxOrNull() ?: 1.0f
        val minValue = floatArray.minOrNull() ?: 0.0f
        val delta = maxValue - minValue
        var tempValue :Byte
//        Log.i(tagName, "maxValue: $maxValue")
//        Log.i(tagName, "minValue: $minValue")

        // Define if float min..max will be mapped to 0..255 or 255..0
        val conversion = when(reverseScale) {
            false -> { v: Float -> ((v - minValue) / delta * 255.0).toInt().toByte() }
            true -> { v: Float -> (255 - (v - minValue) / delta * 255.0).toInt().toByte() }
        }

        // copy each value from float array to RGB channels and set alpha channel
        floatArray.forEachIndexed { i, value ->
            tempValue = conversion(value)
//            Log.i(tagName, "tempValue: $tempValue")
            byteBuffer.put(4 * i, tempValue)
            byteBuffer.put(4 * i + 1, tempValue)
            byteBuffer.put(4 * i + 2, tempValue)
            byteBuffer.put(4 * i + 3, alpha)
        }

        bitmap.copyPixelsFromBuffer(byteBuffer)

        return bitmap
    }

    fun getPersonMat(inputMat: Mat): Mat {
        val width = inputMat.cols()
        val height = inputMat.rows()

        // Size(256.0 * 256.0)
        val inputTensor = bitmapToTensor(matToBitmap(inputMat))

        // get segment Tensor
        // for mobilenet
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()

//        // for deeplab model
//        val outputTensors = module.forward(IValue.from(inputTensor)).toDictStringKey()
//        val outputTensor = outputTensors["out"]!!.toTensor()

        //
        val scoresFloatArray = outputTensor.dataAsFloatArray
        val bitmap = floatArrayToGrayscaleBitmap(scoresFloatArray, width, height)
//        Log.i(tagName, "size: " + scoresFloatArray.size.toString())
//        Log.i(tagName, scoresFloatArray.contentToString())
//        Log.i(tagName, "########################")

        return bitmapToMat(bitmap)
    }

}