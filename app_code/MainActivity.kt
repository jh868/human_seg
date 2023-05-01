package com.msai.myportraitseg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.ImageView
import org.opencv.android.*
import org.opencv.android.CameraBridgeViewBase.*
import org.opencv.android.OpenCVLoader.OPENCV_VERSION
import org.opencv.core.Core
import org.opencv.core.Core.ROTATE_90_COUNTERCLOCKWISE
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.pytorch.Module
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class MainActivity : CameraActivity(), CvCameraViewListener2, View.OnClickListener {
    private val tagName: String = javaClass.simpleName
    private var selectedBackground: Bitmap? = null
    private var selectedBackgroundMat: Mat? = null

    private lateinit var segmentationModule: SegmentationModule
    private lateinit var mOpenCvCameraView: CameraBridgeViewBase
    private lateinit var mRGBA: Mat
    private lateinit var mRGBAT: Mat
    private lateinit var maskMat: Mat

    private lateinit var resizedMat: Mat
    private lateinit var imageViewDefault: ImageView
    private lateinit var imageView0: ImageView
    private lateinit var imageView1: ImageView
    private lateinit var imageView2: ImageView
    private lateinit var imageView3: ImageView
    private lateinit var imageView4: ImageView
    private lateinit var imageView5: ImageView
    private lateinit var imageView6: ImageView
    private lateinit var imageView7: ImageView
    private lateinit var imageView8: ImageView
    private lateinit var imageView9: ImageView

    // to load model
    companion object {
        fun assetFilePath(context: Context, asset: String): String {
            val file = File(context.filesDir, asset)

            try {
                val inpStream: InputStream = context.assets.open(asset)
                try {
                    val outStream = FileOutputStream(file, false)
                    val buffer = ByteArray(4 * 1024)
                    var read: Int

                    while (true) {
                        read = inpStream.read(buffer)
                        if (read == -1) {
                            break
                        }
                        outStream.write(buffer, 0, read)
                    }
                    outStream.flush()
                } catch (ex: Exception) {
                    ex.printStackTrace()
                }
                return file.absolutePath
            } catch (e: Exception) {
                e.printStackTrace()
            }
            return ""
        }
    }

    // load opencv
    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                SUCCESS -> {
                    mOpenCvCameraView.enableView()

                    Log.d(tagName, "OpenCV loaded successfully")
                    Log.d(tagName, "OpenCV Version: $OPENCV_VERSION")
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        Log.d(tagName, "Called onCreate")

        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        imageViewDefault = findViewById(R.id.imageViewDefault)
        imageView0 = findViewById(R.id.imageView0)
        imageView1 = findViewById(R.id.imageView1)
        imageView2 = findViewById(R.id.imageView2)
        imageView3 = findViewById(R.id.imageView3)
        imageView4 = findViewById(R.id.imageView4)
        imageView5 = findViewById(R.id.imageView5)
        imageView6 = findViewById(R.id.imageView6)
        imageView7 = findViewById(R.id.imageView7)
        imageView8 = findViewById(R.id.imageView8)
        imageView9 = findViewById(R.id.imageView9)

        imageViewDefault.setOnClickListener(this)
        imageView0.setOnClickListener(this)
        imageView1.setOnClickListener(this)
        imageView2.setOnClickListener(this)
        imageView3.setOnClickListener(this)
        imageView4.setOnClickListener(this)
        imageView5.setOnClickListener(this)
        imageView6.setOnClickListener(this)
        imageView7.setOnClickListener(this)
        imageView8.setOnClickListener(this)
        imageView9.setOnClickListener(this)

        loadOpenCVConfigs()

        // load model
        val module = Module.load(assetFilePath(this, "Best_for_mobile.ptl"))
        segmentationModule = SegmentationModule(module)
    }

    // openCV Camera
    private fun loadOpenCVConfigs() {
        mOpenCvCameraView = findViewById(R.id.HelloOpenCvView)
        mOpenCvCameraView.setCameraIndex(CAMERA_ID_FRONT)
        mOpenCvCameraView.visibility = VISIBLE
        mOpenCvCameraView.setCvCameraViewListener(this)
        mOpenCvCameraView.setCameraPermissionGranted()

        Log.d(tagName, "CvCameraLoaded")
    }

    // the logic is executed when the camera preview starts
    override fun onCameraViewStarted(width: Int, height: Int) {
        mRGBA = Mat(height, width, CvType.CV_8UC4)
        mRGBAT = Mat()
        maskMat = Mat()
        resizedMat = Mat()
    }

    // the logic is invoked on camera preview interruption
    override fun onCameraViewStopped() {
        mRGBA.release()
        mRGBAT.release()
        maskMat.release()
        resizedMat.release()
    }

    // the logic is executed at the frame delivery time
    override fun onCameraFrame(inputFrame: CvCameraViewFrame?): Mat {
        return if (inputFrame != null) {
            // get image from frame information
            mRGBA = inputFrame.rgba()

            // flipping to show portrait mode properly
            Core.flip(mRGBA, mRGBAT, 1)

            if (selectedBackground == null) { selectedBackgroundMat = null }

            if (selectedBackgroundMat != null) {
                // resize image for model
                Imgproc.resize(mRGBAT, resizedMat, Size(256.0, 256.0))

                // get human image for foreground
                maskMat = segmentationModule.getPersonMat(resizedMat)

                // threshold on mask
                Imgproc.threshold(maskMat, maskMat, 70.0, 255.0, Imgproc.THRESH_BINARY)

                // blur on mask
                Imgproc.blur(maskMat, maskMat, Size(3.0, 3.0))

                // make mask great again!
                Imgproc.resize(maskMat, maskMat, Size(1920.0, 1080.0))

                // masking person
                Core.bitwise_and(maskMat, mRGBAT, mRGBAT)

                // clone background
                var clone = selectedBackgroundMat!!.clone()

                // result
                mRGBAT.copyTo(clone, maskMat)

                mRGBA.release()
                maskMat.release()
                resizedMat.release()

                return clone
            }

            // releasing what's not anymore needed
            mRGBA.release()
            maskMat.release()
            resizedMat.release()

            mRGBAT
        } else {
            mRGBA
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mOpenCvCameraView.disableView()
    }


    override fun onPause() {
        super.onPause()
        mOpenCvCameraView.disableView()
    }

    override fun onResume() {
        super.onResume()
        if (OpenCVLoader.initDebug()) {
            Log.d(tagName, "OpenCV loaded")
            mLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS)
        } else {
            Log.d(tagName, "OpenCV didn't load")
            OpenCVLoader.initAsync(OPENCV_VERSION, this, mLoaderCallback)
        }
    }

    override fun onClick(v: View?) {
        val option = BitmapFactory.Options()
        option.inPreferredConfig = Bitmap.Config.ARGB_8888

        when(v?.id) {
            R.id.imageView0 -> {
                selectedBackground = BitmapFactory.decodeResource(this.resources,
                    R.drawable.background_0, option)
            }

            R.id.imageView1 -> {
                selectedBackground = BitmapFactory.decodeResource(this.resources,
                    R.drawable.background_1, option)
            }

            R.id.imageView2 -> {
                selectedBackground = BitmapFactory.decodeResource(this.resources,
                    R.drawable.background_2, option)
            }

            R.id.imageView3 -> {
                selectedBackground = BitmapFactory.decodeResource(this.resources,
                    R.drawable.background_3, option)
            }

            R.id.imageView4 -> {
                selectedBackground = BitmapFactory.decodeResource(this.resources,
                    R.drawable.background_4, option)
            }

            R.id.imageView5 -> {
                selectedBackground = BitmapFactory.decodeResource(this.resources,
                    R.drawable.background_5, option)
            }

            R.id.imageView6 -> {
                selectedBackground = BitmapFactory.decodeResource(this.resources,
                    R.drawable.background_6, option)
            }

            R.id.imageView7 -> {
                selectedBackground = BitmapFactory.decodeResource(this.resources,
                    R.drawable.background_7, option)
            }

            R.id.imageView8 -> {
                selectedBackground = BitmapFactory.decodeResource(this.resources,
                    R.drawable.background_8, option)
            }

            R.id.imageView9 -> {
                selectedBackground = BitmapFactory.decodeResource(this.resources,
                    R.drawable.background_9, option)
            }

            R.id.imageViewDefault -> {
                selectedBackground = null
            }
        }

        if (selectedBackground != null) {
            selectedBackgroundMat = Mat()
            Utils.bitmapToMat(selectedBackground, selectedBackgroundMat)

            // rotate background
            Core.rotate(selectedBackgroundMat, selectedBackgroundMat, ROTATE_90_COUNTERCLOCKWISE)
            Imgproc.resize(selectedBackgroundMat, selectedBackgroundMat, Size(1920.0, 1080.0))
        }
    }
}