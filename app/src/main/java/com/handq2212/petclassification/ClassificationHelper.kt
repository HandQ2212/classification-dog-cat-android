package com.handq2212.petclassification

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.InputStreamReader
import kotlin.math.exp
import kotlin.math.min

class PetClassifierHelper(private val context: Context) {

    private var compiledModel: CompiledModel? = null
    private var inputBuffers: List<TensorBuffer>? = null
    private var outputBuffers: List<TensorBuffer>? = null

    private val labels = mutableListOf<String>()

    private var activeModel: ModelChoice? = null
    private var imageSize = DEFAULT_IMAGE_SIZE

    /** Buffer pixel dùng lại giữa các lần inference để tránh cấp phát lặp lại. */
    private var pixelInts: IntArray = IntArray(0)
    private var pixelFloats: FloatArray = FloatArray(0)

    /** Chỉ cho một lần load/inference chạy tại một thời điểm. */
    private val lock = Any()

    private var metadataLoaded = false

    /**
     * Load model. Hàm này nặng (compile graph) nên luôn chạy ngoài main thread.
     * Gọi lại với model khác sẽ giải phóng model cũ trước.
     */
    suspend fun setupModel(choice: ModelChoice) = withContext(Dispatchers.IO) {
        synchronized(lock) {
            loadLabelsAndMetadataIfNeeded()
            releaseNative()

            val options = CompiledModel.Options(Accelerator.CPU)
            // LiteRT đọc trực tiếp từ assets (đã cấu hình noCompress "tflite"),
            // không cần copy model ra cacheDir.
            val model = CompiledModel.create(context.assets, choice.fileName, options)

            val pixelCount = imageSize * imageSize
            if (pixelInts.size != pixelCount) {
                pixelInts = IntArray(pixelCount)
                pixelFloats = FloatArray(pixelCount * CHANNELS)
            }

            compiledModel = model
            inputBuffers = model.createInputBuffers()
            outputBuffers = model.createOutputBuffers()
            activeModel = choice

            Log.i(TAG, "Đã load ${choice.fileName} (input ${imageSize}x$imageSize, ${labels.size} nhãn)")
        }
    }

    /**
     * Chạy nhận diện trên [bitmap]. Bitmap phải là ARGB_8888 dạng software
     * (hardware bitmap không đọc được pixel).
     */
    suspend fun classify(bitmap: Bitmap): ClassificationResult = withContext(Dispatchers.Default) {
        synchronized(lock) {
            val model = compiledModel
            val inputs = inputBuffers
            val outputs = outputBuffers
            val choice = activeModel
            if (model == null || inputs == null || outputs == null || choice == null) {
                throw IllegalStateException("Model chưa được khởi tạo, hãy gọi setupModel() trước")
            }

            val startTime = SystemClock.uptimeMillis()

            // 1. Crop giữa và resize ảnh
            val squareBitmap = centerCropSquare(bitmap)
            val scaledBitmap = squareBitmap.scale(imageSize)

            // 2. Chuyển pixel sang FloatArray
            scaledBitmap.getPixels(pixelInts, 0, imageSize, 0, 0, imageSize, imageSize)

            if (scaledBitmap !== squareBitmap) scaledBitmap.recycle()
            if (squareBitmap !== bitmap) squareBitmap.recycle()

            // CẢ 2 MODEL TFLITE ĐỀU ĐÃ CHỨA LAYER CHUẨN HOÁ TRONG GRAPH
            // (Custom CNN dùng Rescaling, MobileNetV2 dùng preprocess_input)
            // Do đó input nạp vào BẮT BUỘC là pixel gốc [0.0f, 255.0f]
            fillInputRaw()

            // 3. Ghi input và chạy suy luận
            inputs[0].writeFloat(pixelFloats)
            model.run(inputs, outputs)
            val rawOutput = outputs[0].readFloat()

            val inferenceTime = SystemClock.uptimeMillis() - startTime

            // ========================================================
            // 4. XỬ LÝ KẾT QUẢ ĐẦU RA NHỊ PHÂN (SIGMOID 1 NEURON)
            // ========================================================
            val (species, confidence, top) = if (rawOutput.size == 1) {
                // Nhãn khi train: 0 = Cat, 1 = Dog -> rawOutput[0] chính là xác suất Dog
                val rawScore = rawOutput[0]
                val dogProb = rawScore.coerceIn(0f, 1f)
                val catProb = 1f - dogProb

                val isDog = dogProb >= 0.5f
                val predSpecies = if (isDog) "Dog".toVietnameseSpecies() else "Cat".toVietnameseSpecies()
                val predConfidence = if (isDog) dogProb else catProb

                val predictions = listOf(
                    Prediction("Dog".toVietnameseSpecies(), dogProb),
                    Prediction("Cat".toVietnameseSpecies(), catProb)
                ).sortedByDescending { it.confidence }

                Log.d(TAG, "Inference raw Sigmoid: $rawScore => Chó: $dogProb, Mèo: $catProb")

                Triple(predSpecies, predConfidence, predictions)
            } else {
                // Fallback nếu chuyển sang model phân loại đa lớp (Softmax)
                val probabilities = ensureProbabilities(rawOutput)
                val list = probabilities.indices
                    .sortedByDescending { probabilities[it] }
                    .take(TOP_K)
                    .map { idx -> toPrediction(idx, probabilities[idx]) }
                val best = list.firstOrNull() ?: Prediction(UNKNOWN_SPECIES, 0f)
                Triple(best.species, best.confidence, list)
            }

            ClassificationResult(
                species = species,
                confidence = confidence,
                inferenceTimeMs = inferenceTime,
                topPredictions = top
            )
        }
    }

    /** Giải phóng native memory. Gọi được từ main thread (chỉ là các lệnh free). */
    fun close() {
        synchronized(lock) { releaseNative() }
    }

    // ---------------------------------------------------------------- internal

    private fun toPrediction(index: Int, confidence: Float): Prediction {
        val label = labels.getOrElse(index) { UNKNOWN_SPECIES }
        return Prediction(
            species = label.toVietnameseSpecies(),
            confidence = confidence
        )
    }

    /**
     * Ghi pixel nguyên bản vào mảng Float (dải 0.0f - 255.0f).
     * Tuyệt đối không chia cho 255.0f tại đây.
     */
    private fun fillInputRaw() {
        var out = 0
        for (pixel in pixelInts) {
            pixelFloats[out++] = (pixel shr 16 and 0xFF).toFloat() // R: 0f - 255f
            pixelFloats[out++] = (pixel shr 8 and 0xFF).toFloat()  // G: 0f - 255f
            pixelFloats[out++] = (pixel and 0xFF).toFloat()         // B: 0f - 255f
        }
    }

    /** Nếu output chưa phải phân phối xác suất (tổng != 1) thì áp dụng softmax (chỉ dùng khi size > 1). */
    private fun ensureProbabilities(values: FloatArray): FloatArray {
        if (values.size <= 1) return values
        val sum = values.sum()
        val looksLikeProbabilities = values.all { it >= 0f } && kotlin.math.abs(sum - 1f) < 0.01f
        if (looksLikeProbabilities) return values

        val max = values.max()
        val exps = FloatArray(values.size) { exp((values[it] - max).toDouble()).toFloat() }
        val expSum = exps.sum()
        if (expSum == 0f) return exps
        for (i in exps.indices) exps[i] = exps[i] / expSum
        return exps
    }

    private fun readInputImageSize(metadata: Map<String, Any>): Int {
        val raw = metadata["img_size"] as? List<*>
        val size = (raw?.firstOrNull() as? Number)?.toInt()?.takeIf { it > 0 }
        if (size == null) {
            Log.w(TAG, "Không đọc được img_size, dùng mặc định $DEFAULT_IMAGE_SIZE")
            return DEFAULT_IMAGE_SIZE
        }
        return size
    }

    private fun loadLabelsAndMetadataIfNeeded() {
        if (metadataLoaded) return

        // 1. labels.txt — bỏ dòng trống, nếu không index sẽ lệch so với output của model
        runCatching {
            context.assets.open(LABELS_ASSET).bufferedReader().useLines { lines ->
                lines.map { it.trim() }
                    .filter { it.isNotEmpty() }
                    .forEach { labels.add(it) }
            }
        }.onFailure { Log.e(TAG, "Không đọc được $LABELS_ASSET", it) }

        // 2. metadata.json — lấy img_size
        runCatching {
            context.assets.open(METADATA_ASSET).use { stream ->
                val type = object : TypeToken<Map<String, Any>>() {}.type
                val data: Map<String, Any> = Gson().fromJson(InputStreamReader(stream), type)

                imageSize = readInputImageSize(data)

                // Nếu labels.txt lỗi, khôi phục nhãn từ idx_to_label
                if (labels.isEmpty()) {
                    @Suppress("UNCHECKED_CAST")
                    val idxToLabel = data["idx_to_label"] as? Map<String, String>
                    idxToLabel?.entries
                        ?.sortedBy { it.key.toIntOrNull() ?: Int.MAX_VALUE }
                        ?.forEach { labels.add(it.value) }
                }
            }
        }.onFailure { Log.e(TAG, "Không đọc được $METADATA_ASSET", it) }

        metadataLoaded = true
    }

    private fun releaseNative() {
        inputBuffers?.forEach { runCatching { it.close() } }
        outputBuffers?.forEach { runCatching { it.close() } }
        runCatching { compiledModel?.close() }
        inputBuffers = null
        outputBuffers = null
        compiledModel = null
        activeModel = null
    }

    private fun centerCropSquare(source: Bitmap): Bitmap {
        val side = min(source.width, source.height)
        if (source.width == side && source.height == side) return source
        val left = (source.width - side) / 2
        val top = (source.height - side) / 2
        return Bitmap.createBitmap(source, left, top, side, side)
    }

    private fun Bitmap.scale(size: Int): Bitmap =
        if (width == size && height == size) this
        else Bitmap.createScaledBitmap(this, size, size, true)

    private fun String.toVietnameseSpecies(): String = when (lowercase()) {
        "cat" -> "Mèo"
        "dog" -> "Chó"
        else -> this
    }

    private companion object {
        const val TAG = "PetClassifierHelper"
        const val LABELS_ASSET = "labels.txt"
        const val METADATA_ASSET = "metadata.json"
        const val DEFAULT_IMAGE_SIZE = 224
        const val CHANNELS = 3
        const val TOP_K = 2
        const val UNKNOWN_SPECIES = "Không xác định"
    }
}