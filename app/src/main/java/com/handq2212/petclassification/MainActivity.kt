package com.handq2212.petclassification

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.handq2212.petclassification.databinding.ActivityMainBinding
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var classifierHelper: PetClassifierHelper

    private var currentBitmap: Bitmap? = null
    private var selectedModel = ModelChoice.DEFAULT

    /** Job đang chạy, để huỷ khi người dùng chọn ảnh/model mới. */
    private var pendingJob: Job? = null

    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri ?: return@registerForActivityResult
            loadImageAndClassify(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.main) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        classifierHelper = PetClassifierHelper(this)

        setupUI()
        loadModel(selectedModel)
    }

    private fun setupUI() {
        val models = ModelChoice.entries.map { it.displayName }
        binding.dropdownModel.setAdapter(
            ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, models)
        )
        binding.dropdownModel.setText(selectedModel.displayName, false)

        binding.dropdownModel.setOnItemClickListener { _, _, position, _ ->
            val choice = ModelChoice.entries[position]
            if (choice == selectedModel) return@setOnItemClickListener
            selectedModel = choice
            loadModel(choice)
        }

        binding.btnChooseImage.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }
    }

    /** Load model ngoài main thread, sau đó chạy lại nhận diện trên ảnh hiện tại nếu có. */
    private fun loadModel(choice: ModelChoice) {
        pendingJob?.cancel()
        pendingJob = lifecycleScope.launch {
            showBusy(true, getString(R.string.status_loading_model))
            try {
                classifierHelper.setupModel(choice)
                if (currentBitmap == null) {
                    showStatus(getString(R.string.status_pick_image))
                } else {
                    classifyCurrentBitmap()
                }
            } catch (e: CancellationException) {
                throw e // job bị huỷ (đổi model/chọn ảnh mới) — không phải lỗi
            } catch (e: Exception) {
                Log.e(TAG, "Load model thất bại", e)
                showError(getString(R.string.error_load_model, e.messageOrClass()))
            } finally {
                showBusy(false)
            }
        }
    }

    private fun loadImageAndClassify(uri: Uri) {
        pendingJob?.cancel()
        pendingJob = lifecycleScope.launch {
            showBusy(true, getString(R.string.status_reading_image))
            try {
                val bitmap = withContext(Dispatchers.IO) { decodeSoftwareBitmap(uri) }
                if (bitmap == null) {
                    showError(getString(R.string.error_read_image))
                    return@launch
                }
                currentBitmap?.recycle()
                currentBitmap = bitmap
                binding.imageViewPet.setImageBitmap(bitmap)
                clearResult()
                classifyCurrentBitmap()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Đọc ảnh thất bại", e)
                showError(getString(R.string.error_read_image_detail, e.messageOrClass()))
            } finally {
                showBusy(false)
            }
        }
    }

    private suspend fun classifyCurrentBitmap() {
        val bitmap = currentBitmap ?: return
        showBusy(true, getString(R.string.status_classifying))
        try {
            val result = classifierHelper.classify(bitmap)
            updateUIWithResult(result)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Nhận diện thất bại", e)
            showError(getString(R.string.error_classify, e.messageOrClass()))
        } finally {
            showBusy(false)
        }
    }

    private fun updateUIWithResult(result: ClassificationResult) {
        binding.tvSpecies.text = result.species

        val percent = (result.confidence * 100).toInt().coerceIn(0, 100)
        binding.tvConfidence.text = getString(R.string.format_percent, percent)
        binding.progressConfidence.setProgressCompat(percent, true)
        binding.tvProcessingTime.text = getString(R.string.format_ms, result.inferenceTimeMs)

        // Top để thấy model đang "do dự" giữa những kết quả nào
        binding.tvTopPredictions.text = result.topPredictions
            .mapIndexed { index, p ->
                getString(
                    R.string.format_top_item,
                    index + 1,
                    p.species,
                    p.confidence * 100
                )
            }
            .joinToString("\n")
        binding.groupResult.visibility = View.VISIBLE

        // Dưới ngưỡng này thì model đang "đoán bừa" — nói rõ cho người dùng
        if (result.confidence < LOW_CONFIDENCE_THRESHOLD) {
            showStatus(getString(R.string.warning_low_confidence))
        } else {
            showStatus(null)
        }
    }

    private fun clearResult() {
        binding.tvSpecies.text = PLACEHOLDER
        binding.tvConfidence.text = getString(R.string.format_percent, 0)
        binding.progressConfidence.setProgressCompat(0, false)
        binding.tvProcessingTime.text = getString(R.string.format_ms, 0L)
        binding.tvTopPredictions.text = ""
    }

    private fun showBusy(busy: Boolean, message: String? = null) {
        binding.progressLoading.visibility = if (busy) View.VISIBLE else View.GONE
        binding.btnChooseImage.isEnabled = !busy
        binding.dropdownModel.isEnabled = !busy
        if (message != null) showStatus(message)
    }

    private fun showStatus(message: String?) {
        binding.tvStatus.text = message ?: ""
        binding.tvStatus.visibility = if (message.isNullOrEmpty()) View.GONE else View.VISIBLE
    }

    private fun showError(message: String) {
        showStatus(message)
        binding.groupResult.visibility = View.GONE
    }

    /**
     * ImageDecoder mặc định trả hardware bitmap — không đọc được pixel bằng getPixels().
     * Buộc allocator về SOFTWARE để helper có thể tiền xử lý ảnh.
     */
    private fun decodeSoftwareBitmap(uri: Uri): Bitmap? = try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = false
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }?.let { if (it.config == Bitmap.Config.ARGB_8888) it else it.toArgb8888() }
    } catch (e: Exception) {
        Log.e(TAG, "decodeSoftwareBitmap thất bại", e)
        null
    }

    private fun Bitmap.toArgb8888(): Bitmap {
        val converted = copy(Bitmap.Config.ARGB_8888, false)
        if (converted !== this) recycle()
        return converted
    }

    private fun Exception.messageOrClass(): String =
        message?.takeIf { it.isNotBlank() } ?: this::class.java.simpleName

    override fun onDestroy() {
        super.onDestroy()
        pendingJob?.cancel()
        classifierHelper.close()
        currentBitmap?.recycle()
        currentBitmap = null
    }

    private companion object {
        const val TAG = "MainActivity"
        const val PLACEHOLDER = "-"
        const val LOW_CONFIDENCE_THRESHOLD = 0.20f
    }
}
