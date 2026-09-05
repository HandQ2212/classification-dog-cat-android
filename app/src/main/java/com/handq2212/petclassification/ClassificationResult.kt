package com.handq2212.petclassification

/** Một dự đoán đơn lẻ: tên loài + xác suất. */
data class Prediction(
    val species: String,
    val confidence: Float
)

data class ClassificationResult(
    val species: String,
    val confidence: Float,
    val inferenceTimeMs: Long,
    /** Top dự đoán, đã sắp xếp giảm dần theo confidence. */
    val topPredictions: List<Prediction> = emptyList()
)

/**
 * Cách chuẩn hoá pixel trước khi đưa vào model.
 *
 * Quan trọng: hai model trong assets KHÔNG giống nhau.
 * - pet_classifier_mobilenet.tflite đã nhúng sẵn lớp preprocess_input của MobileNetV2
 *   (graph bắt đầu bằng MUL 1/127.5 rồi SUB 1.0), nên app phải đưa pixel thô 0..255.
 *   Nếu app tự chuẩn hoá thêm lần nữa thì ảnh bị xử lý hai lần và kết quả sẽ sai.
 * - pet_classifier_custom.tflite không có lớp rescaling nào trong graph (op đầu tiên là
 *   CONV_2D), nên app phải tự chia 255 đúng như lúc train.
 */
enum class Normalization {
    /** Giữ nguyên 0..255 — model đã tự chuẩn hoá bên trong. */
    RAW,

    /** Chia 255 để đưa về 0..1. */
    ZERO_TO_ONE
}

enum class ModelChoice(
    val fileName: String,
    val displayName: String,
    val normalization: Normalization
) {
    MOBILENET(
        fileName = "pet_classifier_mobilenet.tflite",
        displayName = "MobileNetV2 (Transfer Learning)",
        normalization = Normalization.RAW
    ),
    CUSTOM_CNN(
        fileName = "pet_classifier_custom.tflite",
        displayName = "Custom CNN",
        normalization = Normalization.ZERO_TO_ONE
    );

    companion object {
        /** Model mặc định — chọn Custom CNN vì đây là model đang cho kết quả đúng. */
        val DEFAULT = CUSTOM_CNN
    }
}
