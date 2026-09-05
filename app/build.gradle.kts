plugins {
    // AGP 9 đã tích hợp sẵn Kotlin (built-in Kotlin support),
    // không apply thêm org.jetbrains.kotlin.android nữa — sẽ xung đột extension "kotlin".
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.handq2212.petclassification"
    compileSdk = 37

    androidResources {
        // Model .tflite phải để nguyên (không nén) để LiteRT mmap trực tiếp từ assets
        noCompress += "tflite"
    }

    defaultConfig {
        applicationId = "com.handq2212.petclassification"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    buildFeatures {
        viewBinding = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // LiteRT Runtime
    implementation(libs.litert)

    // Gson parse metadata.json
    implementation(libs.gson)

    // Coroutines để chạy inference bất đồng bộ + lifecycleScope
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.androidx.lifecycle.runtime.ktx)
}
