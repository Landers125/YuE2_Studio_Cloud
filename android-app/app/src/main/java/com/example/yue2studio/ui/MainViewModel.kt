package com.example.yue2studio.ui

import android.app.Application
import android.net.Uri
import android.os.Environment
import android.provider.OpenableColumns
import android.util.Log
import android.widget.Toast
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.yue2studio.data.*
import com.example.yue2studio.player.AudioPlayerManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.isActive
import com.example.yue2studio.BuildConfig
import java.io.File
import java.io.FileOutputStream

class MainViewModel(application: Application) : AndroidViewModel(application) {
    val appFlavor: String = BuildConfig.APP_FLAVOR
    val defaultEngine: String = BuildConfig.DEFAULT_ENGINE
    val engineClient = YuE2EngineClient(targetEngine = BuildConfig.DEFAULT_ENGINE)
    val audioPlayer = AudioPlayerManager(application)
    val kaggleClient = KaggleApiClient(application)

    private val _serverHealth = MutableStateFlow(ServerHealth())
    val serverHealth: StateFlow<ServerHealth> = _serverHealth.asStateFlow()

    private val _activeJob = MutableStateFlow<ActiveJob?>(null)
    val activeJob: StateFlow<ActiveJob?> = _activeJob.asStateFlow()

    private val _tracks = MutableStateFlow<List<TrackItem>>(emptyList())
    val tracks: StateFlow<List<TrackItem>> = _tracks.asStateFlow()

    private val _trackTitle = MutableStateFlow("")
    val trackTitle: StateFlow<String> = _trackTitle.asStateFlow()

    private val _trackStyle = MutableStateFlow(
        if (appFlavor == "acestep") "Cyberpunk electro synthwave, distorted reese bass, driving 128 bpm drums, neon arpeggiator"
        else "Rock 80s, driving bass, analog synthesizer, punchy drums, powerful guitar riff"
    )
    val trackStyle: StateFlow<String> = _trackStyle.asStateFlow()

    private val _trackLyrics = MutableStateFlow(
        if (appFlavor == "acestep") "[Intro]\n[Verse 1]\nNeon shadows in the night\n[Chorus]\nElectric fire burning bright\n[Outro]"
        else "[Intro - Synth & Drums]\n[Verse 1]\nОгни ночного города горят вдали,\nМы мчимся сквозь пространство и мечты.\n[Chorus]\nМузыка звучит, сквозь века летит,\nНовый трек рождается в ночи!\n[Guitar Solo]\n[Outro]"
    )
    val trackLyrics: StateFlow<String> = _trackLyrics.asStateFlow()

    private val _trackDuration = MutableStateFlow(if (appFlavor == "acestep") 120f else 180f)
    val trackDuration: StateFlow<Float> = _trackDuration.asStateFlow()

    private val _trackSteps = MutableStateFlow(if (appFlavor == "acestep") 8 else 32)
    val trackSteps: StateFlow<Int> = _trackSteps.asStateFlow()

    fun setTrackTitle(v: String) { _trackTitle.value = v }
    fun setTrackStyle(v: String) { _trackStyle.value = v }
    fun setTrackLyrics(v: String) { _trackLyrics.value = v }
    fun setTrackDuration(v: Float) { _trackDuration.value = v }
    fun setTrackSteps(v: Int) { _trackSteps.value = v }

    private val _selectedCoverAudioUri = MutableStateFlow<Uri?>(null)
    val selectedCoverAudioUri: StateFlow<Uri?> = _selectedCoverAudioUri.asStateFlow()

    private val _selectedCoverAudioName = MutableStateFlow<String?>(null)
    val selectedCoverAudioName: StateFlow<String?> = _selectedCoverAudioName.asStateFlow()

    private val _coverAbcScore = MutableStateFlow("")
    val coverAbcScore: StateFlow<String> = _coverAbcScore.asStateFlow()

    private val _isCoverTranscribing = MutableStateFlow(false)
    val isCoverTranscribing: StateFlow<Boolean> = _isCoverTranscribing.asStateFlow()

    private val _coverTranscribeStage = MutableStateFlow("")
    val coverTranscribeStage: StateFlow<String> = _coverTranscribeStage.asStateFlow()

    private val _coverTranscribeError = MutableStateFlow<String?>(null)
    val coverTranscribeError: StateFlow<String?> = _coverTranscribeError.asStateFlow()

    private val _coverLyrics = MutableStateFlow("[Verse 1]\n\n[Chorus]\n\n[Verse 2]\n\n[Chorus]\n\n[Outro]")
    val coverLyrics: StateFlow<String> = _coverLyrics.asStateFlow()

    private val _coverStyle = MutableStateFlow("Emotional acoustic ballad, gentle grand piano, strings, intimate vocals")
    val coverStyle: StateFlow<String> = _coverStyle.asStateFlow()

    private val _coverTitle = MutableStateFlow("")
    val coverTitle: StateFlow<String> = _coverTitle.asStateFlow()

    private val _coverMelodyOnly = MutableStateFlow(true)
    val coverMelodyOnly: StateFlow<Boolean> = _coverMelodyOnly.asStateFlow()

    private val _selectedMidiAudioUri = MutableStateFlow<Uri?>(null)
    val selectedMidiAudioUri: StateFlow<Uri?> = _selectedMidiAudioUri.asStateFlow()

    private val _selectedMidiAudioName = MutableStateFlow<String?>(null)
    val selectedMidiAudioName: StateFlow<String?> = _selectedMidiAudioName.asStateFlow()

    private val _lastTranscribedMidiFile = MutableStateFlow<File?>(null)
    val lastTranscribedMidiFile: StateFlow<File?> = _lastTranscribedMidiFile.asStateFlow()

    private val _isTranscribing = MutableStateFlow(false)
    val isTranscribing: StateFlow<Boolean> = _isTranscribing.asStateFlow()

    private val _transcribeStage = MutableStateFlow("")
    val transcribeStage: StateFlow<String> = _transcribeStage.asStateFlow()

    private val _transcribeError = MutableStateFlow<String?>(null)
    val transcribeError: StateFlow<String?> = _transcribeError.asStateFlow()

    private val _transcribeModelSize = MutableStateFlow("large")
    val transcribeModelSize: StateFlow<String> = _transcribeModelSize.asStateFlow()

    private val _isKaggleStarting = MutableStateFlow(false)
    val isKaggleStarting: StateFlow<Boolean> = _isKaggleStarting.asStateFlow()

    private val _kaggleStatus = MutableStateFlow<String?>("Загрузка...")
    val kaggleStatus: StateFlow<String?> = _kaggleStatus.asStateFlow()

    private val _serverUrl = MutableStateFlow(engineClient.baseUrl)
    val serverUrl: StateFlow<String> = _serverUrl.asStateFlow()

    private val _kaggleStartStage = MutableStateFlow("")
    val kaggleStartStage: StateFlow<String> = _kaggleStartStage.asStateFlow()

    private val _serverLogs = MutableStateFlow<String>("")
    val serverLogs: StateFlow<String> = _serverLogs.asStateFlow()

    private val _isLoadingLogs = MutableStateFlow<Boolean>(false)
    val isLoadingLogs: StateFlow<Boolean> = _isLoadingLogs.asStateFlow()

    private val _showLogsDialog = MutableStateFlow<Boolean>(false)
    val showLogsDialog: StateFlow<Boolean> = _showLogsDialog.asStateFlow()

    private var pollingJob: Job? = null
    private var startJob: Job? = null


    init {
        refreshLibrary()
        _serverHealth.value = ServerHealth(isOnline = false, version = "Проверка статуса...")
        refreshKaggleStatus()
        startPlaybackProgressTicker()
    }

    private fun startPlaybackProgressTicker() {
        viewModelScope.launch {
            while (true) {
                delay(250)
                audioPlayer.updateProgress()
            }
        }
    }

    fun checkServer() {
        testKaggleConnection()
    }

    fun setServerUrl(url: String) {
        _serverUrl.value = url
    }

    fun refreshKaggleStatus() {
        viewModelScope.launch {
            val res = kaggleClient.getKernelStatus()
            if (res.isSuccess) {
                val st = res.getOrNull()?.uppercase() ?: ""
                _kaggleStatus.value = st

                // isOnline только когда ядро RUNNING И туннель известен
                val tunnelKnown = engineClient.cleanBaseUrl().isNotBlank()
                val isActuallyOnline = st == "RUNNING" && tunnelKnown

                // Если ядро упало или завершилось — сбрасываем baseUrl
                if (st == "ERROR" || st == "CANCELLACKNOWLEDGED" || st == "COMPLETE" ||
                    st == "CANCELACKNOWLEDGED" || st.contains("ERROR") || st.contains("FAILED")
                ) {
                    engineClient.baseUrl = ""
                    _serverUrl.value = ""
                }

                _serverHealth.value = ServerHealth(
                    isOnline = isActuallyOnline,
                    version = when {
                        isActuallyOnline -> "Tesla T4 GPU (16 GB)"
                        st == "QUEUED" -> "В очереди..."
                        st == "RUNNING" && !tunnelKnown -> "Запускается..."
                        else -> "Отключен"
                    }
                )
            }
        }
    }

    fun testKaggleConnection() {
        viewModelScope.launch {
            Toast.makeText(getApplication(), "Проверка Kaggle API...", Toast.LENGTH_SHORT).show()
            val res = kaggleClient.getKernelStatus()
            if (res.isSuccess) {
                val st = res.getOrNull()
                _kaggleStatus.value = st
                val isRunning = st.equals("RUNNING", ignoreCase = true) || st.equals("QUEUED", ignoreCase = true)
                if (st.equals("RUNNING", ignoreCase = true)) {
                    val discoveredUrl = engineClient.autoDiscoverFromNtfy()
                    if (!discoveredUrl.isNullOrBlank()) {
                        engineClient.baseUrl = discoveredUrl
                        val health = engineClient.checkHealth()
                        if (health.isOnline) {
                            _serverHealth.value = health
                            _serverUrl.value = discoveredUrl
                            Toast.makeText(getApplication(), "✓ Сервер готов и подключен!", Toast.LENGTH_SHORT).show()
                            return@launch
                        }
                    }
                }
                _serverHealth.value = ServerHealth(
                    isOnline = isRunning,
                    version = if (isRunning) "Tesla T4 GPU (16 GB)" else "Отключен"
                )
                Toast.makeText(getApplication(), "✓ Kaggle API доступен! Статус ядра: $st", Toast.LENGTH_SHORT).show()
            } else {
                val err = res.exceptionOrNull()?.message ?: "Ошибка"
                Toast.makeText(getApplication(), "Ошибка Kaggle API: $err", Toast.LENGTH_LONG).show()
            }
        }
    }

    private var stopJob: Job? = null
    private val _isKaggleStopping = MutableStateFlow(false)
    val isKaggleStopping: StateFlow<Boolean> = _isKaggleStopping.asStateFlow()

    fun startKaggleServer() {
        if (_isKaggleStarting.value || _isKaggleStopping.value) return
        startJob?.cancel()
        stopJob?.cancel()
        startJob = viewModelScope.launch {
            _isKaggleStarting.value = true
            _kaggleStartStage.value = "Отправка команды запуска в Kaggle API..."
            Toast.makeText(getApplication(), "⚡ Запуск сервера Kaggle (Tesla T4)...", Toast.LENGTH_SHORT).show()

            val res = kaggleClient.pushKernel(getApplication())
            if (!isActive) return@launch
            if (res.isFailure) {
                _isKaggleStarting.value = false
                _kaggleStartStage.value = ""
                val err = res.exceptionOrNull()?.message ?: "Ошибка запуска"
                Toast.makeText(getApplication(), "Ошибка запуска: $err", Toast.LENGTH_LONG).show()
                return@launch
            }

            _kaggleStartStage.value = "Ядро отправлено. Инициализация GPU..."
            var attempts = 0
            var queuedSeconds = 0
            while (isActive && attempts < 80) {
                delay(3000)
                if (!isActive) break
                attempts++
                val stRes = kaggleClient.getKernelStatus()
                val st = stRes.getOrNull()?.uppercase() ?: ""
                _kaggleStatus.value = st
                if (st == "RUNNING") {
                    _kaggleStartStage.value = "Подключение к движку YuE2 (2x Tesla T4)..."
                    var connected = false
                    // Даём 50 попыток по 3 сек = ~2.5 минуты на обнаружение туннеля
                    for (dAttempt in 1..50) {
                        if (!isActive) break
                        val discoveredUrl = engineClient.autoDiscoverFromNtfy()
                        if (!discoveredUrl.isNullOrBlank()) {
                            engineClient.baseUrl = discoveredUrl
                            val health = engineClient.checkHealth()
                            if (health.isOnline) {
                                _serverHealth.value = health
                                _serverUrl.value = discoveredUrl
                                connected = true
                                break
                            }
                        }
                        _kaggleStartStage.value = "Инициализация туннеля... (${dAttempt * 3} сек)"
                        delay(3000)
                    }
                    if (connected) {
                        _kaggleStartStage.value = "Сервер активен и готов к синтезу!"
                        Toast.makeText(getApplication(), "✓ Сервер Kaggle запущен и готов к генерации!", Toast.LENGTH_LONG).show()
                    } else {
                        // Не удалось получить URL — честно сообщаем, не обманываем isOnline
                        _serverHealth.value = ServerHealth(isOnline = false, version = "Туннель недоступен")
                        _kaggleStartStage.value = "Ошибка: туннель не обнаружен"
                        Toast.makeText(getApplication(), "⚠ Сервер запущен но туннель не найден. Попробуйте перезапустить.", Toast.LENGTH_LONG).show()
                    }
                    break
                } else if (st == "QUEUED") {
                    queuedSeconds += 3
                    if (queuedSeconds >= 150) {
                        _kaggleStartStage.value = "Очередь Kaggle зависла (>2.5 мин). Остановка..."
                        try { kaggleClient.stopKernel() } catch (_: Exception) {}
                        _serverHealth.value = ServerHealth(isOnline = false, version = "Очередь зависла")
                        Toast.makeText(getApplication(), "⚠ Kaggle долго не выделяет GPU (очередь > 2.5 мин). Сессия отменена для экономии квоты.", Toast.LENGTH_LONG).show()
                        break
                    } else {
                        _kaggleStartStage.value = "В очереди Kaggle (${queuedSeconds} сек)... Ожидание GPU..."
                    }
                } else if (st.contains("ERROR") || st.contains("FAILED")) {
                    _serverHealth.value = ServerHealth(isOnline = false, version = "Ошибка")
                    Toast.makeText(getApplication(), "Ошибка ядра: $st", Toast.LENGTH_LONG).show()
                    break
                } else {
                    _kaggleStartStage.value = "Инициализация GPU ($st, ${attempts * 3} сек)..."
                }
            }
            _isKaggleStarting.value = false
            refreshKaggleStatus()
        }
    }

    fun stopKaggleServer() {
        if (_isKaggleStopping.value) return
        startJob?.cancel()
        startJob = null
        _isKaggleStarting.value = false
        _kaggleStartStage.value = ""
        
        stopJob?.cancel()
        stopJob = viewModelScope.launch {
            _isKaggleStopping.value = true
            cancelActiveJob()
            Toast.makeText(getApplication(), "🛑 Остановка сервера Kaggle...", Toast.LENGTH_SHORT).show()

            // 1. Send graceful HTTP /shutdown if server URL is reachable
            try {
                if (engineClient.cleanBaseUrl().isNotBlank()) {
                    engineClient.shutdownServer()
                }
            } catch (_: Exception) {}

            // 2. Send control shutdown signal via ntfy (picked up by runner loop within 6s)
            try {
                engineClient.sendRemoteShutdownSignal()
            } catch (_: Exception) {}

            _serverHealth.value = ServerHealth(isOnline = false, version = "Отключен")
            engineClient.baseUrl = ""
            _serverUrl.value = ""
            _kaggleStatus.value = "COMPLETE"

            delay(2000)
            _isKaggleStopping.value = false
            Toast.makeText(getApplication(), "✓ Сервер остановлен. GPU квота сохранена!", Toast.LENGTH_LONG).show()
            refreshKaggleStatus()
        }
    }

    private var logsPollingJob: Job? = null

    fun openLogsDialog() {
        _showLogsDialog.value = true
        startLogsPolling()
    }

    fun closeLogsDialog() {
        _showLogsDialog.value = false
        logsPollingJob?.cancel()
        logsPollingJob = null
    }

    private fun startLogsPolling() {
        logsPollingJob?.cancel()
        logsPollingJob = viewModelScope.launch {
            while (_showLogsDialog.value) {
                fetchKernelLogsInternal()
                delay(3000)
            }
        }
    }

    fun fetchKernelLogs() {
        viewModelScope.launch {
            fetchKernelLogsInternal()
        }
    }

    private suspend fun fetchKernelLogsInternal() {
        _isLoadingLogs.value = true
        val currentUrl = engineClient.cleanBaseUrl()
        val isOnline = _serverHealth.value.isOnline

        // 1. If server is online or we have a base URL, try live server logs first
        if (isOnline || currentUrl.isNotBlank()) {
            val liveRes = engineClient.fetchLiveServerLogs()
            if (liveRes.isSuccess) {
                val liveLogs = liveRes.getOrNull()
                if (!liveLogs.isNullOrBlank()) {
                    _serverLogs.value = liveLogs
                    _isLoadingLogs.value = false
                    return
                }
            }
        }

        // 2. Fallback to Kaggle kernel status & Kaggle API output
        val statusRes = kaggleClient.getKernelStatus()
        val kaggleStatus = statusRes.getOrNull() ?: "UNKNOWN"

        val kaggleLogRes = kaggleClient.getKernelLogs()
        val rawKaggleLog = kaggleLogRes.getOrNull()
        val hasValidKaggleLog = !rawKaggleLog.isNullOrBlank() && !rawKaggleLog.contains("Логи пока пусты")

        val sb = StringBuilder()
        sb.appendLine("=== Мониторинг Kaggle Kernel ===")
        sb.appendLine("Статус сессии: ${kaggleStatus.uppercase()}")
        if (currentUrl.isNotBlank()) {
            sb.appendLine("URL туннеля: $currentUrl")
        }
        sb.appendLine("=================================")
        if (hasValidKaggleLog && rawKaggleLog != null) {
            sb.appendLine(rawKaggleLog)
        } else {
            when {
                kaggleStatus.equals("running", ignoreCase = true) -> {
                    sb.appendLine("Сессия Kaggle активна и выполняется.")
                    if (currentUrl.isBlank()) {
                        sb.appendLine("Ожидание запуска туннеля Cloudflare...")
                    } else {
                        sb.appendLine("Туннель запущен. Подключение к прямому мониторингу...")
                    }
                    sb.appendLine("\n(Детальные логи GPU и синтеза отображаются здесь в реальном времени при связи с сервером)")
                }
                kaggleStatus.equals("queued", ignoreCase = true) -> {
                    sb.appendLine("Сессия находится в очереди Kaggle на выделение 2x Tesla T4 GPU...")
                }
                kaggleStatus.equals("complete", ignoreCase = true) -> {
                    sb.appendLine("Сессия Kaggle завершена.")
                }
                else -> {
                    sb.appendLine("Статус ядра: $kaggleStatus")
                }
            }
        }
        _serverLogs.value = sb.toString().trim()
        _isLoadingLogs.value = false
    }

    fun cancelActiveJob() {
        pollingJob?.cancel()
        pollingJob = null
        _activeJob.value = ActiveJob(
            id = "stopped",
            title = _activeJob.value?.title ?: "",
            status = "failed",
            errorMessage = "Остановлено пользователем"
        )
        refreshKaggleStatus()
        Toast.makeText(getApplication(), "🛑 Генерация остановлена", Toast.LENGTH_SHORT).show()
    }

    fun generateTrack(
        title: String,
        style: String,
        lyrics: String,
        duration: Int,
        engine: String = defaultEngine,
        steps: Int = 8,
        guidanceScale: Float = 7.0f,
        bpm: Int = 120
    ) {
        if (_activeJob.value?.status == "running") {
            Toast.makeText(getApplication(), "Генерация уже выполняется!", Toast.LENGTH_SHORT).show()
            return
        }

        val defaultTitle = if (engine == "acestep") "ACE-Step Track" else "YuE2 Track"
        val actualTitle = title.ifBlank { defaultTitle }

        viewModelScope.launch {
            val currentUrl = engineClient.cleanBaseUrl()
            if (currentUrl.isNotBlank()) {
                submitTrackToRunningServer(
                    title = actualTitle,
                    style = style,
                    lyrics = lyrics,
                    duration = duration,
                    engine = engine,
                    steps = steps,
                    guidanceScale = guidanceScale,
                    bpm = bpm,
                    abc = null,
                    isCover = false
                )
                return@launch
            }

            val kaggleSt = _kaggleStatus.value?.uppercase() ?: ""
            if (kaggleSt == "RUNNING") {
                Toast.makeText(getApplication(), "🔍 Поиск адреса сервера в сети...", Toast.LENGTH_SHORT).show()
                val discoveredUrl = engineClient.autoDiscoverFromNtfy()
                if (!discoveredUrl.isNullOrBlank()) {
                    engineClient.baseUrl = discoveredUrl
                    _serverUrl.value = discoveredUrl
                    _serverHealth.value = ServerHealth(isOnline = true, version = "2x Tesla T4 GPU", url = discoveredUrl)
                    submitTrackToRunningServer(
                        title = actualTitle,
                        style = style,
                        lyrics = lyrics,
                        duration = duration,
                        engine = engine,
                        steps = steps,
                        guidanceScale = guidanceScale,
                        bpm = bpm,
                        abc = null,
                        isCover = false
                    )
                    return@launch
                }
                Toast.makeText(getApplication(), "⏳ Сервер Kaggle ещё поднимает туннель, подождите 10-15 сек и попробуйте снова", Toast.LENGTH_LONG).show()
                return@launch
            }

            Toast.makeText(getApplication(), "Сначала включите сервер на вкладке «Сервер»!", Toast.LENGTH_LONG).show()
        }
    }

    private fun submitTrackToRunningServer(
        title: String,
        style: String,
        lyrics: String,
        duration: Int,
        engine: String,
        steps: Int,
        guidanceScale: Float,
        bpm: Int,
        abc: String?,
        isCover: Boolean
    ) {
        pollingJob?.cancel()
        viewModelScope.launch {
            try {
                _activeJob.value = ActiveJob(
                    id = "job_${System.currentTimeMillis()}",
                    title = title,
                    status = "running",
                    stage = "Отправка задачи на сервер Kaggle (2x Tesla T4)..."
                )

                var targetUrl = engineClient.cleanBaseUrl()
                if (targetUrl.isBlank()) {
                    val disc = engineClient.autoDiscoverFromNtfy()
                    if (!disc.isNullOrBlank()) {
                        engineClient.baseUrl = disc
                        _serverUrl.value = disc
                    }
                }

                val jobId = try {
                    engineClient.submitGeneration(
                        title = title,
                        style = style,
                        lyrics = lyrics,
                        durationSeconds = duration,
                        engine = engine,
                        steps = steps,
                        guidanceScale = guidanceScale,
                        bpm = bpm,
                        abc = abc
                    )
                } catch (netErr: Exception) {
                    // Try auto-discovery once if current URL expired
                    val freshUrl = engineClient.autoDiscoverFromNtfy()
                    if (!freshUrl.isNullOrBlank() && freshUrl != engineClient.cleanBaseUrl()) {
                        engineClient.baseUrl = freshUrl
                        _serverUrl.value = freshUrl
                        engineClient.submitGeneration(
                            title = title,
                            style = style,
                            lyrics = lyrics,
                            durationSeconds = duration,
                            engine = engine,
                            steps = steps,
                            guidanceScale = guidanceScale,
                            bpm = bpm,
                            abc = abc
                        )
                    } else {
                        throw netErr
                    }
                }

                _activeJob.value = _activeJob.value?.copy(
                    stage = "Параллельный синтез 2 вариантов трека (Задача: $jobId)..."
                )

                pollJobUntilComplete(
                    jobId = jobId,
                    title = title,
                    style = style,
                    lyrics = lyrics,
                    duration = duration,
                    isCover = isCover,
                    engine = engine
                )
            } catch (e: Exception) {
                _activeJob.value = ActiveJob(
                    id = "error",
                    title = title,
                    status = "failed",
                    errorMessage = e.localizedMessage ?: "Ошибка отправки задачи на сервер"
                )
            }
        }
    }

    private fun runZeroTunnelGeneration(
        title: String,
        style: String,
        lyrics: String,
        duration: Int,
        steps: Int,
        engine: String,
        isCover: Boolean = false,
        abc: String? = null
    ) {
        pollingJob?.cancel()
        viewModelScope.launch {
            try {
                _activeJob.value = ActiveJob(
                    id = "batch_${System.currentTimeMillis()}",
                    title = title,
                    status = "running",
                    stage = "Отправка задачи в Kaggle (0 туннелей)..."
                )

                val rndSeed = (Math.random() * 1_000_000_000).toLong()
                val pushRes = kaggleClient.pushBatchKernel(
                    context = getApplication(),
                    title = title,
                    style = style,
                    lyrics = lyrics,
                    duration = duration,
                    steps = steps,
                    seed = rndSeed,
                    abc = abc
                )

                if (pushRes.isFailure) {
                    _activeJob.value = ActiveJob(
                        id = "error",
                        title = title,
                        status = "failed",
                        errorMessage = pushRes.exceptionOrNull()?.message ?: "Ошибка отправки в Kaggle"
                    )
                    return@launch
                }

                _activeJob.value = _activeJob.value?.copy(
                    stage = "Синтез аудио на 2x Tesla T4 (Kaggle Batch, 0 туннелей)..."
                )

                var finished = false
                var attempts = 0
                while (!finished && attempts < 180) {
                    delay(5000)
                    attempts++
                    val stRes = kaggleClient.getKernelStatus()
                    val st = stRes.getOrNull()?.uppercase() ?: ""
                    if (st == "COMPLETE") {
                        finished = true
                        break
                    } else if (st.contains("ERROR") || st.contains("FAILED")) {
                        throw RuntimeException("Синтез на Kaggle завершился с ошибкой: $st")
                    }
                    _activeJob.value = _activeJob.value?.copy(
                        stage = "Синтез на 2x Tesla T4 (${attempts * 5} сек)..."
                    )
                }

                if (!finished) {
                    throw RuntimeException("Таймаут синтеза на Kaggle")
                }

                _activeJob.value = _activeJob.value?.copy(stage = "Загрузка готового MP3 трека...")

                val appDir = getApplication<Application>().getExternalFilesDir(Environment.DIRECTORY_MUSIC)
                    ?: getApplication<Application>().filesDir
                val safeTitle = title.replace(Regex("[^a-zA-Z0-9_\\-\\s]"), "").trim().ifBlank { "track" }
                val mp3File = File(appDir, "${safeTitle}_${System.currentTimeMillis()}.mp3")

                val dlRes = kaggleClient.downloadKernelOutput(destFile = mp3File)
                if (dlRes.isSuccess && mp3File.exists()) {
                    val newTrack = TrackItem(
                        id = "batch_${System.currentTimeMillis()}",
                        title = title,
                        style = style,
                        lyrics = lyrics,
                        durationSeconds = duration,
                        audioPath = mp3File.absolutePath,
                        isCover = isCover,
                        engine = engine
                    )
                    _tracks.value = listOf(newTrack) + _tracks.value
                    _activeJob.value = _activeJob.value?.copy(
                        status = "done",
                        stage = "Готово! Сохранено в библиотеку."
                    )
                    Toast.makeText(getApplication(), "🎉 Трек готов: $title", Toast.LENGTH_LONG).show()
                } else {
                    throw RuntimeException(dlRes.exceptionOrNull()?.message ?: "Не удалось скачать файл")
                }
            } catch (e: Exception) {
                _activeJob.value = ActiveJob(
                    id = "error",
                    title = title,
                    status = "failed",
                    errorMessage = e.localizedMessage
                )
            }
        }
    }

    private fun pollJobUntilComplete(
        jobId: String,
        title: String,
        style: String,
        lyrics: String,
        duration: Int,
        isCover: Boolean,
        engine: String = defaultEngine
    ) {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            var finished = false
            var attempts = 0
            while (!finished && attempts < 360) {
                delay(3000)
                attempts++
                val status = engineClient.pollJobStatus(jobId)
                if (status == "done") {
                    finished = true
                    _activeJob.value = _activeJob.value?.copy(stage = "Скачивание готовых треков...")

                    val appDir = getApplication<Application>().getExternalFilesDir(Environment.DIRECTORY_MUSIC)
                        ?: getApplication<Application>().filesDir
                    val prefix = if (engine == "acestep") "acestep" else "yue2"
                    val safeId = jobId.replace(Regex("[^a-zA-Z0-9_-]"), "_")

                    if (jobId.contains("__")) {
                        val file1 = File(appDir, "${prefix}_${safeId}_var1.mp3")
                        val file2 = File(appDir, "${prefix}_${safeId}_var2.mp3")

                        val dl1 = engineClient.downloadAudioResult(jobId, file1, resultNum = 1)
                        val dl2 = engineClient.downloadAudioResult(jobId, file2, resultNum = 2)

                        val newTracks = mutableListOf<TrackItem>()
                        if (dl1 && file1.exists()) {
                            newTracks.add(TrackItem(
                                id = "${jobId}_var1",
                                title = "$title (Вариант 1)",
                                style = style,
                                lyrics = lyrics,
                                durationSeconds = duration,
                                audioPath = file1.absolutePath,
                                isCover = isCover,
                                engine = engine
                            ))
                        }
                        if (dl2 && file2.exists()) {
                            newTracks.add(TrackItem(
                                id = "${jobId}_var2",
                                title = "$title (Вариант 2)",
                                style = style,
                                lyrics = lyrics,
                                durationSeconds = duration,
                                audioPath = file2.absolutePath,
                                isCover = isCover,
                                engine = engine
                            ))
                        }

                        if (newTracks.isNotEmpty()) {
                            _tracks.value = newTracks + _tracks.value
                            _activeJob.value = _activeJob.value?.copy(
                                status = "done",
                                stage = "Готово! 2 варианта сохранены в библиотеку."
                            )
                            Toast.makeText(getApplication(), "🎉 Создано 2 варианта трека на 2x Tesla T4!", Toast.LENGTH_LONG).show()
                        } else {
                            _activeJob.value = _activeJob.value?.copy(
                                status = "failed",
                                errorMessage = "Не удалось загрузить аудиофайлы с сервера"
                            )
                        }
                    } else {
                        val mp3File = File(appDir, "${prefix}_${safeId}.mp3")
                        val downloaded = engineClient.downloadAudioResult(jobId, mp3File, resultNum = 1)
                        if (downloaded && mp3File.exists()) {
                            val newTrack = TrackItem(
                                id = jobId,
                                title = title,
                                style = style,
                                lyrics = lyrics,
                                durationSeconds = duration,
                                audioPath = mp3File.absolutePath,
                                isCover = isCover,
                                engine = engine
                            )
                            _tracks.value = listOf(newTrack) + _tracks.value
                            _activeJob.value = _activeJob.value?.copy(
                                status = "done",
                                stage = "Готово! Сохранено в библиотеку."
                            )
                            Toast.makeText(getApplication(), "Трек готов: $title", Toast.LENGTH_LONG).show()
                        } else {
                            _activeJob.value = _activeJob.value?.copy(
                                status = "failed",
                                errorMessage = "Не удалось загрузить аудиофайл"
                            )
                        }
                    }
                } else if (status == "failed" || status == "error") {
                    finished = true
                    _activeJob.value = _activeJob.value?.copy(
                        status = "failed",
                        errorMessage = "Ошибка генерации на сервере"
                    )
                } else {
                    _activeJob.value = _activeJob.value?.copy(
                        stage = "Синтез аудио (${attempts * 3} сек)..."
                    )
                }
            }
        }
    }

    fun selectCoverAudio(uri: Uri) {
        _selectedCoverAudioUri.value = uri
        _selectedCoverAudioName.value = getFileNameFromUri(uri)
    }

    fun selectMidiAudio(uri: Uri) {
        _selectedMidiAudioUri.value = uri
        _selectedMidiAudioName.value = getFileNameFromUri(uri)
    }

    fun setTranscribeModelSize(size: String) {
        _transcribeModelSize.value = size
    }

    fun transcribeToMidi() {
        val uri = _selectedMidiAudioUri.value ?: return
        if (_isTranscribing.value) return

        viewModelScope.launch {
            _isTranscribing.value = true
            _transcribeError.value = null
            _transcribeStage.value = "Подготовка аудиофайла..."
            try {
                val tempAudio = copyUriToTempFile(uri, "input_audio")
                _transcribeStage.value = "Отправка на GPU 1 (MuScriptor ${_transcribeModelSize.value.uppercase()})..."
                val downloadDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                val midiFile = File(downloadDir, "yue2_transcription_${System.currentTimeMillis()}.mid")

                val res = engineClient.transcribeAudioToMidi(
                    tempAudio,
                    midiFile,
                    modelSize = _transcribeModelSize.value,
                    onProgress = { stage -> _transcribeStage.value = stage }
                )
                if (res.isSuccess) {
                    val saved = res.getOrThrow()
                    _lastTranscribedMidiFile.value = saved
                    _transcribeStage.value = "Готово! Сохранено: ${saved.name}"
                    Toast.makeText(getApplication(), "MIDI сохранен в Загрузки: ${saved.name}", Toast.LENGTH_LONG).show()
                } else {
                    val err = res.exceptionOrNull()?.message ?: "Ошибка транскрибации"
                    _transcribeError.value = err
                    Toast.makeText(getApplication(), "Ошибка: $err", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                val err = e.localizedMessage ?: "Ошибка выполнения"
                _transcribeError.value = err
                Toast.makeText(getApplication(), "Ошибка: $err", Toast.LENGTH_LONG).show()
            } finally {
                _isTranscribing.value = false
            }
        }
    }

    fun setCoverAbcScore(abc: String) {
        _coverAbcScore.value = abc
    }

    fun setCoverLyrics(lyrics: String) {
        _coverLyrics.value = lyrics
    }

    fun setCoverStyle(style: String) {
        _coverStyle.value = style
    }

    fun setCoverTitle(title: String) {
        _coverTitle.value = title
    }

    fun setCoverMelodyOnly(melodyOnly: Boolean) {
        _coverMelodyOnly.value = melodyOnly
    }

    fun transcribeCoverAudio() {
        val uri = _selectedCoverAudioUri.value ?: run {
            Toast.makeText(getApplication(), "Сначала выберите аудиофайл оригинала!", Toast.LENGTH_SHORT).show()
            return
        }
        if (_isCoverTranscribing.value) return

        viewModelScope.launch {
            _isCoverTranscribing.value = true
            _coverTranscribeError.value = null
            _coverTranscribeStage.value = "Подготовка аудиозаписи..."
            try {
                val tempAudio = copyUriToTempFile(uri, "cover_input")
                _coverTranscribeStage.value = "Транскрибация аудио в ноты (SheetSage2)..."
                val res = engineClient.transcribeAudioToAbc(
                    tempAudio,
                    melodyOnly = _coverMelodyOnly.value,
                    onProgress = { stage -> _coverTranscribeStage.value = stage }
                )
                if (res.isSuccess) {
                    val score = res.getOrThrow()
                    _coverAbcScore.value = score
                    _coverTranscribeStage.value = "Партитура ABC готова!"
                    Toast.makeText(getApplication(), "Партитура успешно извлечена!", Toast.LENGTH_SHORT).show()
                } else {
                    val err = res.exceptionOrNull()?.message ?: "Ошибка транскрибации"
                    _coverTranscribeError.value = err
                    Toast.makeText(getApplication(), "Ошибка: $err", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                val err = e.localizedMessage ?: "Ошибка выполнения"
                _coverTranscribeError.value = err
                Toast.makeText(getApplication(), "Ошибка: $err", Toast.LENGTH_LONG).show()
            } finally {
                _isCoverTranscribing.value = false
            }
        }
    }

    fun createCover(
        title: String,
        targetStyle: String,
        lyrics: String,
        abc: String,
        duration: Int,
        steps: Int = 32
    ) {
        if (_activeJob.value?.status == "running") {
            Toast.makeText(getApplication(), "Генерация уже выполняется!", Toast.LENGTH_SHORT).show()
            return
        }

        val autoTitle = title.ifBlank { "Cover: ${_selectedCoverAudioName.value ?: "Melody"}" }

        viewModelScope.launch {
            val currentUrl = engineClient.cleanBaseUrl()
            if (currentUrl.isNotBlank()) {
                submitTrackToRunningServer(
                    title = autoTitle,
                    style = targetStyle,
                    lyrics = lyrics.ifBlank { "[Verse]\n[Chorus]" },
                    duration = duration,
                    engine = defaultEngine,
                    steps = if (defaultEngine == "acestep") 8 else steps,
                    guidanceScale = 7.0f,
                    bpm = 120,
                    abc = abc.ifBlank { null },
                    isCover = true
                )
                return@launch
            }

            val kaggleSt = _kaggleStatus.value?.uppercase() ?: ""
            if (kaggleSt == "RUNNING") {
                Toast.makeText(getApplication(), "🔍 Поиск адреса сервера в сети...", Toast.LENGTH_SHORT).show()
                val discoveredUrl = engineClient.autoDiscoverFromNtfy()
                if (!discoveredUrl.isNullOrBlank()) {
                    engineClient.baseUrl = discoveredUrl
                    _serverUrl.value = discoveredUrl
                    _serverHealth.value = ServerHealth(isOnline = true, version = "2x Tesla T4 GPU", url = discoveredUrl)
                    submitTrackToRunningServer(
                        title = autoTitle,
                        style = targetStyle,
                        lyrics = lyrics.ifBlank { "[Verse]\n[Chorus]" },
                        duration = duration,
                        engine = defaultEngine,
                        steps = if (defaultEngine == "acestep") 8 else steps,
                        guidanceScale = 7.0f,
                        bpm = 120,
                        abc = abc.ifBlank { null },
                        isCover = true
                    )
                    return@launch
                }
                Toast.makeText(getApplication(), "⏳ Сервер Kaggle ещё поднимает туннель, подождите 10-15 сек и попробуйте снова", Toast.LENGTH_LONG).show()
                return@launch
            }

            Toast.makeText(getApplication(), "Сначала включите сервер на вкладке «Сервер»!", Toast.LENGTH_LONG).show()
        }
    }

    fun playTrack(track: TrackItem) {
        track.audioPath?.let { path ->
            val file = File(path)
            if (file.exists()) {
                audioPlayer.playFile(file, track.title)
            } else {
                Toast.makeText(getApplication(), "Файл не найден на диске", Toast.LENGTH_SHORT).show()
            }
        }
    }

    fun refreshLibrary() {
        val dirs = listOfNotNull(
            getApplication<Application>().getExternalFilesDir(Environment.DIRECTORY_MUSIC),
            getApplication<Application>().getExternalFilesDir(null),
            getApplication<Application>().filesDir,
            getApplication<Application>().cacheDir
        )
        val allFiles = mutableListOf<File>()
        for (dir in dirs) {
            val list = dir.listFiles { f ->
                f.extension.equals("mp3", ignoreCase = true) || f.extension.equals("wav", ignoreCase = true)
            }
            if (list != null) {
                allFiles.addAll(list)
            }
        }
        val loaded = allFiles.distinctBy { it.name }.sortedByDescending { it.lastModified() }.map { file ->
            val name = file.nameWithoutExtension
            val safeName = name
                .replace(Regex("_var1$"), " (Вариант 1)")
                .replace(Regex("_var2$"), " (Вариант 2)")
                .replace(Regex("^yue2_"), "")
                .replace(Regex("^acestep_"), "")
                .replace(Regex("_[a-f0-9]{8,}$"), "")
                .replace("_", " ")
                .trim()
            TrackItem(
                id = file.nameWithoutExtension,
                title = safeName.ifBlank { file.nameWithoutExtension },
                style = "Аудиозапись",
                lyrics = "",
                durationSeconds = 180,
                audioPath = file.absolutePath,
                isCover = file.name.contains("cover", ignoreCase = true),
                engine = if (file.name.contains("acestep", ignoreCase = true)) "acestep" else "yue2"
            )
        }
        _tracks.value = loaded
    }

    private fun getFileNameFromUri(uri: Uri): String {
        var name = "audio_file"
        getApplication<Application>().contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (nameIndex != -1 && cursor.moveToFirst()) {
                name = cursor.getString(nameIndex)
            }
        }
        return name
    }

    private suspend fun copyUriToTempFile(uri: Uri, prefix: String): File = withContext(Dispatchers.IO) {
        val context = getApplication<Application>()
        val ext = when (context.contentResolver.getType(uri)) {
            "audio/wav", "audio/x-wav" -> ".wav"
            "audio/ogg" -> ".ogg"
            "audio/flac" -> ".flac"
            else -> ".mp3"
        }
        val tempFile = File.createTempFile(prefix, ext, context.cacheDir)
        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(tempFile).use { output ->
                input.copyTo(output)
            }
        }
        tempFile
    }

    override fun onCleared() {
        super.onCleared()
        audioPlayer.stop()
    }
}
