package com.example.yue2studio.ui.main

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.yue2studio.data.ActiveJob
import com.example.yue2studio.data.ServerHealth
import com.example.yue2studio.data.TrackItem
import com.example.yue2studio.player.PlaybackState
import com.example.yue2studio.ui.MainViewModel
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: MainViewModel = viewModel(),
    modifier: Modifier = Modifier
) {
    val isAceStep = viewModel.appFlavor == "acestep"
    val serverTabIdx = if (isAceStep) 4 else 3
    var selectedTab by remember { mutableIntStateOf(0) }
    val serverHealth by viewModel.serverHealth.collectAsStateWithLifecycle()
    val activeJob by viewModel.activeJob.collectAsStateWithLifecycle()
    val playbackState by viewModel.audioPlayer.playbackState.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = if (isAceStep) Icons.Default.Bolt else Icons.Default.MusicNote,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary
                        )
                        Column {
                            Text(
                                if (isAceStep) "ACE-Step Studio" else "YuE2 Studio",
                                fontWeight = FontWeight.Bold,
                                fontSize = 18.sp
                            )
                            Text(
                                if (isAceStep) "Kaggle Tesla T4 Engine • Turbo DiT" else "Kaggle Tesla T4 Engine • Deep LM",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                },
                actions = {
                    ServerStatusBadge(
                        health = serverHealth,
                        onClick = { selectedTab = serverTabIdx }
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        },
        bottomBar = {
            Column {
                if (playbackState.isPlaying || playbackState.currentPositionMs > 0) {
                    MiniPlayerBar(
                        playbackState = playbackState,
                        onPlayPause = {
                            if (playbackState.isPlaying) viewModel.audioPlayer.pause()
                            else viewModel.audioPlayer.resume()
                        },
                        onStop = { viewModel.audioPlayer.stop() },
                        onSeek = { viewModel.audioPlayer.seekTo(it) }
                    )
                }

                NavigationBar {
                    if (isAceStep) {
                        NavigationBarItem(
                            selected = selectedTab == 0,
                            onClick = { selectedTab = 0 },
                            icon = { Icon(Icons.Default.Bolt, contentDescription = "Генерация") },
                            label = { Text("Трек") }
                        )
                        NavigationBarItem(
                            selected = selectedTab == 1,
                            onClick = { selectedTab = 1 },
                            icon = { Icon(Icons.Default.AutoFixHigh, contentDescription = "Кавер") },
                            label = { Text("Кавер") }
                        )
                        NavigationBarItem(
                            selected = selectedTab == 2,
                            onClick = { selectedTab = 2 },
                            icon = { Icon(Icons.Default.GraphicEq, contentDescription = "MIDI") },
                            label = { Text("MIDI") }
                        )
                        NavigationBarItem(
                            selected = selectedTab == 3,
                            onClick = { selectedTab = 3 },
                            icon = { Icon(Icons.Default.LibraryMusic, contentDescription = "Библиотека") },
                            label = { Text("Библиотека") }
                        )
                        NavigationBarItem(
                            selected = selectedTab == 4,
                            onClick = { selectedTab = 4 },
                            icon = { Icon(Icons.Default.CloudQueue, contentDescription = "Сервер") },
                            label = { Text("Сервер") }
                        )
                    } else {
                        NavigationBarItem(
                            selected = selectedTab == 0,
                            onClick = { selectedTab = 0 },
                            icon = { Icon(Icons.Default.MusicNote, contentDescription = "Генерация") },
                            label = { Text("Трек") }
                        )
                        NavigationBarItem(
                            selected = selectedTab == 1,
                            onClick = { selectedTab = 1 },
                            icon = { Icon(Icons.Default.AutoFixHigh, contentDescription = "Кавер") },
                            label = { Text("Кавер") }
                        )
                        NavigationBarItem(
                            selected = selectedTab == 2,
                            onClick = { selectedTab = 2 },
                            icon = { Icon(Icons.Default.LibraryMusic, contentDescription = "Библиотека") },
                            label = { Text("Библиотека") }
                        )
                        NavigationBarItem(
                            selected = selectedTab == 3,
                            onClick = { selectedTab = 3 },
                            icon = { Icon(Icons.Default.CloudQueue, contentDescription = "Сервер") },
                            label = { Text("Сервер") }
                        )
                    }
                }
            }
        }
    ) { paddingValues ->
        Box(
            modifier = modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            if (isAceStep) {
                when (selectedTab) {
                    0 -> PromptGenerationTab(viewModel = viewModel, activeJob = activeJob)
                    1 -> CoverGenerationTab(viewModel = viewModel, activeJob = activeJob)
                    2 -> AudioToMidiTab(viewModel = viewModel)
                    3 -> LibraryTab(viewModel = viewModel)
                    4 -> ServerSettingsTab(viewModel = viewModel, serverHealth = serverHealth)
                }
            } else {
                when (selectedTab) {
                    0 -> PromptGenerationTab(viewModel = viewModel, activeJob = activeJob)
                    1 -> CoverGenerationTab(viewModel = viewModel, activeJob = activeJob)
                    2 -> LibraryTab(viewModel = viewModel)
                    3 -> ServerSettingsTab(viewModel = viewModel, serverHealth = serverHealth)
                }
            }
        }
    }
}

@Composable
fun ServerStatusBadge(health: ServerHealth, onClick: () -> Unit) {
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = if (health.isOnline) Color(0xFF064E3B) else Color(0xFF374151),
        modifier = Modifier
            .padding(end = 12.dp)
            .clickable { onClick() }
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(if (health.isOnline) Color(0xFF34D399) else Color(0xFF9CA3AF))
            )
            Text(
                text = if (health.isOnline) "🟢 Сервер: On" else "⚪ Сервер: Off",
                fontSize = 12.sp,
                color = if (health.isOnline) Color(0xFF6EE7B7) else Color(0xFFD1D5DB),
                fontWeight = FontWeight.Medium
            )
        }
    }
}

@Composable
fun MiniPlayerBar(
    playbackState: PlaybackState,
    onPlayPause: () -> Unit,
    onStop: () -> Unit,
    onSeek: (Int) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 4.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        elevation = CardDefaults.cardElevation(defaultElevation = 6.dp)
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = playbackState.currentTrackTitle.ifBlank { "Воспроизведение" },
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Text(
                        text = "${formatDuration(playbackState.currentPositionMs.toLong())} / ${formatDuration(playbackState.durationMs.toLong())}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = onPlayPause) {
                        Icon(
                            imageVector = if (playbackState.isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                            contentDescription = "Play/Pause",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                    IconButton(onClick = onStop) {
                        Icon(
                            imageVector = Icons.Default.Stop,
                            contentDescription = "Stop"
                        )
                    }
                }
            }

            if (playbackState.durationMs > 0) {
                Slider(
                    value = playbackState.currentPositionMs.toFloat().coerceIn(0f, playbackState.durationMs.toFloat()),
                    onValueChange = { onSeek(it.toInt()) },
                    valueRange = 0f..playbackState.durationMs.toFloat(),
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(20.dp)
                )
            }
        }
    }
}

@Composable
fun ActiveJobCard(job: ActiveJob?) {
    if (job == null || job.status == "done") return

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (job.status == "failed") MaterialTheme.colorScheme.errorContainer
            else MaterialTheme.colorScheme.secondaryContainer
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = job.title,
                    fontWeight = FontWeight.Bold,
                    fontSize = 15.sp
                )
                if (job.status == "running") {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                }
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = if (job.status == "failed") "Ошибка: ${job.errorMessage}" else job.stage,
                style = MaterialTheme.typography.bodyMedium,
                color = if (job.status == "failed") MaterialTheme.colorScheme.onErrorContainer
                else MaterialTheme.colorScheme.onSecondaryContainer
            )
            if (job.status == "running") {
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
fun PromptGenerationTab(
    viewModel: MainViewModel,
    activeJob: ActiveJob?
) {
    val isAceStep = viewModel.appFlavor == "acestep"
    val selectedEngine = if (isAceStep) "acestep" else "yue2"
    val title by viewModel.trackTitle.collectAsStateWithLifecycle()
    val style by viewModel.trackStyle.collectAsStateWithLifecycle()
    val lyrics by viewModel.trackLyrics.collectAsStateWithLifecycle()
    val durationSeconds by viewModel.trackDuration.collectAsStateWithLifecycle()
    val steps by viewModel.trackSteps.collectAsStateWithLifecycle()
    var guidanceScale by remember { mutableFloatStateOf(7.0f) }
    var bpm by remember { mutableIntStateOf(120) }

    val aceStylePresets = listOf(
        "Synthwave 80s" to "Synthwave, 80s retro, analog synths, punchy drums, neon vibe, driving bass",
        "Modern Pop" to "Modern radio pop, catchy hook, bright vocals, polished production, deep bass",
        "Cyberpunk Metal" to "Cyberpunk industrial metal, heavy distorted guitars, electronic pulse, fast",
        "Russian Disco" to "Soviet 80s disco pop, energetic synths, disco beat, melodic brass, joyful",
        "EDM Festival" to "EDM festival anthemic house, emotional buildup, powerful melodic drop, 128 bpm",
        "Lo-Fi Chill" to "Lo-Fi hip hop, relaxing Fender Rhodes, warm vinyl crackle, gentle beat",
        "Acoustic Folk" to "Acoustic folk ballad, acoustic guitar, warm cello, gentle and intimate"
    )

    val yueStylePresets = listOf(
        "Rock 80s" to "Rock 80s, driving bass, synthesizer, hard guitar",
        "Synthwave" to "Synthwave, 80s retro, analog synths, punchy drums, neon vibe",
        "Lo-Fi Chill" to "Lo-Fi hip hop, relaxing electric piano, vinyl crackle, mellow beat",
        "Soviet Pop" to "Soviet 80s pop, energetic synths, disco beat, melodic brass",
        "Cyberpunk Metal" to "Cyberpunk industrial metal, heavy distortion, electronic pulse, fast",
        "Acoustic Folk" to "Acoustic folk ballad, acoustic guitar, warm cello, gentle",
        "EDM Club" to "EDM festival house, uplifting chords, energetic drop, punchy 128 bpm"
    )

    val lyricTags = listOf("[Intro]", "[Verse 1]", "[Verse 2]", "[Chorus]", "[Bridge]", "[Guitar Solo]", "[Outro]")

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        ActiveJobCard(job = activeJob)

        Card(
            colors = CardDefaults.cardColors(
                containerColor = if (isAceStep) MaterialTheme.colorScheme.primaryContainer
                else MaterialTheme.colorScheme.secondaryContainer
            ),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier.padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .clip(CircleShape)
                        .background(
                            if (isAceStep) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.secondary
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (isAceStep) Icons.Default.Bolt else Icons.Default.MusicNote,
                        contentDescription = null,
                        tint = if (isAceStep) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSecondary
                    )
                }
                Column {
                    Text(
                        if (isAceStep) "ACE-Step 1.5 Turbo" else "YuE2 Q8_0 Dual-Stage",
                        fontWeight = FontWeight.Bold,
                        fontSize = 15.sp,
                        color = if (isAceStep) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSecondaryContainer
                    )
                    Text(
                        if (isAceStep) "Сверхбыстрый диффузионный синтез на GPU 1 (~30 сек)"
                        else "Глубокая языковая композиция (AR + Flow-Matching, ~8 мин)",
                        fontSize = 12.sp,
                        color = if (isAceStep) MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                        else MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.8f)
                    )
                }
            }
        }

        OutlinedTextField(
            value = title,
            onValueChange = { viewModel.setTrackTitle(it) },
            label = { Text("Название трека") },
            placeholder = { Text(if (isAceStep) "Например: Киберпанк Город" else "Например: Ночной Неон") },
            trailingIcon = if (title.isNotBlank()) {
                {
                    IconButton(onClick = { viewModel.setTrackTitle("") }) {
                        Icon(Icons.Default.Clear, contentDescription = "Очистить название")
                    }
                }
            } else null,
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )

        OutlinedTextField(
            value = style,
            onValueChange = { viewModel.setTrackStyle(it) },
            label = { Text(if (selectedEngine == "acestep") "Стиль, теги и настроение (ACE-Step)" else "Стиль музыки и инструменты (YuE2)") },
            trailingIcon = if (style.isNotBlank()) {
                {
                    IconButton(onClick = { viewModel.setTrackStyle("") }) {
                        Icon(Icons.Default.Clear, contentDescription = "Очистить стиль")
                    }
                }
            } else null,
            modifier = Modifier.fillMaxWidth(),
            minLines = 2
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Быстрые пресеты стилей:", style = MaterialTheme.typography.bodySmall)
            if (style.isNotBlank()) {
                TextButton(
                    onClick = { viewModel.setTrackStyle("") },
                    contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp)
                ) {
                    Icon(Icons.Default.DeleteOutline, contentDescription = null, modifier = Modifier.size(14.dp), tint = MaterialTheme.colorScheme.error)
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Очистить", fontSize = 11.sp, color = MaterialTheme.colorScheme.error)
                }
            }
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            val presets = if (selectedEngine == "acestep") aceStylePresets else yueStylePresets
            presets.forEach { (name, prompt) ->
                AssistChip(
                    onClick = { viewModel.setTrackStyle(prompt) },
                    label = { Text(name, fontSize = 12.sp) }
                )
            }
        }

        OutlinedTextField(
            value = lyrics,
            onValueChange = { viewModel.setTrackLyrics(it) },
            label = { Text("Текст песни и структура") },
            trailingIcon = if (lyrics.isNotBlank()) {
                {
                    IconButton(onClick = { viewModel.setTrackLyrics("") }) {
                        Icon(Icons.Default.Clear, contentDescription = "Очистить текст")
                    }
                }
            } else null,
            modifier = Modifier.fillMaxWidth(),
            minLines = 6
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Теги структуры песни:", style = MaterialTheme.typography.bodySmall)
            if (lyrics.isNotBlank()) {
                TextButton(
                    onClick = { viewModel.setTrackLyrics("") },
                    contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp)
                ) {
                    Icon(Icons.Default.DeleteOutline, contentDescription = null, modifier = Modifier.size(14.dp), tint = MaterialTheme.colorScheme.error)
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Очистить", fontSize = 11.sp, color = MaterialTheme.colorScheme.error)
                }
            }
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            lyricTags.forEach { tag ->
                AssistChip(
                    onClick = { viewModel.setTrackLyrics(if (lyrics.endsWith("\n")) "$lyrics$tag\n" else "$lyrics\n$tag\n") },
                    label = { Text(tag, fontSize = 12.sp) }
                )
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Длительность:")
            Text(
                "${durationSeconds.toInt()} сек (${durationSeconds.toInt() / 60}:${String.format("%02d", durationSeconds.toInt() % 60)})",
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
        }

        Slider(
            value = durationSeconds,
            onValueChange = { viewModel.setTrackDuration(it) },
            valueRange = 30f..360f,
            steps = 10,
            modifier = Modifier.fillMaxWidth()
        )

        if (selectedEngine == "acestep") {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Шаги DiT:")
                Text(
                    "$steps (Turbo: 8)",
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
            Slider(
                value = steps.toFloat(),
                onValueChange = { viewModel.setTrackSteps(it.toInt()) },
                valueRange = 8f..50f,
                steps = 5,
                modifier = Modifier.fillMaxWidth()
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Сила промпта (CFG):")
                Text(
                    String.format("%.1f", guidanceScale),
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
            Slider(
                value = guidanceScale,
                onValueChange = { guidanceScale = it },
                valueRange = 1f..12f,
                steps = 10,
                modifier = Modifier.fillMaxWidth()
            )
        } else {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Шаги (Flow-Matching):")
                Text(
                    "$steps (Реком. 32)",
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
            Slider(
                value = steps.toFloat(),
                onValueChange = { viewModel.setTrackSteps(it.toInt()) },
                valueRange = 20f..50f,
                steps = 6,
                modifier = Modifier.fillMaxWidth()
            )
        }

        Button(
            onClick = {
                viewModel.generateTrack(
                    title = title,
                    style = style,
                    lyrics = lyrics,
                    duration = durationSeconds.toInt(),
                    engine = selectedEngine,
                    steps = steps,
                    guidanceScale = guidanceScale,
                    bpm = bpm
                )
            },
            enabled = activeJob?.status != "running",
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(
                imageVector = if (isAceStep) Icons.Default.Bolt else Icons.Default.MusicNote,
                contentDescription = null
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                if (isAceStep)
                    "Сгенерировать ACE-Step 1.5 (~30с)"
                else
                    "Сгенерировать YuE2 3B Q8 (~8 мин)",
                fontSize = 14.sp,
                maxLines = 1
            )
        }

        Spacer(modifier = Modifier.height(24.dp))
    }
}

@Composable
fun CoverGenerationTab(
    viewModel: MainViewModel,
    activeJob: ActiveJob?
) {
    val selectedCoverUri by viewModel.selectedCoverAudioUri.collectAsStateWithLifecycle()
    val selectedCoverName by viewModel.selectedCoverAudioName.collectAsStateWithLifecycle()
    val coverAbcScore by viewModel.coverAbcScore.collectAsStateWithLifecycle()
    val isCoverTranscribing by viewModel.isCoverTranscribing.collectAsStateWithLifecycle()
    val coverTranscribeStage by viewModel.coverTranscribeStage.collectAsStateWithLifecycle()
    val coverTranscribeError by viewModel.coverTranscribeError.collectAsStateWithLifecycle()
    val coverLyrics by viewModel.coverLyrics.collectAsStateWithLifecycle()
    val coverStyle by viewModel.coverStyle.collectAsStateWithLifecycle()
    val coverTitle by viewModel.coverTitle.collectAsStateWithLifecycle()
    val coverMelodyOnly by viewModel.coverMelodyOnly.collectAsStateWithLifecycle()

    var durationSeconds by remember { mutableFloatStateOf(150f) }

    val filePicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { viewModel.selectCoverAudio(it) }
    }

    val coverPresets = listOf(
        "Acoustic Ballad" to "Emotional acoustic ballad, gentle grand piano, warm strings, intimate soulful vocals",
        "80s Synth Pop" to "80s retro synth pop, analog Juno synthesizers, punchy gated reverb drums, bright upbeat chorus",
        "Hard Rock" to "Hard rock anthem cover, overdrive electric guitars, pounding live drums, powerful raspy vocals",
        "Cyber Metal" to "Industrial metal cover, heavy 8-string riffs, aggressive drop, electronic dark synth bass",
        "Jazz Lounge" to "Smooth jazz lounge cover, upright acoustic bass, brushed snare drums, tenor saxophone, warm Rhodes piano",
        "Lo-Fi Chill" to "Lo-fi hip hop chill cover, vintage vinyl crackle, mellow electric piano, relaxed laid-back beat",
        "Symphonic" to "Cinematic symphonic rock, epic orchestral strings, brass section, soaring lead melody"
    )

    val demoScores = listOf(
        "Поп-баллада (C)" to "X:1\nT:Ballad Melody\nM:4/4\nL:1/8\nQ:1/4=90\nK:C\nV:Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nV:Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n[V:Vocal] [Intro]\n| C2 E2 G2 c2 | e4 d2 c2 | B2 A2 G4 | E4 D2 C2 |\n[V:Vocal] [Verse 1]\n| c4 B2 A2 | G4 E4 | F2 G2 A2 B2 | c8 |",
        "Рок-рифф (Am)" to "X:1\nT:Rock Riff\nM:4/4\nL:1/8\nQ:1/4=130\nK:Am\nV:Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nV:Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n[V:Vocal] [Intro]\n| A,2 C2 E2 A2 | c4 B2 A2 | G2 E2 D2 C2 | A,8 |\n[V:Vocal] [Verse 1]\n| A2 A2 c2 d2 | e4 d2 c2 | A2 c2 e2 g2 | a8 |",
        "Джаз (Dm)" to "X:1\nT:Jazz Standard\nM:4/4\nL:1/8\nQ:1/4=110\nK:Dm\nV:Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nV:Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n[V:Vocal] [Intro]\n| D2 F2 A2 d2 | f4 e2 d2 | ^c2 A2 G2 F2 | D8 |\n[V:Vocal] [Verse 1]\n| d2 e2 f2 g2 | a4 g2 f2 | e2 f2 g2 a2 | d8 |"
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        ActiveJobCard(job = activeJob)

        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Icon(Icons.Default.AutoFixHigh, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
            Text(
                "Кавер-версия (Символический пайплайн)",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
        }

        // ШАГ 1: Аудиозапись оригинала
        Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    "1. Исходная аудиозапись оригинала",
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 15.sp
                )
                Text(
                    "Выберите песню или вокальную дорожку. SheetSage2 снимет ноты и ритм мелодии в партитуру ABC.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                if (selectedCoverUri != null) {
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = MaterialTheme.colorScheme.surface,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.padding(10.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Icon(Icons.Default.AudioFile, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                            Text(
                                selectedCoverName ?: "Аудиофайл оригинала",
                                fontWeight = FontWeight.Bold,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                }

                OutlinedButton(
                    onClick = { filePicker.launch("audio/*") },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(Icons.Default.UploadFile, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(if (selectedCoverUri != null) "Заменить аудиофайл" else "Выбрать аудиозапись песни")
                }

                // Переключатель режима транскрипции
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    FilterChip(
                        selected = coverMelodyOnly,
                        onClick = { viewModel.setCoverMelodyOnly(true) },
                        label = { Text("Только мелодия", fontSize = 12.sp) }
                    )
                    FilterChip(
                        selected = !coverMelodyOnly,
                        onClick = { viewModel.setCoverMelodyOnly(false) },
                        label = { Text("Мелодия + аккорды", fontSize = 12.sp) }
                    )
                }

                // Кнопка запуска SheetSage2
                Button(
                    onClick = { viewModel.transcribeCoverAudio() },
                    enabled = selectedCoverUri != null && !isCoverTranscribing,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    if (isCoverTranscribing) {
                        CircularProgressIndicator(
                            color = MaterialTheme.colorScheme.onPrimary,
                            strokeWidth = 2.dp,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Транскрибация...")
                    } else {
                        Icon(Icons.Default.MusicNote, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(if (coverAbcScore.isNotBlank()) "Обновить ноты через SheetSage2" else "Транскрибировать в ноты (SheetSage2)")
                    }
                }

                if (isCoverTranscribing) {
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        Text(
                            coverTranscribeStage.ifBlank { "Анализ аудиозаписи..." },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }

                if (coverTranscribeError != null) {
                    Text(
                        "Ошибка: $coverTranscribeError",
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }

        // ШАГ 2: Символическая партитура ABC
        Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "2. Нотная партитура (Диалект ABC)",
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 15.sp
                    )
                    if (coverAbcScore.isNotBlank()) {
                        Text(
                            "✓ Ноты готовы",
                            color = Color(0xFF2DD4BF),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
                Text(
                    "YuE2 строго следует этой нотной сетке. Ноты и разметку тактов можно редактировать вручную.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                OutlinedTextField(
                    value = coverAbcScore,
                    onValueChange = { viewModel.setCoverAbcScore(it) },
                    placeholder = {
                        Text(
                            "X:1\nT:Melody\nM:4/4\nL:1/8\nQ:1/4=120\nK:C\n[V:Vocal] [Intro]\n| C2 E2 G2 c2 | e4 d2 c2 |\n[V:Vocal] [Verse 1]\n| c4 B2 A2 | G4 E4 |",
                            fontFamily = FontFamily.Monospace,
                            fontSize = 12.sp
                        )
                    },
                    modifier = Modifier.fillMaxWidth(),
                    textStyle = androidx.compose.ui.text.TextStyle(
                        fontFamily = FontFamily.Monospace,
                        fontSize = 12.sp
                    ),
                    minLines = 5,
                    maxLines = 10
                )

                Text("Примеры готовых партитур (для теста без аудио):", style = MaterialTheme.typography.bodySmall)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    demoScores.forEach { (name, score) ->
                        AssistChip(
                            onClick = { viewModel.setCoverAbcScore(score) },
                            label = { Text(name, fontSize = 11.sp) }
                        )
                    }
                    if (coverAbcScore.isNotBlank()) {
                        AssistChip(
                            onClick = { viewModel.setCoverAbcScore("") },
                            label = { Text("Очистить", fontSize = 11.sp, color = MaterialTheme.colorScheme.error) }
                        )
                    }
                }
            }
        }

        // ШАГ 3: Слова песни (Lyrics)
        Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    "3. Слова песни под партитуру (Lyrics)",
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 15.sp
                )
                Text(
                    "Слова или слоговый текст оригинала. Слоги должны ритмически ложиться на ноты партитуры.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                OutlinedTextField(
                    value = coverLyrics,
                    onValueChange = { viewModel.setCoverLyrics(it) },
                    placeholder = { Text("[Verse 1]\nСлова куплета...\n\n[Chorus]\nСлова припева...") },
                    trailingIcon = if (coverLyrics.isNotBlank()) {
                        {
                            IconButton(onClick = { viewModel.setCoverLyrics("") }) {
                                Icon(Icons.Default.Clear, contentDescription = "Очистить слова кавера")
                            }
                        }
                    } else null,
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 4,
                    maxLines = 8
                )
            }
        }

        // ШАГ 4: Новый целевой стиль (Style)
        Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    "4. Желаемый стиль кавер-версии (Style)",
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 15.sp
                )
                Text(
                    "В каком жанре, тембре и аранжировке модель споёт исходную мелодию:",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                OutlinedTextField(
                    value = coverStyle,
                    onValueChange = { viewModel.setCoverStyle(it) },
                    placeholder = { Text("Например: acoustic indie folk, female vocals, fingerstyle acoustic guitar") },
                    trailingIcon = if (coverStyle.isNotBlank()) {
                        {
                            IconButton(onClick = { viewModel.setCoverStyle("") }) {
                                Icon(Icons.Default.Clear, contentDescription = "Очистить стиль кавера")
                            }
                        }
                    } else null,
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Пресеты жанров:", style = MaterialTheme.typography.bodySmall)
                    if (coverStyle.isNotBlank()) {
                        TextButton(
                            onClick = { viewModel.setCoverStyle("") },
                            contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp)
                        ) {
                            Icon(Icons.Default.DeleteOutline, contentDescription = null, modifier = Modifier.size(14.dp), tint = MaterialTheme.colorScheme.error)
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Очистить", fontSize = 11.sp, color = MaterialTheme.colorScheme.error)
                        }
                    }
                }
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    coverPresets.forEach { (name, prompt) ->
                        AssistChip(
                            onClick = { viewModel.setCoverStyle(prompt) },
                            label = { Text(name, fontSize = 12.sp) }
                        )
                    }
                }
            }
        }

        // ШАГ 5: Параметры и название
        OutlinedTextField(
            value = coverTitle,
            onValueChange = { viewModel.setCoverTitle(it) },
            label = { Text("Название кавер-трека") },
            placeholder = { Text(if (selectedCoverName != null) "Cover: $selectedCoverName" else "My Cover Track") },
            trailingIcon = if (coverTitle.isNotBlank()) {
                {
                    IconButton(onClick = { viewModel.setCoverTitle("") }) {
                        Icon(Icons.Default.Clear, contentDescription = "Очистить название")
                    }
                }
            } else null,
            modifier = Modifier.fillMaxWidth()
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Длительность кавера:")
            Text(
                "${durationSeconds.toInt()} сек (${durationSeconds.toInt() / 60}:${String.format("%02d", durationSeconds.toInt() % 60)})",
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
        }

        Slider(
            value = durationSeconds,
            onValueChange = { durationSeconds = it },
            valueRange = 30f..360f,
            steps = 10,
            modifier = Modifier.fillMaxWidth()
        )

        val canGenerate = (coverAbcScore.isNotBlank() || selectedCoverUri != null) && activeJob?.status != "running"

        Button(
            onClick = {
                viewModel.createCover(
                    title = coverTitle,
                    targetStyle = coverStyle,
                    lyrics = coverLyrics,
                    abc = coverAbcScore,
                    duration = durationSeconds.toInt(),
                    steps = 32
                )
            },
            enabled = canGenerate,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(
                imageVector = if (viewModel.appFlavor == "acestep") Icons.Default.Bolt else Icons.Default.AutoFixHigh,
                contentDescription = null
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                if (viewModel.appFlavor == "acestep")
                    "⚡ Создать кавер ACE-Step (~30 сек)"
                else
                    "🎼 Создать кавер YuE2 по партитуре (~8 мин)",
                fontWeight = FontWeight.Bold,
                fontSize = 15.sp
            )
        }

        Spacer(modifier = Modifier.height(24.dp))
    }
}

@Composable
fun AudioToMidiTab(viewModel: MainViewModel) {
    val context = LocalContext.current
    val selectedMidiUri by viewModel.selectedMidiAudioUri.collectAsStateWithLifecycle()
    val selectedMidiName by viewModel.selectedMidiAudioName.collectAsStateWithLifecycle()
    val lastMidiFile by viewModel.lastTranscribedMidiFile.collectAsStateWithLifecycle()
    val isTranscribing by viewModel.isTranscribing.collectAsStateWithLifecycle()
    val transcribeStage by viewModel.transcribeStage.collectAsStateWithLifecycle()
    val transcribeError by viewModel.transcribeError.collectAsStateWithLifecycle()
    val transcribeModelSize by viewModel.transcribeModelSize.collectAsStateWithLifecycle()

    val filePicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { viewModel.selectMidiAudio(it) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            "Транскрибация Аудио в MIDI (MuScriptor)",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold
        )

        Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text(
                    "Нейросетевая транскрибация аудио в MIDI на GPU 1 (Tesla T4 #2). Извлекает ноты, аккорды, полифонию и темп.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(12.dp))

                if (selectedMidiUri != null) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(Icons.Default.AudioFile, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Text(
                            selectedMidiName ?: "Аудиозапись выбрана",
                            fontWeight = FontWeight.Bold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }

                OutlinedButton(
                    onClick = { filePicker.launch("audio/*") },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(Icons.Default.UploadFile, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(if (selectedMidiUri != null) "Выбрать другую аудиозапись" else "Выбрать аудиозапись (.wav/.mp3/.flac)")
                }
            }
        }

        Text("Модель MuScriptor (качество MIDI):", fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            listOf("large" to "Large 5GB (Макс)", "medium" to "Medium (1GB)", "small" to "Small (300MB)").forEach { (key, label) ->
                FilterChip(
                    selected = transcribeModelSize == key,
                    onClick = { viewModel.setTranscribeModelSize(key) },
                    label = { Text(label, fontSize = 12.sp) },
                    modifier = Modifier.weight(1f)
                )
            }
        }

        if (isTranscribing) {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(28.dp))
                    Column {
                        Text(
                            if (transcribeStage.isNotBlank()) transcribeStage else "Транскрибация на GPU 1...",
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp
                        )
                        Text("Извлечение нот и квантизация MIDI-дорожки", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }

        if (transcribeError != null) {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF7F1D1D)),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(Icons.Default.Error, contentDescription = null, tint = Color(0xFFF87171))
                        Text("Ошибка транскрибации", fontWeight = FontWeight.Bold, color = Color.White)
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(transcribeError ?: "", fontSize = 12.sp, color = Color(0xFFFECACA))
                }
            }
        }

        Button(
            onClick = { viewModel.transcribeToMidi() },
            enabled = selectedMidiUri != null && !isTranscribing,
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(Icons.Default.GraphicEq, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Преобразовать в MIDI (${transcribeModelSize.uppercase()})")
        }

        if (lastMidiFile != null && lastMidiFile!!.exists()) {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF14532D)),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(Icons.Default.CheckCircle, contentDescription = null, tint = Color(0xFF4ADE80))
                        Text(
                            "🎉 MIDI-файл готов и сохранен!",
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFFF0FDF4)
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        "📄 Файл: ${lastMidiFile!!.name}",
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 13.sp,
                        color = Color(0xFFDCFCE7)
                    )
                    Text(
                        "📊 Размер: ${lastMidiFile!!.length() / 1024} КБ (${lastMidiFile!!.length()} байт)",
                        fontSize = 12.sp,
                        color = Color(0xFFBBF7D0)
                    )
                    Text(
                        "📁 Папка: Загрузки (Download)",
                        fontSize = 11.sp,
                        color = Color(0xFF86EFAC)
                    )
                    Spacer(modifier = Modifier.height(12.dp))

                    Button(
                        onClick = {
                            val uri = FileProvider.getUriForFile(
                                context,
                                "${context.packageName}.fileprovider",
                                lastMidiFile!!
                            )
                            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                                type = "audio/midi"
                                putExtra(Intent.EXTRA_STREAM, uri)
                                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            }
                            context.startActivity(Intent.createChooser(shareIntent, "Поделиться MIDI-файлом"))
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF16A34A))
                    ) {
                        Icon(Icons.Default.Share, contentDescription = null)
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Поделиться / Открыть в секвенсоре")
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
    }
}

@Composable
fun LibraryTab(viewModel: MainViewModel) {
    val tracks by viewModel.tracks.collectAsStateWithLifecycle()
    val context = LocalContext.current

    LaunchedEffect(Unit) {
        viewModel.refreshLibrary()
    }

    if (tracks.isEmpty()) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.padding(32.dp)
            ) {
                Icon(
                    Icons.Default.LibraryMusic,
                    contentDescription = null,
                    modifier = Modifier.size(64.dp),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    "Библиотека пуста",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    "Сгенерируйте свой первый трек на вкладке 'Трек' или создайте 'Кавер'!",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                OutlinedButton(
                    onClick = { viewModel.refreshLibrary() },
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Обновить список файлов")
                }
            }
        }
    } else {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 12.dp, vertical = 8.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 4.dp, vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "Сохраненные треки (${tracks.size})",
                    fontWeight = FontWeight.Bold,
                    fontSize = 15.sp,
                    color = MaterialTheme.colorScheme.onSurface
                )
                IconButton(onClick = { viewModel.refreshLibrary() }) {
                    Icon(Icons.Default.Refresh, contentDescription = "Обновить библиотеку")
                }
            }

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(tracks, key = { it.id }) { track ->
                    TrackCard(
                        track = track,
                        onPlay = { viewModel.playTrack(track) },
                        onShare = {
                            track.audioPath?.let { path ->
                                val file = File(path)
                                if (file.exists()) {
                                    val uri = FileProvider.getUriForFile(
                                        context,
                                        "${context.packageName}.fileprovider",
                                        file
                                    )
                                    val shareIntent = Intent(Intent.ACTION_SEND).apply {
                                        type = "audio/mpeg"
                                        putExtra(Intent.EXTRA_STREAM, uri)
                                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                    }
                                    context.startActivity(Intent.createChooser(shareIntent, "Поделиться треком"))
                                }
                            }
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun TrackCard(
    track: TrackItem,
    onPlay: () -> Unit,
    onShare: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            IconButton(
                onClick = onPlay,
                modifier = Modifier
                    .size(44.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.primary)
            ) {
                Icon(
                    Icons.Default.PlayArrow,
                    contentDescription = "Воспроизвести",
                    tint = MaterialTheme.colorScheme.onPrimary
                )
            }

            Column(modifier = Modifier.weight(1f)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        track.title,
                        fontWeight = FontWeight.Bold,
                        fontSize = 15.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Surface(
                        color = if (track.engine.equals("acestep", ignoreCase = true)) Color(0xFF0284C7) else Color(0xFF7C3AED),
                        shape = RoundedCornerShape(4.dp)
                    ) {
                        Text(
                            if (track.engine.equals("acestep", ignoreCase = true)) "ACE-STEP" else "YuE2",
                            modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp),
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                    }
                    if (track.isCover) {
                        Surface(
                            color = MaterialTheme.colorScheme.secondary,
                            shape = RoundedCornerShape(4.dp)
                        ) {
                            Text(
                                "КАВЕР",
                                modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp),
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
                Text(
                    track.style,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    "${track.durationSeconds} сек • MP3 48kHz",
                    style = MaterialTheme.typography.bodySmall,
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            IconButton(onClick = onShare) {
                Icon(Icons.Default.Share, contentDescription = "Поделиться")
            }
        }
    }
}

@Composable
fun ServerSettingsTab(
    viewModel: MainViewModel,
    serverHealth: ServerHealth
) {
    val activeJob by viewModel.activeJob.collectAsStateWithLifecycle()
    val kaggleStatus by viewModel.kaggleStatus.collectAsStateWithLifecycle()
    val isKaggleStarting by viewModel.isKaggleStarting.collectAsStateWithLifecycle()
    val isKaggleStopping by viewModel.isKaggleStopping.collectAsStateWithLifecycle()
    val kaggleStartStage by viewModel.kaggleStartStage.collectAsStateWithLifecycle()

    var kaggleUser by remember { mutableStateOf(viewModel.kaggleClient.username) }
    var kaggleKey by remember { mutableStateOf(viewModel.kaggleClient.apiKey) }

    val isServerRunning = (serverHealth.isOnline ||
        kaggleStatus?.contains("RUNNING", ignoreCase = true) == true ||
        kaggleStatus?.contains("QUEUED", ignoreCase = true) == true ||
        isKaggleStarting) && !isKaggleStopping

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            "Облачный сервер Kaggle (API)",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold
        )

        // 1. СТАТУС И УПРАВЛЕНИЕ СЕРВЕРОМ (ВКЛЮЧИТЬ / ВЫКЛЮЧИТЬ)
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = when {
                    isKaggleStarting -> Color(0xFF1E3A8A)
                    isServerRunning -> Color(0xFF064E3B)
                    else -> Color(0xFF374151)
                }
            )
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    if (isKaggleStarting) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(24.dp),
                            color = Color(0xFF93C5FD),
                            strokeWidth = 2.5.dp
                        )
                    } else {
                        Icon(
                            imageVector = if (isServerRunning) Icons.Default.CheckCircle else Icons.Default.PowerSettingsNew,
                            contentDescription = null,
                            tint = if (isServerRunning) Color(0xFF34D399) else Color(0xFF9CA3AF),
                            modifier = Modifier.size(24.dp)
                        )
                    }
                    Column {
                        Text(
                            text = when {
                                isKaggleStarting -> "Запуск сервера Kaggle..."
                                isServerRunning -> "Сервер работает (Tesla T4 16GB)"
                                else -> "Сервер Kaggle отключен"
                            },
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp,
                            color = Color.White
                        )
                        Text(
                            text = when {
                                isKaggleStarting -> kaggleStartStage.ifBlank { "Инициализация GPU на Kaggle..." }
                                isServerRunning -> "Статус в облаке: ${kaggleStatus ?: "RUNNING"}"
                                else -> "Квота GPU сохранена (0 часов расхода)"
                            },
                            fontSize = 12.sp,
                            color = Color.White.copy(alpha = 0.8f)
                        )
                    }
                }

                Text(
                    text = when {
                        isKaggleStarting -> "Отправка ядра через Kaggle API ключ. Выделяется нода Tesla T4 GPU."
                        isServerRunning -> "Сервер активен в облаке Kaggle. Генерация треков и каверов отправляется прямо на сервер."
                        else -> "Сервер выключен (автоотключение при 10 мин простое). Включите сервер кнопкой ниже."
                    },
                    color = Color.White.copy(alpha = 0.85f),
                    fontSize = 13.sp
                )

                // КНОПКА ВКЛЮЧЕНИЯ / ВЫКЛЮЧЕНИЯ СЕРВЕРА
                if (isServerRunning || isKaggleStopping) {
                    Button(
                        onClick = { viewModel.stopKaggleServer() },
                        enabled = !isKaggleStopping,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(52.dp),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFDC2626))
                    ) {
                        if (isKaggleStopping) {
                            CircularProgressIndicator(modifier = Modifier.size(20.dp), color = Color.White, strokeWidth = 2.dp)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Остановка сервера в Kaggle...", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                        } else {
                            Icon(Icons.Default.PowerSettingsNew, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            val stopLabel = if (kaggleStatus?.contains("QUEUED", ignoreCase = true) == true || isKaggleStarting) {
                                "🛑 Отменить очередь / запуск (Остановить GPU)"
                            } else {
                                "🛑 Выключить сервер Kaggle (Остановить GPU)"
                            }
                            Text(stopLabel, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                        }
                    }
                } else {
                    Button(
                        onClick = { viewModel.startKaggleServer() },
                        enabled = !isKaggleStarting && !isKaggleStopping,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(52.dp),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF16A34A))
                    ) {
                        Icon(Icons.Default.PowerSettingsNew, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("⚡ Включить сервер на Kaggle (API)", fontWeight = FontWeight.Bold, fontSize = 15.sp)
                    }
                }
            }
        }

        // 2. СТАТУС ОЧЕРЕДИ KAGGLE И ПРОВЕРКА ПОДКЛЮЧЕНИЯ
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "Статус ядра в Kaggle:",
                        fontWeight = FontWeight.Bold,
                        fontSize = 15.sp
                    )
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Text(
                            kaggleStatus ?: "Неизвестно",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.primary
                        )
                        IconButton(
                            onClick = { viewModel.refreshKaggleStatus() },
                            modifier = Modifier.size(24.dp)
                        ) {
                            Icon(Icons.Default.Refresh, contentDescription = "Обновить статус", modifier = Modifier.size(18.dp))
                        }
                    }
                }

                if (activeJob?.status == "running") {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                                Text(
                                    activeJob?.stage ?: "Выполнение задачи на Kaggle GPU...",
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.Medium,
                                    modifier = Modifier.weight(1f)
                                )
                            }
                            Button(
                                onClick = { viewModel.cancelActiveJob() },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(44.dp),
                                shape = RoundedCornerShape(8.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFDC2626))
                            ) {
                                Icon(Icons.Default.Close, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("🛑 Отменить текущую задачу", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                            }
                        }
                    }
                }

                Button(
                    onClick = { viewModel.testKaggleConnection() },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(Icons.Default.CloudSync, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("🔌 Проверить подключение к Kaggle API", fontWeight = FontWeight.SemiBold)
                }

                // КНОПКА МОНИТОРИНГА И ЛОГОВ СЕРВЕРА
                OutlinedButton(
                    onClick = { viewModel.openLogsDialog() },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp),
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.primary)
                ) {
                    Icon(Icons.Default.Terminal, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("📋 Логи и мониторинг сервера Kaggle", fontWeight = FontWeight.SemiBold)
                }
            }
        }

        val showLogs by viewModel.showLogsDialog.collectAsStateWithLifecycle()
        val serverLogs by viewModel.serverLogs.collectAsStateWithLifecycle()
        val isLoadingLogs by viewModel.isLoadingLogs.collectAsStateWithLifecycle()

        if (showLogs) {
            val logsScrollState = rememberScrollState()
            LaunchedEffect(serverLogs) {
                if (serverLogs.isNotBlank()) {
                    logsScrollState.animateScrollTo(logsScrollState.maxValue)
                }
            }

            AlertDialog(
                onDismissRequest = { viewModel.closeLogsDialog() },
                title = {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Мониторинг сервера", fontWeight = FontWeight.Bold, fontSize = 16.sp)
                        IconButton(onClick = { viewModel.fetchKernelLogs() }, enabled = !isLoadingLogs) {
                            if (isLoadingLogs) {
                                CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                            } else {
                                Icon(Icons.Default.Refresh, contentDescription = "Обновить логи")
                            }
                        }
                    }
                },
                text = {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 200.dp, max = 400.dp)
                    ) {
                        Surface(
                            color = Color(0xFF1E1E1E),
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f)
                        ) {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(8.dp)
                                    .verticalScroll(logsScrollState)
                            ) {
                                Text(
                                    text = if (isLoadingLogs && serverLogs.isBlank()) "Загрузка логов..." else serverLogs.ifBlank { "Логи пока отсутствуют" },
                                    color = Color(0xFF4ADE80),
                                    fontSize = 11.sp,
                                    fontFamily = FontFamily.Monospace,
                                    lineHeight = 15.sp
                                )
                            }
                        }
                    }
                },
                confirmButton = {
                    TextButton(onClick = { viewModel.closeLogsDialog() }) {
                        Text("Закрыть")
                    }
                }
            )
        }

        // 3. УЧЕТНЫЕ ДАННЫЕ KAGGLE API
        Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text("Учетные данные Kaggle API:", fontWeight = FontWeight.Bold, fontSize = 14.sp)

                OutlinedTextField(
                    value = kaggleUser,
                    onValueChange = {
                        kaggleUser = it
                        viewModel.kaggleClient.username = it
                    },
                    label = { Text("Username") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )

                OutlinedTextField(
                    value = kaggleKey,
                    onValueChange = {
                        kaggleKey = it
                        viewModel.kaggleClient.apiKey = it
                    },
                    label = { Text("API Key") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )
            }
        }

        // 4. ОПИСАНИЕ АРХИТЕКТУРЫ
        Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Text("Работа через Kaggle API ключ:", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                Text("• Поднятие/отключение сервера: Прямой вызов kernels/push через API ключ", fontSize = 12.sp)
                Text("• Отправка задач: Генерация передается через Kaggle API без сторонних прокси", fontSize = 12.sp)
                Text("• Получение результата: MP3 скачивается напрямую через kernels/output", fontSize = 12.sp)
                Text("• Аппаратная платформа: 2x Tesla T4 GPU (16 GB + 16 GB VRAM)", fontSize = 12.sp)
                Text("• Безопасность: 100% официальный API, без туннелей ngrok/Cloudflare", fontSize = 12.sp)
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
    }
}

private fun formatDuration(millis: Long): String {
    val totalSeconds = (millis / 1000).toInt()
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return String.format("%d:%02d", minutes, seconds)
}
