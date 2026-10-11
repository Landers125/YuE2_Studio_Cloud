package com.example.yue2studio.player

import android.content.Context
import android.media.MediaPlayer
import android.net.Uri
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

data class PlaybackState(
    val isPlaying: Boolean = false,
    val currentTrackTitle: String = "",
    val durationMs: Int = 0,
    val currentPositionMs: Int = 0
)

class AudioPlayerManager(private val context: Context) {
    private var mediaPlayer: MediaPlayer? = null

    private val _playbackState = MutableStateFlow(PlaybackState())
    val playbackState: StateFlow<PlaybackState> = _playbackState.asStateFlow()

    fun playFile(file: File, title: String) {
        stop()
        try {
            mediaPlayer = MediaPlayer().apply {
                setDataSource(context, Uri.fromFile(file))
                prepare()
                start()
                _playbackState.value = PlaybackState(
                    isPlaying = true,
                    currentTrackTitle = title,
                    durationMs = duration,
                    currentPositionMs = 0
                )

                setOnCompletionListener {
                    _playbackState.value = _playbackState.value.copy(
                        isPlaying = false,
                        currentPositionMs = 0
                    )
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
            _playbackState.value = PlaybackState()
        }
    }

    fun pause() {
        mediaPlayer?.let { player ->
            if (player.isPlaying) {
                player.pause()
                _playbackState.value = _playbackState.value.copy(isPlaying = false)
            }
        }
    }

    fun resume() {
        mediaPlayer?.let { player ->
            player.start()
            _playbackState.value = _playbackState.value.copy(isPlaying = true)
        }
    }

    fun togglePlayPause() {
        mediaPlayer?.let { player ->
            if (player.isPlaying) {
                pause()
            } else {
                resume()
            }
        }
    }

    fun seekTo(positionMs: Int) {
        mediaPlayer?.seekTo(positionMs)
        _playbackState.value = _playbackState.value.copy(currentPositionMs = positionMs)
    }

    fun updateProgress() {
        mediaPlayer?.let { player ->
            if (player.isPlaying) {
                _playbackState.value = _playbackState.value.copy(
                    currentPositionMs = player.currentPosition,
                    durationMs = player.duration
                )
            }
        }
    }

    fun stop() {
        mediaPlayer?.let {
            try {
                if (it.isPlaying) {
                    it.stop()
                }
                it.release()
            } catch (e: Exception) {
                // Ignore
            }
        }
        mediaPlayer = null
        _playbackState.value = PlaybackState()
    }
}
