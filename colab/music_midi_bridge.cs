using System;
using System.IO;
using System.Net;
using System.Threading;

class MusicMidiBridge
{
    static string NormalizePath(string path)
    {
        if (string.IsNullOrEmpty(path)) return path;
        if (path.StartsWith(@"\\?\UNC\"))
        {
            return @"\\" + path.Substring(8);
        }
        if (path.StartsWith(@"\\?\"))
        {
            return path.Substring(4);
        }
        return path;
    }

    static int Main(string[] args)
    {
        string audioPath = null;
        string outPath = null;
        string modelDir = null;

        for (int i = 0; i < args.Length; i++)
        {
            if ((args[i] == "--transcribe" || args[i] == "--transcribe-raw") && i + 1 < args.Length)
            {
                audioPath = args[++i];
            }
            else if (args[i] == "--out" && i + 1 < args.Length)
            {
                outPath = args[++i];
            }
            else if (args[i] == "--model" && i + 1 < args.Length)
            {
                modelDir = args[++i];
            }
        }

        audioPath = NormalizePath(audioPath);
        outPath = NormalizePath(outPath);
        modelDir = NormalizePath(modelDir);

        if (string.IsNullOrEmpty(audioPath) || string.IsNullOrEmpty(outPath))
        {
            Console.Error.WriteLine("[music-midi-bridge] Error: --transcribe and --out are required.");
            return 1;
        }

        if (!File.Exists(audioPath))
        {
            Console.Error.WriteLine("[music-midi-bridge] Error: input audio file not found: " + audioPath);
            return 1;
        }

        string modelSize = "large";
        if (!string.IsNullOrEmpty(modelDir))
        {
            if (modelDir.IndexOf("small", StringComparison.OrdinalIgnoreCase) >= 0) modelSize = "small";
            else if (modelDir.IndexOf("medium", StringComparison.OrdinalIgnoreCase) >= 0) modelSize = "medium";
        }

        Console.WriteLine("{\"type\":\"progress\",\"completed\":5,\"total\":100}");

        string bridgeUrl = "http://127.0.0.1:18087/transcribe";
        byte[] audioBytes = File.ReadAllBytes(audioPath);

        HttpWebRequest request = (HttpWebRequest)WebRequest.Create(bridgeUrl);
        request.Method = "POST";
        request.ContentType = "application/octet-stream";
        request.Headers.Add("X-Model-Size", modelSize);
        request.Headers.Add("X-Audio-Filename", Path.GetFileName(audioPath));
        request.Timeout = 600000; // 10 minutes max
        request.ContentLength = audioBytes.Length;

        // Progress simulator thread while waiting for remote Colab GPU
        bool finished = false;
        int simulatedProgress = 10;
        Thread progressThread = new Thread(() =>
        {
            while (!finished && simulatedProgress < 95)
            {
                Thread.Sleep(2000);
                if (finished) break;
                simulatedProgress += 5;
                Console.WriteLine("{\"type\":\"progress\",\"completed\":" + simulatedProgress + ",\"total\":100}");
            }
        });
        progressThread.IsBackground = true;
        progressThread.Start();

        try
        {
            using (Stream reqStream = request.GetRequestStream())
            {
                reqStream.Write(audioBytes, 0, audioBytes.Length);
            }

            using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
            {
                if (response.StatusCode != HttpStatusCode.OK)
                {
                    Console.Error.WriteLine("[music-midi-bridge] Remote Colab error: HTTP " + (int)response.StatusCode);
                    finished = true;
                    return 1;
                }

                using (MemoryStream ms = new MemoryStream())
                {
                    using (Stream respStream = response.GetResponseStream())
                    {
                        respStream.CopyTo(ms);
                    }
                    byte[] midiBytes = ms.ToArray();
                    if (midiBytes.Length == 0)
                    {
                        Console.Error.WriteLine("[music-midi-bridge] Received empty MIDI response.");
                        finished = true;
                        return 1;
                    }

                    string dir = Path.GetDirectoryName(outPath);
                    if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                    {
                        Directory.CreateDirectory(dir);
                    }

                    File.WriteAllBytes(outPath, midiBytes);
                }
            }
        }
        catch (WebException wex)
        {
            finished = true;
            string extra = "";
            if (wex.Response != null)
            {
                try
                {
                    using (StreamReader reader = new StreamReader(wex.Response.GetResponseStream()))
                    {
                        extra = " -> " + reader.ReadToEnd();
                    }
                }
                catch { }
            }
            Console.Error.WriteLine("[music-midi-bridge] Colab transcription error: " + wex.Message + extra);
            Console.Error.Flush();
            Console.Out.Flush();
            return 1;
        }
        catch (Exception ex)
        {
            finished = true;
            Console.Error.WriteLine("[music-midi-bridge] Error: " + ex.Message);
            Console.Error.Flush();
            Console.Out.Flush();
            return 1;
        }

        finished = true;
        Console.WriteLine("{\"type\":\"progress\",\"completed\":100,\"total\":100}");
        Console.WriteLine("{\"type\":\"done\"}");
        Console.Out.Flush();
        return 0;
    }
}
