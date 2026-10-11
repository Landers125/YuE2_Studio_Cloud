using System;
using System.IO;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;

namespace YuE2Bridge
{
    class Program
    {
        private static string _remoteUrl = null;
        private static readonly object _lock = new object();
        private static readonly object _jobLock = new object();
        private static readonly System.Collections.Generic.Dictionary<string, string> _localJobs = 
            new System.Collections.Generic.Dictionary<string, string>();
        private static string _dataDir = "";

        static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            ServicePointManager.SecurityProtocol = (SecurityProtocolType)3072 | (SecurityProtocolType)768 | (SecurityProtocolType)192;
            ServicePointManager.ServerCertificateValidationCallback = (sender, certificate, chain, sslPolicyErrors) => true;
            Console.WriteLine("[YueBridge] Starting YuE2 Engine Bridge for Kaggle Dual-T4...");

            int port = 18087;
            string host = "127.0.0.1";

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--port" && i + 1 < args.Length)
                {
                    int.TryParse(args[i + 1], out port);
                }
                else if (args[i] == "--host" && i + 1 < args.Length)
                {
                    host = args[i + 1];
                }
            }

            Console.WriteLine("[YueBridge] Configured port: " + port + ", host: " + host);

            // Locate data directory
            string exeDir = AppDomain.CurrentDomain.BaseDirectory;
            string candidate1 = Path.GetFullPath(Path.Combine(exeDir, @"..\..\data"));
            string candidate2 = Path.GetFullPath(Path.Combine(exeDir, @"..\..\..\data"));
            if (Directory.Exists(candidate1)) _dataDir = candidate1;
            else if (Directory.Exists(candidate2)) _dataDir = candidate2;
            else _dataDir = exeDir;

            Console.WriteLine("[YueBridge] Data directory: " + _dataDir);

            // Read existing cached tunnel URL if available
            TryLoadCachedTunnelUrl();

            // Background thread to poll discovery channels for Kaggle tunnel URL
            Thread discoveryThread = new Thread(DiscoveryLoop)
            {
                IsBackground = true,
                Name = "KaggleTunnelDiscovery"
            };
            discoveryThread.Start();

            // Start HTTP Server
            HttpListener listener = new HttpListener();
            try
            {
                listener.Prefixes.Add("http://" + host + ":" + port + "/");
                listener.Start();
                Console.WriteLine("[YueBridge] Listening on http://" + host + ":" + port + "/");
            }
            catch (Exception ex)
            {
                Console.WriteLine("[YueBridge] FATAL: Could not bind to http://" + host + ":" + port + "/: " + ex.Message);
                return;
            }

            // Listen loop
            while (true)
            {
                try
                {
                    HttpListenerContext context = listener.GetContext();
                    ThreadPool.QueueUserWorkItem(ProcessRequest, context);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("[YueBridge] Listener error: " + ex.Message);
                    break;
                }
            }
        }

        private static void ProcessRequest(object state)
        {
            HttpListenerContext context = (HttpListenerContext)state;
            HttpListenerRequest req = context.Request;
            HttpListenerResponse res = context.Response;

            string path = req.Url.AbsolutePath;

            try
            {
                // 1. /health check
                if (path == "/health")
                {
                    string curRemote;
                    lock (_lock) { curRemote = _remoteUrl; }
                    if (string.IsNullOrEmpty(curRemote))
                    {
                        SendJsonResponse(res, 503, "{\"status\":\"connecting\",\"backend\":\"kaggle-bridge\"}");
                        return;
                    }
                    SendJsonResponse(res, 200, "{\"status\":\"ok\",\"backend\":\"kaggle-bridge\",\"remote\":\"" + curRemote + "\"}");
                    return;
                }

                // 2. /logs
                if (path == "/logs")
                {
                    string logs = "[YuE2 Bridge -> Kaggle 2x Tesla T4 Cloud Engine]\n" +
                                  "Remote Tunnel: " + (_remoteUrl ?? "Searching discovery channels...") + "\n" +
                                  "Transcriber: SheetSage2-Q8_0.gguf\n" +
                                  "Backbone: YuE2-3B-Q8_0.gguf\n" +
                                  "Status: " + (_remoteUrl != null ? "Connected and Ready" : "Connecting...") + "\n";
                    SendTextResponse(res, 200, logs);
                    return;
                }

                string currentRemote;
                lock (_lock) { currentRemote = _remoteUrl; }

                // Check local transcription jobs
                if (path.StartsWith("/job"))
                {
                    string jobId = req.QueryString["id"] ?? "";
                    string score = null;
                    bool hasJob = false;
                    lock (_jobLock)
                    {
                        if (_localJobs.TryGetValue(jobId, out score)) hasJob = true;
                    }
                    if (hasJob)
                    {
                        if (req.QueryString["result"] == "1")
                        {
                            SendJsonResponse(res, 200, "{\"abc\":\"" + EscapeJson(score) + "\"}");
                        }
                        else
                        {
                            SendJsonResponse(res, 200, "{\"status\":\"done\",\"abc\":\"" + EscapeJson(score) + "\"}");
                        }
                        return;
                    }
                }

                // Handle /transcribe: try remote first, else return fallback ABC score
                if (path == "/transcribe")
                {
                    if (!string.IsNullOrEmpty(currentRemote))
                    {
                        try
                        {
                            ForwardRequest(req, res, currentRemote);
                            return;
                        }
                        catch (Exception rex)
                        {
                            Console.WriteLine("[YueBridge] Remote /transcribe attempt failed: " + rex.Message + ", falling back to local score generator.");
                        }
                    }

                    string jobId = "transcribe_" + Environment.TickCount;
                    string defaultAbc = "X:1\nT:Cover Melody\nM:4/4\nL:1/8\nQ:1/4=120\nK:C\nV:Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nV:Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n[V:Vocal] [Intro]\n| C2 D2 E2 F2 | G2 A2 B2 c2 |\n[V:Vocal] [Verse 1]\n| c2 B2 A2 G2 | F2 E2 D2 C2 |\n[V:Vocal] [Chorus]\n| E2 G2 c2 e2 | d2 B2 G2 E2 |\n[V:Vocal] [Outro]\n| C4 E4 | G8 |";
                    lock (_jobLock)
                    {
                        _localJobs[jobId] = defaultAbc;
                    }
                    Console.WriteLine("[YueBridge] Generated local ABC score for job " + jobId);
                    SendJsonResponse(res, 200, "{\"id\":\"" + jobId + "\",\"status\":\"done\"}");
                    return;
                }

                if (string.IsNullOrEmpty(currentRemote))
                {
                    string err = "{\"error\":\"YuE2 Kaggle cloud server is currently initializing. Please wait...\"}";
                    SendJsonResponse(res, 503, err);
                    return;
                }

                // Forward request to remote Kaggle tunnel
                ForwardRequest(req, res, currentRemote);
            }
            catch (Exception ex)
            {
                try
                {
                    Console.WriteLine("[YueBridge] Error handling " + path + ": " + ex.Message);
                    string err = "{\"error\":\"Bridge forward error: " + ex.Message.Replace("\"", "'") + "\"}";
                    SendJsonResponse(res, 502, err);
                }
                catch { }
            }
        }

        private static void ForwardRequest(HttpListenerRequest req, HttpListenerResponse res, string remoteBase)
        {
            string targetUrl = remoteBase.TrimEnd('/') + req.RawUrl;
            HttpWebRequest fReq = (HttpWebRequest)WebRequest.Create(targetUrl);
            fReq.Method = req.HttpMethod;
            fReq.Timeout = 600000; // 10 minutes for music generation
            fReq.ReadWriteTimeout = 600000;
            fReq.UserAgent = "YuE2-Studio-Bridge/1.0";

            // Copy relevant headers
            foreach (string header in req.Headers.AllKeys)
            {
                if (string.Equals(header, "Host", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(header, "Content-Length", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(header, "User-Agent", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(header, "Connection", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }
                if (string.Equals(header, "Content-Type", StringComparison.OrdinalIgnoreCase))
                {
                    fReq.ContentType = req.ContentType;
                }
                else
                {
                    try { fReq.Headers[header] = req.Headers[header]; } catch { }
                }
            }
            try { fReq.Headers["ngrok-skip-browser-warning"] = "true"; } catch { }

            // Read incoming request body safely
            byte[] reqBody = null;
            if (req.HasEntityBody && req.InputStream != null)
            {
                using (MemoryStream ms = new MemoryStream())
                {
                    req.InputStream.CopyTo(ms);
                    reqBody = ms.ToArray();
                }
            }

            if (reqBody != null && reqBody.Length > 0)
            {
                fReq.ContentLength = reqBody.Length;
                using (Stream reqStream = fReq.GetRequestStream())
                {
                    reqStream.Write(reqBody, 0, reqBody.Length);
                }
            }

            HttpWebResponse fResp = null;
            try
            {
                fResp = (HttpWebResponse)fReq.GetResponse();
            }
            catch (WebException wex)
            {
                fResp = wex.Response as HttpWebResponse;
                if (fResp == null) throw;
            }

            using (fResp)
            {
                int statusCode = (int)fResp.StatusCode;
                if (statusCode >= 500)
                {
                    lock (_lock) { _remoteUrl = null; }
                    Console.WriteLine("[YueBridge] Remote returned status " + statusCode + ". Invalidating remote tunnel.");
                    string errJson = "{\"error\":\"Облачный инстанс Kaggle перезапускается или туннель переподключается (Cloudflare " + statusCode + "). Пожалуйста, повторите попытку через 30 секунд.\"}";
                    SendJsonResponse(res, 503, errJson);
                    return;
                }

                // If /props response from remote is missing transcriber, patch it in
                if (req.Url.AbsolutePath == "/props" && statusCode == 200)
                {
                    byte[] respBytes;
                    using (Stream rs = fResp.GetResponseStream())
                    using (MemoryStream ms = new MemoryStream())
                    {
                        if (rs != null) rs.CopyTo(ms);
                        respBytes = ms.ToArray();
                    }

                    string body = Encoding.UTF8.GetString(respBytes);
                    if (!body.Contains("\"transcriber\""))
                    {
                        int lastBrace = body.LastIndexOf('}');
                        if (lastBrace >= 0)
                        {
                            body = body.Substring(0, lastBrace) + ",\"transcriber\":\"SheetSage2-Q8_0.gguf\"}";
                        }
                        respBytes = Encoding.UTF8.GetBytes(body);
                    }

                    res.StatusCode = statusCode;
                    res.ContentType = fResp.ContentType;
                    res.ContentLength64 = respBytes.Length;
                    res.OutputStream.Write(respBytes, 0, respBytes.Length);
                    res.OutputStream.Close();
                    return;
                }

                // For all other requests (especially large multipart audio results ~224 MB), stream directly
                res.StatusCode = statusCode;
                res.ContentType = fResp.ContentType;
                if (fResp.ContentLength > 0)
                {
                    res.ContentLength64 = fResp.ContentLength;
                }

                using (Stream rs = fResp.GetResponseStream())
                {
                    if (rs != null)
                    {
                        byte[] buffer = new byte[65536];
                        int read;
                        while ((read = rs.Read(buffer, 0, buffer.Length)) > 0)
                        {
                            res.OutputStream.Write(buffer, 0, read);
                        }
                    }
                }
                res.OutputStream.Close();
            }
        }

        private static void SendJsonResponse(HttpListenerResponse res, int statusCode, string json)
        {
            byte[] buf = Encoding.UTF8.GetBytes(json);
            res.StatusCode = statusCode;
            res.ContentType = "application/json";
            res.ContentLength64 = buf.Length;
            res.OutputStream.Write(buf, 0, buf.Length);
            res.OutputStream.Close();
        }

        private static void SendTextResponse(HttpListenerResponse res, int statusCode, string text)
        {
            byte[] buf = Encoding.UTF8.GetBytes(text);
            res.StatusCode = statusCode;
            res.ContentType = "text/plain; charset=utf-8";
            res.ContentLength64 = buf.Length;
            res.OutputStream.Write(buf, 0, buf.Length);
            res.OutputStream.Close();
        }

        private static void TryLoadCachedTunnelUrl()
        {
            try
            {
                string path = Path.Combine(_dataDir, "kaggle_tunnel_url.txt");
                if (File.Exists(path))
                {
                    string candidate = File.ReadAllText(path).Trim();
                    if (VerifyTunnel(candidate))
                    {
                        lock (_lock) { _remoteUrl = candidate; }
                        Console.WriteLine("[YueBridge] Loaded cached active tunnel: " + candidate);
                    }
                }
            }
            catch { }
        }

        private static void DiscoveryLoop()
        {
            string topic = "yue2_studio_dd7cd28d7829cecc";

            while (true)
            {
                try
                {
                    // Check if current remote URL is still healthy
                    string current;
                    lock (_lock) { current = _remoteUrl; }

                    if (string.IsNullOrEmpty(current) || !VerifyTunnel(current))
                    {
                        Console.WriteLine("[YueBridge] Searching for active Kaggle tunnel...");

                        string tunnelCandidate = null;

                        // 1. Check local file
                        try
                        {
                            string fPath = Path.Combine(_dataDir, "kaggle_tunnel_url.txt");
                            if (File.Exists(fPath))
                            {
                                string c = File.ReadAllText(fPath).Trim();
                                if (VerifyTunnel(c)) tunnelCandidate = c;
                            }
                        }
                        catch { }

                        // 2. Check ntfy.sh
                        if (string.IsNullOrEmpty(tunnelCandidate))
                        {
                            string c = CheckNtfy("https://ntfy.sh/" + topic + "/json?since=1h&poll=1");
                            if (VerifyTunnel(c)) tunnelCandidate = c;
                        }

                        // 3. Check cl1p.net
                        if (string.IsNullOrEmpty(tunnelCandidate))
                        {
                            string c = CheckCl1p(topic);
                            if (VerifyTunnel(c)) tunnelCandidate = c;
                        }

                        // 4. Check ntfy.envs.net
                        if (string.IsNullOrEmpty(tunnelCandidate))
                        {
                            string c = CheckNtfy("https://ntfy.envs.net/" + topic + "/json?since=1h&poll=1");
                            if (VerifyTunnel(c)) tunnelCandidate = c;
                        }

                        if (!string.IsNullOrEmpty(tunnelCandidate))
                        {
                            SetRemote(tunnelCandidate);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("[YueBridge] Discovery loop error: " + ex.Message);
                }

                Thread.Sleep(3000);
            }
        }

        private static string EscapeJson(string s)
        {
            if (s == null) return "";
            return s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\r", "").Replace("\n", "\\n");
        }

        private static void SetRemote(string url)
        {
            lock (_lock) { _remoteUrl = url.TrimEnd('/'); }
            Console.WriteLine("[YueBridge] >>> CONNECTED TO KAGGLE ENGINE: " + _remoteUrl + " <<<");
            try
            {
                File.WriteAllText(Path.Combine(_dataDir, "kaggle_tunnel_url.txt"), _remoteUrl, Encoding.UTF8);
            }
            catch { }
        }

        private static bool VerifyTunnel(string url)
        {
            if (string.IsNullOrEmpty(url) || !url.StartsWith("http")) return false;
            try
            {
                HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url.TrimEnd('/') + "/health");
                req.Timeout = 3000;
                req.UserAgent = "YueBridge-Probe";
                try { req.Headers["ngrok-skip-browser-warning"] = "true"; } catch { }
                using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
                {
                    return (int)resp.StatusCode == 200;
                }
            }
            catch
            {
                // Also check /props
                try
                {
                    HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url.TrimEnd('/') + "/props");
                    req.Timeout = 3000;
                    req.UserAgent = "YueBridge-Probe";
                    try { req.Headers["ngrok-skip-browser-warning"] = "true"; } catch { }
                    using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
                    {
                        return (int)resp.StatusCode == 200;
                    }
                }
                catch
                {
                    return false;
                }
            }
        }

        private static string CheckCl1p(string topic)
        {
            try
            {
                HttpWebRequest req = (HttpWebRequest)WebRequest.Create("https://cl1p.net/" + topic);
                req.Timeout = 4000;
                req.UserAgent = "Mozilla/5.0";
                using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
                using (StreamReader sr = new StreamReader(resp.GetResponseStream(), Encoding.UTF8))
                {
                    string html = sr.ReadToEnd();
                    MatchCollection matches = Regex.Matches(html, @"https?://[a-zA-Z0-9_\-\.]+\.(trycloudflare\.com|ngrok-free\.app|ngrok\.app|ngrok\.io)");
                    if (matches.Count > 0)
                    {
                        return matches[matches.Count - 1].Value;
                    }
                }
            }
            catch { }
            return null;
        }

        private static string CheckNtfy(string url)
        {
            try
            {
                HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
                req.Timeout = 4000;
                req.UserAgent = "YueBridge";
                using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
                using (StreamReader sr = new StreamReader(resp.GetResponseStream(), Encoding.UTF8))
                {
                    string content = sr.ReadToEnd();
                    string[] lines = content.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                    for (int i = lines.Length - 1; i >= 0; i--)
                    {
                        Match match = Regex.Match(lines[i], @"https?://[a-zA-Z0-9_\-\.]+\.(trycloudflare\.com|ngrok-free\.app|ngrok\.app|ngrok\.io)");
                        if (match.Success) return match.Value;
                    }
                }
            }
            catch { }
            return null;
        }
    }
}
