using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using NAudio.Wave;

class Program
{
    // ------------------------------------------------------------
    // SETTINGS
    // ------------------------------------------------------------

    // Your training images are already 224x224.
    const int FaceSize = 224;

    // Lower number = stricter music trigger.
    // The program will still print its best guess even if the distance is above this.
    const double MatchThreshold = 65;

    // Same person must be detected this many frames in a row before music triggers.
    const int StableFrameRequirement = 5;

    static VideoCapture _capture;
    static LBPHFaceRecognizer _recognizer;
    static CascadeClassifier _faceDetector;
    static List<string> _labelNames = new List<string>();

    static IWavePlayer _audioOutput;
    static AudioFileReader _audioFile;
    static string _currentlyPlaying = "";

    static DateTime _lastPlayTime = DateTime.MinValue;

    // Used to prevent one random bad frame from triggering music.
    static string _lastDetectedName = "";
    static int _samePersonCount = 0;

    static void Main()
    {
        // 1. Find the Face Detector XML
        string xmlPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "haarcascade_frontalface_default.xml");

        if (!File.Exists(xmlPath))
        {
            Console.WriteLine("ERROR: haarcascade_frontalface_default.xml not found in output folder!");
            return;
        }

        _faceDetector = new CascadeClassifier(xmlPath);

        // 2. Train the recognizer based on the 'People' folder
        if (!TrainRecognizer())
        {
            return;
        }

        // 3. Start the Webcam
        _capture = new VideoCapture(0);
        _capture.ImageGrabbed += (s, e) => ProcessFrame();
        _capture.Start();

        Console.WriteLine("System Active. Press Enter to quit.");
        Console.ReadLine();

        // Clean up camera/audio when closing
        _capture?.Stop();
        _capture?.Dispose();
        _audioOutput?.Stop();
        _audioOutput?.Dispose();
        _audioFile?.Dispose();
    }

    static bool TrainRecognizer()
    {
        string peoplePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "People");

        if (!Directory.Exists(peoplePath))
        {
            Directory.CreateDirectory(peoplePath);
        }

        var faceImages = new List<Mat>();
        var faceLabels = new List<int>();
        _labelNames.Clear();

        string[] folders = Directory.GetDirectories(peoplePath);
        int id = 0;

        foreach (var folder in folders)
        {
            string personName = Path.GetFileName(folder);
            _labelNames.Add(personName);

            foreach (var file in Directory.GetFiles(folder, "*.png"))
            {
                // Load your already-cropped 224x224 training image.
                var img = new Image<Gray, byte>(file);

                // Since you said all training images are already 224x224,
                // this checks for mistakes instead of silently resizing.
                if (img.Width != FaceSize || img.Height != FaceSize)
                {
                    Console.WriteLine($"Skipping {file}");
                    Console.WriteLine($"Reason: image is {img.Width}x{img.Height}, not {FaceSize}x{FaceSize}");
                    continue;
                }

                faceImages.Add(img.Mat);
                faceLabels.Add(id);
            }

            id++;
        }

        if (faceImages.Count == 0)
        {
            Console.WriteLine("No valid training images found!");
            Console.WriteLine($"Put {FaceSize}x{FaceSize} .png photos in People/[Name] folders.");
            return false;
        }

        // Use double.MaxValue so LBPH always returns its best guess.
        // We manually use MatchThreshold later only for triggering music.
        _recognizer = new LBPHFaceRecognizer(1, 8, 8, 8, double.MaxValue);

        _recognizer.Train(
            new VectorOfMat(faceImages.ToArray()),
            new VectorOfInt(faceLabels.ToArray())
        );

        Console.WriteLine($"Trained on {faceImages.Count} images for {_labelNames.Count} people.");
        Console.WriteLine($"Face size: {FaceSize}x{FaceSize}");
        Console.WriteLine($"Music trigger threshold: {MatchThreshold}");

        return true;
    }

    static void ProcessFrame()
    {
        Mat frame = new Mat();
        _capture.Retrieve(frame);

        if (frame.IsEmpty)
        {
            return;
        }

        var gray = frame.ToImage<Gray, byte>();

        // Detect faces in the webcam frame.
        var faces = _faceDetector.DetectMultiScale(gray, 1.1, 10);

        if (faces.Length == 0)
        {
            _currentlyPlaying = "";
            _lastDetectedName = "";
            _samePersonCount = 0;
            return;
        }

        foreach (var rect in faces)
        {
            // Crop the detected face from the webcam.
            // Then resize the webcam crop to 224x224 so it matches your training data.
            // Add padding around the detected face.
            // Use this if your training images include some hair, ears, chin, or forehead.
            int padding = 20;

            int x = Math.Max(rect.X - padding, 0);
            int y = Math.Max(rect.Y - padding, 0);
            int width = Math.Min(rect.Width + padding * 2, gray.Width - x);
            int height = Math.Min(rect.Height + padding * 2, gray.Height - y);

            Rectangle paddedRect = new Rectangle(x, y, width, height);

            // Crop the padded face area, then resize to 224x224 to match training data.
            var faceImg = gray.Copy(paddedRect).Resize(FaceSize, FaceSize, Emgu.CV.CvEnum.Inter.Cubic);

            var result = _recognizer.Predict(faceImg);

            // This should almost always have a label now because the recognizer threshold is double.MaxValue.
            if (result.Label == -1)
            {
                Console.WriteLine($"Detected: No guess | Distance: {result.Distance:0.0}");

                _lastDetectedName = "";
                _samePersonCount = 0;

                continue;
            }

            string name = _labelNames[result.Label];

            // Always output the best guess, like your original version.
            Console.WriteLine($"Detected: {name} | Distance: {result.Distance:0.0}");

            // ------------------------------------------------------------
            // STABLE FRAME CHECK
            // ------------------------------------------------------------
            // This makes the program wait until the same person is detected
            // several frames in a row before playing music.
            if (name == _lastDetectedName)
            {
                _samePersonCount++;
            }
            else
            {
                _lastDetectedName = name;
                _samePersonCount = 1;
            }

            bool stableMatch = _samePersonCount >= StableFrameRequirement;

            // ------------------------------------------------------------
            // MUSIC TRIGGER LOGIC
            // ------------------------------------------------------------
            bool isConfident = result.Distance < MatchThreshold;
            bool isNewPerson = _currentlyPlaying != name;
            bool cooldownFinished = (DateTime.Now - _lastPlayTime).TotalSeconds >= 10;

            if (isConfident && stableMatch && isNewPerson)
            {
                if (cooldownFinished)
                {
                    Console.WriteLine($">>> [TRIGGER]: Stable confident match found. Playing {name}'s music!");

                    PlayMusic(name);

                    _currentlyPlaying = name;
                    _lastPlayTime = DateTime.Now;
                }
                else
                {
                    double secondsLeft = 10 - (DateTime.Now - _lastPlayTime).TotalSeconds;
                    Console.WriteLine($"... Match found, but waiting on cooldown ({secondsLeft:0}s left) ...");
                }
            }
            else if (!stableMatch)
            {
                Console.WriteLine($"... Waiting for stable match: {_samePersonCount}/{StableFrameRequirement} ...");
            }
            else if (!isConfident)
            {
                Console.WriteLine($"... Best guess was {name}, but distance {result.Distance:0.0} is above trigger threshold {MatchThreshold} ...");
            }
        }
    }

    static void PlayMusic(string name)
    {
        try
        {
            string musicPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Music", $"{name}.mp3");

            if (!File.Exists(musicPath))
            {
                Console.WriteLine($"Music file not found for {name}");
                return;
            }

            _audioOutput?.Stop();
            _audioOutput?.Dispose();
            _audioFile?.Dispose();

            _audioOutput = new WaveOutEvent();
            _audioFile = new AudioFileReader(musicPath);

            _audioOutput.Init(_audioFile);
            _audioOutput.Play();
        }
        catch (Exception ex)
        {
            Console.WriteLine("Audio error: " + ex.Message);
        }
    }
}