using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using NAudio.Wave;

class Program
{
    
    static VideoCapture _capture;
    static LBPHFaceRecognizer _recognizer;
    static CascadeClassifier _faceDetector;
    static List<string> _labelNames = new List<string>();

    static IWavePlayer _audioOutput;
    static AudioFileReader _audioFile;
    static string _currentlyPlaying = "";

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
        if (!TrainRecognizer()) return;

        // 3. Start the Webcam
        _capture = new VideoCapture(0);
        _capture.ImageGrabbed += (s, e) => ProcessFrame();
        _capture.Start();

        Console.WriteLine("System Active. Press Enter to quit.");
        Console.ReadLine();
    }

    static bool TrainRecognizer()
    {
        string peoplePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "People");
        if (!Directory.Exists(peoplePath)) Directory.CreateDirectory(peoplePath);

        var faceImages = new List<Mat>();
        var faceLabels = new List<int>();
        _labelNames.Clear();

        string[] folders = Directory.GetDirectories(peoplePath);
        int id = 0;

        foreach (var folder in folders)
        {
            string personName = Path.GetFileName(folder);
            _labelNames.Add(personName);

            foreach (var file in Directory.GetFiles(folder, "*.jpg"))
            {
                // Load image, convert to Grayscale, and resize to 200x200
                var img = new Image<Gray, byte>(file).Resize(200, 200, Emgu.CV.CvEnum.Inter.Cubic);
                faceImages.Add(img.Mat);
                faceLabels.Add(id);
            }
            id++;
        }

        if (faceImages.Count == 0)
        {
            Console.WriteLine("No training images found! Put .jpg photos in People/[Name] folders.");
            return false;
        }

        _recognizer = new LBPHFaceRecognizer();
        _recognizer.Train(new VectorOfMat(faceImages.ToArray()), new VectorOfInt(faceLabels.ToArray()));

        Console.WriteLine($"Trained on {faceImages.Count} images for {_labelNames.Count} people.");
        return true;
    }

    static DateTime _lastPlayTime = DateTime.MinValue;

    static void ProcessFrame()
    {
        Mat frame = new Mat();
        _capture.Retrieve(frame);
        if (frame.IsEmpty) return;

        var gray = frame.ToImage<Gray, byte>();
        var faces = _faceDetector.DetectMultiScale(gray, 1.1, 10);

        if (faces.Length == 0)
        {
            _currentlyPlaying = "";
            return;
        }

        foreach (var rect in faces)
        {
            var faceImg = gray.Copy(rect).Resize(200, 200, Emgu.CV.CvEnum.Inter.Cubic);
            var result = _recognizer.Predict(faceImg);

            if (result.Label != -1)
            {
                string name = _labelNames[result.Label];

                // --- ALWAYS DISPLAY CONFIDENCE ---
                // This stays outside the timer so you can see the numbers constantly
                Console.WriteLine($"Detected: {name} | Confidence: {result.Distance:0.0}");

                // --- MUSIC TRIGGER LOGIC WITH TIMER ---
                bool isConfident = result.Distance < 85; // Adjust this number if needed
                bool isNewPerson = _currentlyPlaying != name;
                bool cooldownFinished = (DateTime.Now - _lastPlayTime).TotalSeconds >= 10;

                if (isConfident && isNewPerson)
                {
                    if (cooldownFinished)
                    {
                        Console.WriteLine($">>> [TRIGGER]: Cooldown over. Playing {name}'s music!");
                        PlayMusic(name);
                        _currentlyPlaying = name;
                        _lastPlayTime = DateTime.Now; // Start the 10-second timer
                    }
                    else
                    {
                        // Optional: Tells you the system recognizes you but is waiting on the timer
                        double secondsLeft = 10 - (DateTime.Now - _lastPlayTime).TotalSeconds;
                        Console.WriteLine($"... Match found, but waiting on cooldown ({secondsLeft:0}s left) ...");
                    }
                }
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
        catch (Exception ex) { Console.WriteLine("Audio error: " + ex.Message); }
    }
}