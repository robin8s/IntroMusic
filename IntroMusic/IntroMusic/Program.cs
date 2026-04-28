using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Media;
using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Structure;

class FaceMusicPlayer
{
    private static VideoCapture _capture;
    private static LBPHFaceRecognizer _recognizer;
    private static List<string> _names = new List<string>();
    private static Dictionary<string, string> _musicDb = new Dictionary<string, string>();
    private static string _lastPlayed = "";

    static void Main()
    {
        // 1. Setup Database (Link names to music)
        _musicDb.Add("Alice", "alice_theme.wav");
        _musicDb.Add("Bob", "bob_song.wav");

        // 2. Train the Model
        TrainModel();

        // 3. Setup Webcam
        _capture = new VideoCapture(0); // '0' is default webcam
        _capture.ImageGrabbed += ProcessFrame;
        _capture.Start();

        Console.WriteLine("System Active. Press Enter to stop.");
        Console.ReadLine();
    }

    static void TrainModel()
    {
        _recognizer = new LBPHFaceRecognizer();
        List<Image<Gray, byte>> trainingImages = new List<Image<Gray, byte>>();
        List<int> labels = new List<int>();

        // Load images from a folder named 'TrainedFaces'
        // Files should be named like: 0_Alice.jpg, 1_Bob.jpg
        string[] files = Directory.GetFiles("./TrainedFaces", "*.jpg");

        foreach (var file in files)
        {
            var name = Path.GetFileNameWithoutExtension(file).Split('_')[1];
            if (!_names.Contains(name)) _names.Add(name);

            trainingImages.Add(new Image<Gray, byte>(file).Resize(200, 200, Emgu.CV.CvEnum.Inter.Cubic));
            labels.Add(_names.IndexOf(name));
        }

        _recognizer.Train(trainingImages.ToArray(), labels.ToArray());
    }

    static void ProcessFrame(object sender, EventArgs e)
    {
        Mat frame = new Mat();
        _capture.Retrieve(frame);
        var grayFrame = frame.ToImage<Gray, byte>();

        // Detect face location
        CascadeClassifier classifier = new CascadeClassifier("haarcascade_frontalface_default.xml");
        Rectangle[] faces = classifier.DetectMultiScale(grayFrame, 1.1, 10);

        foreach (var face in faces)
        {
            var result = _recognizer.Predict(grayFrame.Copy(face).Resize(200, 200, Emgu.CV.CvEnum.Inter.Cubic));

            // result.Label is the ID, result.Distance is how confident it is (lower is better)
            if (result.Label != -1 && result.Distance < 60)
            {
                string personName = _names[result.Label];

                if (_lastPlayed != personName)
                {
                    Console.WriteLine($"Matched: {personName}");
                    PlayMusic(_musicDb[personName]);
                    _lastPlayed = personName;
                }
            }
        }
    }

    static void PlayMusic(string path)
    {
        SoundPlayer player = new SoundPlayer(path);
        player.Play();
    }
}