using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Navigation;
using Microsoft.UI.Xaml.Shapes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.Storage.Pickers;

using NReco.VideoConverter;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Windows.Storage.Streams;
using Windows.AI.MachineLearning;
using Path = System.IO.Path;
using System.Reflection;
using Windows.Storage;
using System.Threading.Tasks;


// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace SubtitleGenerator
{
    /// <summary>
    /// An empty window that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainWindow : Window
    {

        public List<string> Languages = new List<string>(Utils.languageCodes.Keys);
        private string VideoFilePath { get; set; }
        public enum TaskType
        {
            Translate = 50358,
            Transcribe = 50359
        }

        public MainWindow()
        {
            InitializeComponent();
            Title = "Subtitles Generator";
            AppWindow.MoveAndResize(new Windows.Graphics.RectInt32(100, 100, 1100, 1100));
        }

        private void Combo2_Loaded(object sender, RoutedEventArgs e)
        {
            Combo2.SelectedIndex = 2;
        }
        private void BatchSeconds_Loaded(object sender, RoutedEventArgs e)
        {
            BatchSeconds.Value = 30;
        }

        private async void GenerateSubtitles_ButtonClick(object sender, RoutedEventArgs e)
        {
            var audioBytes = Utils.LoadAudioBytes(VideoFilePath);
            var srtBatches = new List<string>();

            var dynamicChunks = DetectVoice(audioBytes);

            foreach (var chunk in dynamicChunks.Select((value, i) => (value, i)))
            {
                // Assuming you have or can create a method for extracting audio segments by start/end times.
                // This will involve modifying your Utils.ExtractAudioFromVideo method or creating a new one that can handle this.
                // The new method might return a byte array of the audio for the specified chunk.
                // The method might look like: Utils.ExtractAudioSegment(audioBytes, chunk.start, chunk.end);
                var audioSegment = Utils.ExtractAudioSegment(VideoFilePath, chunk.value.start, chunk.value.end);

                // Assuming TranscribeAsync can take the audio segment directly along with other parameters.
                // The provided chunk might also be used to adjust how you number/name the subtitle batches.
                // Adjusting the call to TranscribeAsync to use the chunk's start time or a combination of start and end times for naming uniqueness.
                var transcription = await TranscribeAsync(audioSegment, Combo2.SelectedValue.ToString(), Switch1.IsOn ? TaskType.Translate : TaskType.Transcribe, (int)chunk.value.start);

                srtBatches.Add(transcription);
            }

            //var audioData = Utils.ExtractAudioFromVideo(VideoFilePath, (int)BatchSeconds.Value);
            //foreach (var batch in audioData.Select((value, i) => (value, i)))
            //{
            //    srtBatches.Add(await TranscribeAsync(batch.value, Combo2.SelectedValue.ToString(), Switch1.IsOn ? TaskType.Translate : TaskType.Transcribe, batch.i, (int)BatchSeconds.Value));
            //}
            var srtFilePath = Utils.SaveSrtContentToTempFile(srtBatches, Path.GetFileNameWithoutExtension(VideoFilePath));
            //OpenVideo(addSubtitles(VideoFilePath, srtFilePath));
            OpenVideo(VideoFilePath, srtFilePath);
        }

        private List<Chunk> GetTimeStamps(List<DetectionResult> voiceAreas, double totalSeconds)
        {
            const int maxLength = 30;
            List<Chunk> chunks = new();
            int currChunk = 1;
            double startTime = 0;
            for(int i=1;i<voiceAreas.Count - 1;i+=2)
            {
                if (voiceAreas[i].Seconds > startTime + maxLength && chunks.Count < currChunk)
                {
                    chunks.Add(new Chunk(startTime, currChunk * maxLength));
                    currChunk++;
                    startTime = currChunk * maxLength;
                }
                //TODO: This is a very basic check, we can check for a threshold of values instead, Amrutha will work on that
                if (voiceAreas[i].Seconds <= startTime + maxLength && (i == voiceAreas.Count - 1 || voiceAreas[i + 1].Seconds > startTime + maxLength)) {
                    chunks.Add(new Chunk(startTime, voiceAreas[i].Seconds));
                    currChunk++;
                    startTime = voiceAreas[i].Seconds;
                }   
            }

            double j;
            //Sometimes the last chunk is really large
            for(j=startTime; j<totalSeconds;j+= maxLength)
            {
                chunks.Add(new Chunk(j, Math.Min(j + maxLength, totalSeconds)));
            }
            return chunks;
        }

        private async void GetAudioFromVideoButtonClick(object sender, RoutedEventArgs e)
        {
            // Clear previous returned file name, if it exists, between iterations of this scenario

            // Create a file picker
            var openPicker = new Windows.Storage.Pickers.FileOpenPicker();

            // See the sample code below for how to make the window accessible from the App class.
            var window = this;

            // Retrieve the window handle (HWND) of the current WinUI 3 window.
            var hWnd = WinRT.Interop.WindowNative.GetWindowHandle(window);

            // Initialize the file picker with the window handle (HWND).
            WinRT.Interop.InitializeWithWindow.Initialize(openPicker, hWnd);

            // Set options for your file picker
            openPicker.ViewMode = PickerViewMode.Thumbnail;
            openPicker.FileTypeFilter.Add("*");

            // Open the picker for the user to pick a file
            var file = await openPicker.PickSingleFileAsync();
            if (file != null)
            {
                PickAFileOutputTextBlock.Text = "File selected: " + file.Name;
            }
            else
            {
                PickAFileOutputTextBlock.Text = "Operation cancelled.";
            }

            this.VideoFilePath = file.Path;

        }

        public async Task<byte[]> ConvertStorageFileToByteArray(StorageFile storageFile)
        {
            if (storageFile == null)
                throw new ArgumentNullException(nameof(storageFile));

            // Open the file for reading
            using (IRandomAccessStreamWithContentType stream = await storageFile.OpenReadAsync())
            {
                // Create a buffer large enough to hold the file's contents
                var buffer = new Windows.Storage.Streams.Buffer((uint)stream.Size);

                // Read the file into the buffer
                await stream.ReadAsync(buffer, buffer.Capacity, InputStreamOptions.None);

                // Convert the buffer to a byte array and return it
                return buffer.ToArray();
            }
        }

        public class DetectionResult
        {
            public string Type { get; set; }
            public double Seconds { get; set; }
        }

        public class Chunk
        {
            public double start { get; set; }
            public double end { get; set; }

            public Chunk(double start, double end)
            {
                this.start = start;
                this.end = end;
            }

        }

        private List<Chunk> DetectVoice(byte[] audioBytes)
        {
            var assemblyLocation = Assembly.GetExecutingAssembly().Location;
            var assemblyPath = Path.GetDirectoryName(assemblyLocation);
            string vadModelPath = Path.GetFullPath(Path.Combine(assemblyPath, "..\\..\\..\\..\\..\\Assets\\silero_vad.onnx"));

            var MODEL_PATH = vadModelPath;
            var SAMPLE_RATE = 16000;
            var START_THRESHOLD = 0.5f;
            var END_THRESHOLD = 0.45f;
            var MIN_SILENCE_DURATION_MS = 600;
            var SPEECH_PAD_MS = 500;
            var WINDOW_SIZE_SAMPLES = 2048;

            SlieroVadDetector vadDetector;
            vadDetector = new SlieroVadDetector(MODEL_PATH, START_THRESHOLD, END_THRESHOLD, SAMPLE_RATE, MIN_SILENCE_DURATION_MS, SPEECH_PAD_MS);

            int bytesPerSample = 2; //TODO: This should be 2 right?
            int bytesPerWindow = WINDOW_SIZE_SAMPLES * bytesPerSample;

            float totalSeconds = audioBytes.Length / (SAMPLE_RATE * 2);
            var result = new List<DetectionResult>();

            for (int offset = 0; offset + bytesPerWindow <= audioBytes.Length; offset += bytesPerWindow)
            {
                byte[] data = new byte[bytesPerWindow];
                Array.Copy(audioBytes, offset, data, 0, bytesPerWindow);

                // Simulating the process as if data was being read in chunks
                try
                {
                    var detectResult = vadDetector.Apply(data, true);
                    // iterate over detectResult and apply the data to result:
                    foreach (var (key, value) in detectResult)
                    {
                        result.Add(new DetectionResult { Type = key, Seconds = value });
                    }
                }
                catch (Exception e)
                {
                    Console.Error.WriteLine($"Error applying VAD detector: {e.Message}");
                    // Depending on the need, you might want to break out of the loop or just report the error
                }
            }
            var stamps = GetTimeStamps(result, totalSeconds);
            return stamps;
        }

        private async Task<string> TranscribeAsync(float[] pcmAudioData, string inputLanguage, TaskType taskType, int batchSeconds)
        {
            var assemblyLocation = Assembly.GetExecutingAssembly().Location;
            var assemblyPath = Path.GetDirectoryName(assemblyLocation);
            string whisperModelPath = Path.GetFullPath(Path.Combine(assemblyPath, "..\\..\\..\\..\\..\\Assets\\model_medium.onnx"));
            
            //string modelPath = "C:\\Users\\gkhmyznikov\\Develop\\temp\\model_srb_only.onnx";
            //string modelPath = "C:\\Users\\gkhmyznikov\\Develop\\temp\\model_17.onnx";


            //var modelName = "model.onnx";
            //Uri fileUri = new Uri($"ms-appdata:///Assets/{modelName}");
            //StorageFile file = await StorageFile.GetFileFromApplicationUriAsync(fileUri);
            //byte[] modelData = await ConvertStorageFileToByteArray(file);

            var audioTensor = new DenseTensor<float>(pcmAudioData, [1, pcmAudioData.Length]);
            var timestampsEnableTensor = new DenseTensor<int>(new[] { 1 }, [1]);

            int task = (int)taskType;
            int langCode = Utils.GetLangId(inputLanguage);
            var decoderInputIds = new int[] { 50258, langCode, task };
            var langAndModeTensor = new DenseTensor<int>(decoderInputIds, [1, 3]);

            SessionOptions options = new SessionOptions();
            options.RegisterOrtExtensions();
            //options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            //options.EnableMemoryPattern = false;
            //options.AppendExecutionProvider_DML(1);
            //options.LogSeverityLevel = 0;
            options.AppendExecutionProvider_CPU();

            using var session = new InferenceSession(whisperModelPath, options);

            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("audio_pcm", audioTensor),
                NamedOnnxValue.CreateFromTensor("min_length", new DenseTensor<int>(new int[] { 0 }, [1])),
                NamedOnnxValue.CreateFromTensor("max_length", new DenseTensor<int>(new int[] { 448 }, [1])),
                NamedOnnxValue.CreateFromTensor("num_beams", new DenseTensor<int>(new int[] {2}, [1])),
                NamedOnnxValue.CreateFromTensor("num_return_sequences", new DenseTensor<int>(new int[] { 1 }, [1])),
                NamedOnnxValue.CreateFromTensor("length_penalty", new DenseTensor<float>(new float[] { 1.0f }, [1])),
                NamedOnnxValue.CreateFromTensor("repetition_penalty", new DenseTensor<float>(new float[] { 1.0f }, [1])),
                //NamedOnnxValue.CreateFromTensor("attention_mask", config.attention_mask)
                NamedOnnxValue.CreateFromTensor("logits_processor", timestampsEnableTensor),
                NamedOnnxValue.CreateFromTensor("decoder_input_ids", langAndModeTensor)
            };

            using var results = session.Run(inputs);
            var output = ProcessResults(results);
            //var srtPath = Utils.ConvertToSrt(output, Path.GetFileNameWithoutExtension(videoFileName), batch);
            var srtText = Utils.ConvertToSrt(output, batchSeconds);

            //PickAFileOutputTextBlock.Text = "Generated SRT File at: " + srtPath;

            return srtText;
        }

        private static string ProcessResults(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
        {
            foreach (var result in results)
            {
                if (result.Name == "str") // Replace "output_name" with the actual output name of your model
                {
                    var tensor = result.AsTensor<string>();
                    return tensor.GetValue(0); // Simplified; actual extraction may differ
                }
            }

            return "Unable to extract transcription.";
        }
        private string FixPath(string path)
        {
            return path.Replace("\\", "\\\\\\\\").Insert(1, "\\\\");
        }

        private string addSubtitles(string videoPath, string srtPath)
        {
            string documentsFolderPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            string outputFilePath = Path.Combine(documentsFolderPath, Path.GetFileNameWithoutExtension(videoPath) + "Subtitled" + Path.GetExtension(videoPath));

            if (File.Exists(outputFilePath))
            {
                File.Delete(outputFilePath);
            }

            var ffMpegConverter = new FFMpegConverter();
            string newSrtPath = FixPath(srtPath);
            ffMpegConverter.Invoke($"-i \"{videoPath}\" -vf subtitles=\"{newSrtPath}\"  \"{outputFilePath}\"");

            return outputFilePath;
        }

        private void OpenVideo(string videoFilePath, string srtFilePath)
        {
            //ProcessStartInfo startInfo = new ProcessStartInfo
            //{
            //    FileName = videoFilePath,
            //    UseShellExecute = true,
            //    Arguments = srtFilePath
            //};

            string vlcPath = @"C:\Program Files\VideoLAN\VLC\vlc.exe";
            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = vlcPath,
                UseShellExecute = false,
                Arguments = $"\"{videoFilePath}\" --sub-file=\"{srtFilePath}\" --no-osd"
            };

            Process.Start(startInfo);
        }

    }
}
