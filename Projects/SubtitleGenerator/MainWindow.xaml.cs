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

            var dynamicChunks = Chunking.SmartChunking(audioBytes);

            //// Transform the operations into a collection of tasks.
            //var transcriptionTasks = dynamicChunks.Select(async (chunk, i) =>
            //{
            //    var audioSegment = Utils.ExtractAudioSegment(VideoFilePath, chunk.start, chunk.end - chunk.start);
            //    return await TranscribeAsync(audioSegment, Combo2.SelectedValue.ToString(), Switch1.IsOn ? TaskType.Translate : TaskType.Transcribe, (int)chunk.start);
            //}).ToList();

            //// Wait for all tasks to complete while preserving the order.
            //var transcriptions = await Task.WhenAll(transcriptionTasks);

            //// At this point, 'transcriptions' contains the results in the order of 'dynamicChunks'.
            //// Add the transcription results to your srtBatches list.
            //var srtBatches = transcriptions.ToList();

            foreach (var chunk in dynamicChunks.Select((value, i) => (value, i)))
            {
                // Assuming you have or can create a method for extracting audio segments by start/end times.
                // This will involve modifying your Utils.ExtractAudioFromVideo method or creating a new one that can handle this.
                // The new method might return a byte array of the audio for the specified chunk.
                // The method might look like: Utils.ExtractAudioSegment(audioBytes, chunk.start, chunk.end);
                var audioSegment = Utils.ExtractAudioSegment(VideoFilePath, chunk.value.start, chunk.value.end - chunk.value.start);

                // Assuming TranscribeAsync can take the audio segment directly along with other parameters.
                // The provided chunk might also be used to adjust how you number/name the subtitle batches.
                // Adjusting the call to TranscribeAsync to use the chunk's start time or a combination of start and end times for naming uniqueness.
                var transcription = await TranscribeAsync(audioSegment, Combo2.SelectedValue.ToString(), Switch1.IsOn ? TaskType.Translate : TaskType.Transcribe, (int)chunk.value.start);

                srtBatches.Add(transcription);
            }

            var srtFilePath = Utils.SaveSrtContentToTempFile(srtBatches, Path.GetFileNameWithoutExtension(VideoFilePath));

            // srtBatches = new() { "So for this project, we used Whisper which is a transformer based model available on how you face that's that performs transcription and translation really well for a lot of different languages."};

            //srtBatches = srtBatches.Select(s => new string(s.Where(c => char.IsLetter(c) || char.IsWhiteSpace(c)).ToArray())).ToList();
            srtBatches = srtBatches.SelectMany(s => s.Split("\n")).Where(s => !string.IsNullOrEmpty(s)).Where(s => s.Length > 2).ToList();

            var rankings = SemanticSearch.GetRankings("games", srtBatches.ToArray());

            var mostRelevantSentence = srtBatches[rankings[0]];
            //OpenVideo(addSubtitles(VideoFilePath, srtFilePath));
            OpenVideo(VideoFilePath, srtFilePath);
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
        

        private async Task<string> TranscribeAsync(float[] pcmAudioData, string inputLanguage, TaskType taskType, int batchSeconds)
        {
            var assemblyLocation = Assembly.GetExecutingAssembly().Location;
            var assemblyPath = Path.GetDirectoryName(assemblyLocation);
            string whisperModelPath = Path.GetFullPath(Path.Combine(assemblyPath, "Assets\\model_small.onnx"));
            
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
            
            // for multithread need to try AsyncRun
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
