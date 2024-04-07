using Libs.SemanticSearch.MiniLM;
using Libs.SemanticSearch;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Navigation;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Libs.VoiceActivity;
using SubtitleGenerator;
using System.Diagnostics;
using System.Threading.Tasks;
using Libs.VoiceRecognition;
using static Libs.VoiceRecognition.Whisper;
using System.Reflection;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace AudioEditor
{
    /// <summary>
    /// An empty window that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            //testSemanticSearch();
            //testChunkingAndWhisper();
        }

        private void myButton_Click(object sender, RoutedEventArgs e)
        {
            myButton.Content = "Clicked";
        }

        private async void testChunkingAndWhisper()
        {
            var assemblyLocation = Assembly.GetExecutingAssembly().Location;
            var assemblyPath = Path.GetDirectoryName(assemblyLocation);
            string inputAudioPath = Path.GetFullPath(Path.Combine(assemblyPath, "Resources\\test.mp3"));

            var audioBytes = Utils.LoadAudioBytes(inputAudioPath);
            var srtBatches = new List<string>();

            var dynamicChunks = Chunking.SmartChunking(audioBytes);

            foreach (var chunk in dynamicChunks.Select((value, i) => (value, i)))
            {
          
                var audioSegment = Utils.ExtractAudioSegment(inputAudioPath, chunk.value.start, chunk.value.end - chunk.value.start);

                var transcription = await TranscribeAsync(audioSegment, "en", TaskType.Transcribe, (int)chunk.value.start);

                srtBatches.Add(transcription);
            }

            var srtFilePath = Utils.SaveSrtContentToTempFile(srtBatches, Path.GetFileNameWithoutExtension(inputAudioPath));
        }
        private void testSemanticSearch ()
        {
            var miniLM = new MiniLML6v2(new MiniLML6v2Config());

            // Corpus of multiple strings
            string[] corpusArray = {
                "Wow, Xiao Wang.",
                "Hey, long time no see. How have you been?",
                "I'm good, I'm good.",
                "I'm busy, I changed jobs.",
                "How's your new job?",
                "I'm not in school anymore, I'm a lawyer.",
                "That's good.",
                "How's your boyfriend?",
                "He's not bad.",
                "He's been in China for the past two weeks.",
                "He's seen his parents, he's seen his friends.",
                "You live here too?",
                "Yes, we bought a house here.",
                "That's great. How's your husband?",
                "He's also very good. We'll go see a basketball game next weekend.",
                "I didn't know you all liked basketball.",
                "Xiao Ming loves basketball very much. We go see a basketball game every week.",
                "That's great. I'm leaving first. I have something else to do.",
                "Okay, you're really busy. Goodbye.",
                "Xiao Wang, wait a minute. Please give me your phone number.",
                "My phone number is...",
                "537-408.",
                "Call me.",
                "Okay, I will. Bye.",
            };

            // Single search query
            string searchQuery = "activities";
            string[] searchQueryArray = { searchQuery };

            // Generate embeddings for the corpus
            var corpusEmbeddings = corpusArray
                .Select(text => miniLM.GenerateEmbeddings(new string[] { text }))
                .ToArray();

            // Generate embeddings for the search query
            var searchQueryEmbeddings = miniLM.GenerateEmbeddings(searchQueryArray); // Assuming one query

            // Calculate similarities
            var similarityScores = corpusEmbeddings
                .Select(embedding => Similarity.CosineSimilarity(searchQueryEmbeddings, embedding))
                .ToArray();

            // Order by similarity in desc order and select indexes
            var sortedIndexBySimilarity = similarityScores
                .Select((score, index) => new { Index = index, Score = score, Text = corpusArray[index] })
                .OrderByDescending(x => x.Score)
                .ToArray();


            string[] query1 = { "That is a happy person" };
            //string[] query2 = { "That is a happy person" };
            //var query1Embeddings = miniLM.GenerateEmbeddings(query1);
            //var query2Embeddings = miniLM.GenerateEmbeddings(query2);
            //TorchTensor corpus = Float32Tensor.from(
            //        query1Embeddings,
            //        [1, query1Embeddings.Length]);
            //TorchTensor query = Float32Tensor.from(
            //        query2Embeddings,
            //        [1, query2Embeddings.Length]);
            //var topK = Similarity.TopKByCosineSimilarity(
            //    corpus,
            //    query,
            //    query1.Length);

            //var scores = topK.Values.Data<float>().GetEnumerator();
            //foreach (var index in topK.Indexes.Data<long>().ToArray())
            //{
            //    scores.MoveNext();
            //    Console.WriteLine($"Cosine similarity score: {scores.Current*100:f12}");
            //    Console.WriteLine();
            //}

            //var dotP = Similarity.DotProduct(query1Embeddings, query2Embeddings);
            //Console.WriteLine($"Dot product similarity score: {dotP * 100:f12}");
        }
    }
}
