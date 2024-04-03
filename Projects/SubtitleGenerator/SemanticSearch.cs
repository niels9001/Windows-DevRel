using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text.RegularExpressions;
using System.Timers;
using TorchSharp;
using Microsoft.ML.OnnxRuntime;
using BERTTokenizers.Base;
using static TorchSharp.torch.nn;

namespace SubtitleGenerator
{
    public class SemanticSearch
    {
        private static string modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets");
        private static InferenceSession _inferenceSession;
        private static Vector[] _embeddings;
        private string[] _content;
        private static void InitModel()
        {
            if (_inferenceSession != null)
            {
                return;
            }

            var sessionOptions = new SessionOptions();
            sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            sessionOptions.AppendExecutionProvider_CPU();
            _inferenceSession = new InferenceSession($@"{modelDir}\semsearchmodel.onnx", sessionOptions);
        }

        public static int[] GetRankings(string query, params string[] sentences)
        {
            var vectors = new Vector[sentences.Length];
            for (int i = 0; i < sentences.Length; i++)
            {
                var content = Regex.Replace(sentences[i], @"[^\u0000-\u007F]", "");
                content = Regex.Replace(content, @"^\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n", string.Empty, RegexOptions.Multiline);
                vectors[i] = new Vector { data = GetEmbeddings(content) };
            }
            _embeddings = vectors;

            var queryEmbedding = GetEmbeddings(query);
            var ranking = CalculateRanking(new Vector { data = queryEmbedding }, _embeddings);

            return ranking;

        }

        public static int[] CalculateRanking(Vector searchVector, Vector[] vectors)
        {
            float[] scores = new float[vectors.Length];
            int[] indexranks = new int[vectors.Length];

            for (int i = 0; i < vectors.Length; i++)
            {
                var score = CosineSimilarity(vectors[i].data, searchVector.data);
                scores[i] = score;
            }

            var indexedFloats = scores.Select((value, index) => new { Value = value, Index = index })
              .ToArray();

            // Sort the indexed floats by value in descending order
            Array.Sort(indexedFloats, (a, b) => b.Value.CompareTo(a.Value));

            // Extract the top k indices
            indexranks = indexedFloats.Select(item => item.Index).ToArray();

            return indexranks;
        }
        public static float[] GetEmbeddings(params string[] sentences)
        {
            InitModel();

            //sentences = new() { "something" };
            var tokenizer = new MyTokenizer($@"{modelDir}\vocab.txt");
            var tokens = tokenizer.Tokenize(sentences.ToArray());
            var encoded = tokenizer.Encode(tokens.Count(), sentences.ToArray());

            var input = new ModelInput()
            {
                InputIds = encoded.Select(t => t.InputIds).ToArray(),
                AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                TokenTypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
            };
         
            var runOptions = new RunOptions();


            // Create input tensors over the input data.
            using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(input.InputIds,
                  new long[] { sentences.Length, input.InputIds.Length });

            using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(input.AttentionMask,
                  new long[] { sentences.Length, input.AttentionMask.Length });

            using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(input.TokenTypeIds,
                  new long[] { sentences.Length, input.TokenTypeIds.Length });

            var inputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "attention_mask", attMaskOrtValue },
                { "token_type_ids", typeIdsOrtValue }
            };

            using var output = _inferenceSession.Run(runOptions, inputs, _inferenceSession.OutputNames);
            var data = output.ToList()[0].GetTensorDataAsSpan<float>().ToArray();


            var sentence_embeddings = MeanPooling(data, input.AttentionMask, sentences.Length, input.AttentionMask.Length, 384);
            var denom = sentence_embeddings.norm(1, true, 2).clamp_min(1e-12).expand_as(sentence_embeddings);
            var results = sentence_embeddings / denom;
            return results.data<float>().ToArray();

        }

        public void ProcessQuery(string text)
        {
            text = text.Trim();
            _content = text.Split('.', '\r', '\n');
            _content = _content.Where(x => !string.IsNullOrWhiteSpace(x)).ToArray();

            if (_content.Length == 0)
            {
                return;
            }

            var vectors = new Vector[_content.Length];
            for (int i = 0; i < _content.Length; i++)
            {
                var content = Regex.Replace(_content[i], @"[^\u0000-\u007F]", "");
                vectors[i] = new Vector { data = GetEmbeddings(content) };
            }
            _embeddings = vectors;
        }

        public static torch.Tensor MeanPooling(float[] embeddings, long[] attentionMask, long batchSize, long sequence, int hiddenSize)
        {
            var tokenEmbeddings = torch.tensor(embeddings, new[] { batchSize, sequence, hiddenSize });
            var attentionMaskExpanded = torch.tensor(attentionMask, new[] { batchSize, sequence }).unsqueeze(-1).expand(tokenEmbeddings.shape).@float();

            var sumEmbeddings = (tokenEmbeddings * attentionMaskExpanded).sum(new[] { 1L });
            var sumMask = attentionMaskExpanded.sum(new[] { 1L }).clamp(1e-9, float.MaxValue);

            return sumEmbeddings / sumMask;
        }

        public static float CheckOverflow(double x)
        {
            if (x >= double.MaxValue)
            {
                throw new OverflowException("operation caused overflow");
            }
            return (float)x;
        }
        public static float DotProduct(float[] a, float[] b)
        {
            float result = 0.0f;
            for (int i = 0; i < a.Length; i++)
            {
                result = CheckOverflow(result + CheckOverflow(a[i] * b[i]));
            }
            return result;
        }
        public static float Magnitude(float[] v)
        {
            float result = 0.0f;
            for (int i = 0; i < v.Length; i++)
            {
                result = CheckOverflow(result + CheckOverflow(v[i] * v[i]));
            }
            return (float)Math.Sqrt(result);
        }
        public static float CosineSimilarity(float[] v1, float[] v2)
        {
            if (v1.Length != v2.Length)
            {
                throw new ArgumentException("Vectors must have the same length.");
            }
            int size = v1.Length;
            float m1 = Magnitude(v1);
            float m2 = Magnitude(v2);
            /*                        var normalizedList1 = raw1.Select(o => o / m1).ToArray();
                                    var normalizedList2 = raw2.Select(o => o / m2).ToArray();
            */
            /*// Vectors should already be normalized.
            if (Math.Abs(m1 - m2) > 0.4f || Math.Abs(m1 - 1.0f) > 0.4f)
            {
                throw new InvalidOperationException("Vectors are not normalized.");
            }*/
            return DotProduct(v1, v2);
        }
    }

    public struct Vector
    {
        public float[] data;
    }

    public class ModelInput
    {
        public long[] InputIds { get; set; }

        public long[] AttentionMask { get; set; }

        public long[] TokenTypeIds { get; set; }
    }

    public class MyTokenizer : UncasedTokenizer
    {
        public MyTokenizer(string vocabPath) : base(vocabPath) { }
    }
}
