
using System;
using System.IO;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using NReco.VideoConverter;

namespace SubtitleGenerator
{
    public static class Utils
    {
        public static string ConvertToSrt(string subtitleString, int offsetInSeconds)
        {
            Regex pattern = new Regex(@"<\|([\d.]+)\|>([^<]+)<\|([\d.]+)\|>");
            MatchCollection matches = pattern.Matches(subtitleString);
            // Placeholder for srt content
            string srtContent = "";

            // Calculate the time offset based on the batch number. Each batch represents an additional 30 seconds.
            double batchOffset = offsetInSeconds; // batchIndex * batchSizeInSeconds; // 30 seconds per batch

            for (int i = 0; i < matches.Count; i++)
            {
                // Parse the original start and end times
                double start = double.Parse(matches[i].Groups[1].Value);
                double end = double.Parse(matches[i].Groups[3].Value);

                // Apply the batch offset to the start and end times
                start += batchOffset;
                end += batchOffset;

                // Convert the adjusted start and end times into the SRT format
                string startSrt = $"{(int)(start / 3600):D2}:{(int)((start % 3600) / 60):D2}:{(int)(start % 60):D2},{(int)((start * 1000) % 1000):D3}";
                string endSrt = $"{(int)(end / 3600):D2}:{(int)((end % 3600) / 60):D2}:{(int)(end % 60):D2},{(int)((end * 1000) % 1000):D3}";

                // Build the SRT content string, incrementing the subtitle index by 1 for readability
                srtContent += $"{i + 1}\n{startSrt} --> {endSrt}\n{matches[i].Groups[2].Value.Trim()}\n\n";
            }

            // The SaveSrtContentToTempFile method needs to exist and handle the saving of the SRT content to a file
            // Make sure to implement it or adjust this part as per your application's requirements
            return srtContent;
            //return SaveSrtContentToTempFile(srtContent, fileName);
        }

        public static string SaveSrtContentToTempFile(List<string> srtContent, string fileName)
        {
            string srtFilePath = "";
            try
            {
                // Use MyDocuments as the directory to save the SRT file
                string documentsFolderPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);

                // Create a unique filename for the SRT file
                string uniqueFileName = $"{fileName}.srt";

                // Combine the documents folder path with the unique file name to get the full path
                srtFilePath = Path.Combine(documentsFolderPath, uniqueFileName);

                // Join the list of strings into a single string with newline characters
                string combinedContent = string.Join("\n", srtContent);

                // Write the combined string content to the file
                File.WriteAllText(srtFilePath, combinedContent);

                Console.WriteLine($"SRT file saved to: {srtFilePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error saving SRT file: {ex.Message}");
            }
            return srtFilePath;
        }

        public static byte[] LoadAudioBytes(string file)
        {
            var ffmpeg = new FFMpegConverter();
            var output = new MemoryStream();

            var extension = Path.GetExtension(file).Substring(1);

            // Convert to PCM
            ffmpeg.ConvertMedia(inputFile: file,
                                inputFormat: extension,
                                outputStream: output,
                                //  DE s16le PCM signed 16-bit little-endian
                                outputFormat: "s16le",
                                new ConvertSettings()
                                {
                                    AudioCodec = "pcm_s16le",
                                    AudioSampleRate = 16000,
                                    // Convert to mono
                                    CustomOutputArgs = "-ac 1"
                                });

            return output.ToArray();
        }

        public static List<float[]> ExtractAudioFromVideo(string inPath, int batchSizeInSeconds)
        {
            try
            {
                var extension = System.IO.Path.GetExtension(inPath).Substring(1);
                var output = new MemoryStream();

                var convertSettings = new ConvertSettings
                {
                    AudioCodec = "pcm_s16le",
                    AudioSampleRate = 16000,
                    CustomOutputArgs = "-vn -ac 1",
                };

                var ffMpegConverter = new FFMpegConverter();
                ffMpegConverter.ConvertMedia(
                    inputFile: inPath,
                    inputFormat: extension,
                    outputStream: output,
                    outputFormat: "s16le",

                    convertSettings);

                var buffer = output.ToArray();
                // Calculate number of samples in 30 seconds; Sample rate * 30 (assuming 16K sample rate)
                int samplesPerSeconds = 16000 * batchSizeInSeconds;
                // Calculate bytes per sample, assuming 16-bit depth (2 bytes per sample)
                int bytesPerSample = 2;

                // Calculate total samples in the buffer
                int totalSamples = buffer.Length / bytesPerSample;

                List<float[]> batches = new List<float[]>();
                for (int startSample = 0; startSample < totalSamples; startSample += samplesPerSeconds)
                {
                    int endSample = Math.Min(startSample + samplesPerSeconds, totalSamples);
                    int numSamples = endSample - startSample;
                    float[] batch = new float[numSamples];

                    for (int i = 0; i < numSamples; i++)
                    {
                        int bufferIndex = (startSample + i) * bytesPerSample;
                        short sample = (short)(buffer[bufferIndex + 1] << 8 | buffer[bufferIndex]);
                        batch[i] = sample / 32768.0f;
                    }

                    batches.Add(batch);
                }

                return batches;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error during the audio extraction: " + ex.Message);
                return new List<float[]>(0);
            }
        }

        public static float[] ExtractAudioSegment(string inPath, double startTimeInSeconds, double segmentDurationInSeconds)
        {
            try
            {
                var extension = System.IO.Path.GetExtension(inPath).Substring(1);
                var output = new MemoryStream();

                var convertSettings = new ConvertSettings
                {
                    Seek = (float?)startTimeInSeconds,
                    MaxDuration = (float?)segmentDurationInSeconds,
                    AudioCodec = "pcm_s16le",
                    AudioSampleRate = 16000,
                    CustomOutputArgs = "-vn -ac 1",
                };

                var ffMpegConverter = new FFMpegConverter();
                ffMpegConverter.ConvertMedia(
                    inputFile: inPath,
                    inputFormat: extension,
                    outputStream: output,
                    outputFormat: "s16le",
                    convertSettings);

                var buffer = output.ToArray();
                int bytesPerSample = 2; // Assuming 16-bit depth (2 bytes per sample)

                // Calculate total samples in the buffer
                int totalSamples = buffer.Length / bytesPerSample;
                float[] samples = new float[totalSamples];

                for (int i = 0; i < totalSamples; i++)
                {
                    int bufferIndex = i * bytesPerSample;
                    short sample = (short)(buffer[bufferIndex + 1] << 8 | buffer[bufferIndex]);
                    samples[i] = sample / 32768.0f; // Normalize to range [-1,1] for floating point samples
                }

                return samples;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error during the audio extraction: " + ex.Message);
                return new float[0]; // Return an empty array in case of exception
            }
        }
    }
}
