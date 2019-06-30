using System;
using System.IO;
using Microsoft.ML;

namespace TaxiFarePrediction
{
    class Program
    {
        /// <summary>
        /// Ścieżka do pliku trenującego
        /// </summary>
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        /// <summary>
        /// Ścieżka do pliku testującego
        /// </summary>
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        /// <summary>
        /// Ścieżka do pliku wynikowego
        /// </summary>
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            // Nowy kontekst ML
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _trainDataPath);

        }

        /// <summary>
        /// Metoda trenująca
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="dataPath"></param>
        /// <returns></returns>
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {

        }
    }
}
