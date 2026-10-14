using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JustModelAnsamble
{
    internal class DecisionTreeRegressor
    {
        private DecisionNode _root;
        private const int MaxDepth = 6;
        private const int MinSamples = 3;

        public void Fit(List<CropSample> data) => _root = Build(data, 0);

        private DecisionNode Build(List<CropSample> data, int depth)
        {
            if (data.Count < MinSamples || depth >= MaxDepth)
                return new DecisionNode { Prediction = data.Average(s => s.Yield) };

            var (bestFeature, bestThreshold) = FindBestSplit(data);

            if (bestFeature == null)
                return new DecisionNode { Prediction = data.Average(s => s.Yield) };

            var left = data.Where(s => GetFeature(s, bestFeature) <= bestThreshold).ToList();
            var right = data.Where(s => GetFeature(s, bestFeature) > bestThreshold).ToList();

            if (left.Count == 0 || right.Count == 0)
                return new DecisionNode { Prediction = data.Average(s => s.Yield) };

            return new DecisionNode
            {
                Feature = bestFeature,
                Threshold = bestThreshold,
                Left = Build(left, depth + 1),
                Right = Build(right, depth + 1)
            };
        }

        private (string, double) FindBestSplit(List<CropSample> data)
        {
            string[] features = { "Temperature", "Rainfall", "SoilMoisture", "Ph" };
            string bestFeature = null;
            double bestThreshold = 0, bestMse = double.MaxValue;

            foreach (var feature in features)
            {
                var values = data.Select(s => GetFeature(s, feature)).Distinct().OrderBy(v => v);
                foreach (var threshold in values)
                {
                    var left = data.Where(s => GetFeature(s, feature) <= threshold).ToList();
                    var right = data.Where(s => GetFeature(s, feature) > threshold).ToList();
                    if (left.Count == 0 || right.Count == 0) continue;

                    double mse = WeightedMse(left, right);
                    if (mse < bestMse)
                    {
                        bestMse = mse;
                        bestFeature = feature;
                        bestThreshold = threshold;
                    }
                }
            }
            return (bestFeature, bestThreshold);
        }

        private double WeightedMse(List<CropSample> left, List<CropSample> right)
        {
            double total = left.Count + right.Count;
            double lMean = left.Average(s => s.Yield);
            double rMean = right.Average(s => s.Yield);
            double lMse = left.Sum(s => Math.Pow(s.Yield - lMean, 2));
            double rMse = right.Sum(s => Math.Pow(s.Yield - rMean, 2));
            return (lMse + rMse) / total;
        }

        private double GetFeature(CropSample s, string name) => name switch
        {
            "Temperature" => s.Temperature,
            "Rainfall" => s.Rainfall,
            "SoilMoisture" => s.SoilMoisture,
            "Ph" => s.Ph,
            _ => 0
        };

        public double Predict(CropSample sample)
        {
            var node = _root;
            while (!node.IsLeaf)
            {
                double val = GetFeature(sample, node.Feature);
                node = val <= node.Threshold ? node.Left : node.Right;
            }
            return node.Prediction.Value;
        }
    }
}
