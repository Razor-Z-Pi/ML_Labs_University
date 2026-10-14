using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace JustModelAnsamble
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private DecisionTreeRegressor _model;
        private List<CropSample> _trainingData;

        public MainWindow()
        {
            InitializeComponent();
            _trainingData = GenerateSyntheticData(500);
            TrainModel();
            UpdatePrediction();
        }

        private void TrainModel()
        {
            _model = new DecisionTreeRegressor();
            _model.Fit(_trainingData);
        }

        private void Retrain_Click(object sender, RoutedEventArgs e)
        {
            _trainingData = GenerateSyntheticData(500);
            TrainModel();
            UpdatePrediction();

            // Анимация вспышки
            var anim = new DoubleAnimation(0.4, 1.0, TimeSpan.FromSeconds(0.5));
            YieldText.BeginAnimation(OpacityProperty, anim);
        }

        private void Slider_ValueChanged(object sender,
            RoutedPropertyChangedEventArgs<double> e)
        {
            if (_model != null) UpdatePrediction();
        }

        private void UpdatePrediction()
        {
            var sample = new CropSample
            {
                Temperature = TempSlider.Value,
                Rainfall = RainSlider.Value,
                SoilMoisture = MoistSlider.Value,
                Ph = PhSlider.Value
            };

            double predicted = _model.Predict(sample);

            AnimateText(YieldText, predicted);
            AnimateBar(YieldBar, Math.Min(predicted, 120));

            // Оценка риска
            (string risk, string advice) = EvaluateRisk(sample, predicted);
            RiskText.Text = risk;
            AdviceText.Text = advice;
        }

        private void AnimateText(TextBlock tb, double value)
        {
            // Плавный счёт от текущего к новому
            double from = 0;
            if (double.TryParse(tb.Text.Replace(" ц/га", ""), out var cur)) from = cur;

            var anim = new DoubleAnimation(from, value,
                TimeSpan.FromMilliseconds(450))
            {
                EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut }
            };

            var dummy = new DependencyProperty[] { };
            var counter = new DoubleAnimationUsingKeyFrames();
            var timer = new System.Windows.Threading.DispatcherTimer
            { Interval = TimeSpan.FromMilliseconds(16) };
            double start = from, total = value - from, elapsed = 0, duration = 450;

            timer.Tick += (s, e) =>
            {
                elapsed += 16;
                double t = Math.Min(elapsed / duration, 1);
                double eased = 1 - Math.Pow(1 - t, 3);
                tb.Text = $"{start + total * eased:F1} ц/га";
                if (t >= 1) timer.Stop();
            };
            timer.Start();
        }

        private void AnimateBar(ProgressBar bar, double target)
        {
            var anim = new DoubleAnimation(bar.Value, target,
                TimeSpan.FromMilliseconds(500))
            {
                EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut }
            };
            bar.BeginAnimation(ProgressBar.ValueProperty, anim);
        }

        private (string risk, string advice) EvaluateRisk(CropSample s, double yield)
        {
            if (yield < 30)
                return ("Высокий риск",
                        "Рекомендуется увеличить полив и скорректировать pH до 6.0–7.0. " +
                        "Рассмотрите засухоустойчивые культуры.");

            if (yield < 60)
                return ("Средний риск",
                        "Условия приемлемы. Следите за влажностью почвы в период цветения.");

            return ("Низкий риск",
                    "Оптимальные условия! Ожидается хороший урожай. " +
                    "Поддерживайте текущий режим агротехники.");
        }

        private List<CropSample> GenerateSyntheticData(int n)
        {
            var rnd = new Random(42);
            var list = new List<CropSample>(n);
            for (int i = 0; i < n; i++)
            {
                double t = 10 + rnd.NextDouble() * 25;       // 10..35 °C
                double r = 100 + rnd.NextDouble() * 700;     // 100..800 мм
                double m = 20 + rnd.NextDouble() * 60;       // 20..80 %
                double ph = 4.5 + rnd.NextDouble() * 3.5;    // 4.5..8

                // Имитационная зависимость урожая
                double optimal = 1.0
                    - Math.Pow((t - 24) / 15, 2) * 0.4
                    - Math.Pow((r - 500) / 400, 2) * 0.3
                    - Math.Pow((m - 60) / 40, 2) * 0.2
                    - Math.Pow((ph - 6.5) / 1.5, 2) * 0.2;

                optimal = Math.Max(0, optimal);
                double yield = 20 + optimal * 90 + rnd.NextDouble() * 8 - 4;

                list.Add(new CropSample
                {
                    Temperature = t,
                    Rainfall = r,
                    SoilMoisture = m,
                    Ph = ph,
                    Yield = Math.Max(0, yield)
                });
            }
            return list;
        }
    }
}