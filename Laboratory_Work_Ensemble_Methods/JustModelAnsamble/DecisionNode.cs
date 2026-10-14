using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JustModelAnsamble
{
    public class DecisionNode
    {
        public string Feature { get; set; }       // имя признака, напр. "Temperature"
        public double Threshold { get; set; }     // порог
        public DecisionNode Left { get; set; }    // ветвь "<= порога"
        public DecisionNode Right { get; set; }   // ветвь "> порога"
        public double? Prediction { get; set; }   // если лист — значение прогноза

        public bool IsLeaf => Prediction.HasValue;
    }
}
