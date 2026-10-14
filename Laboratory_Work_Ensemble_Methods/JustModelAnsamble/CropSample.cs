using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JustModelAnsamble
{
    public class CropSample
    {
        public double Temperature { get; set; }   // °C
        public double Rainfall { get; set; }      // мм
        public double SoilMoisture { get; set; }  // %
        public double Ph { get; set; }            // pH
        public double Yield { get; set; }         // ц/га — целевая переменная
    }
}
