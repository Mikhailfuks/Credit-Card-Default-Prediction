using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace CreditCardDefaultPrediction
{
    // Define the data class for credit card features
    public class CreditCardData
    {
        [LoadColumn(0)]
        public float LIMIT_BAL { get; set; } // Amount of the given credit (NT dollar)

        [LoadColumn(1)]
        public float SEX { get; set; } // Gender (1 = male; 2 = female)

        [LoadColumn(2)]
        public float EDUCATION { get; set; } // Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)

        [LoadColumn(3)]
        public float MARRIAGE { get; set; } // Marital status (1 = married; 2 = single; 3 = others)

        [LoadColumn(4)]
        public float AGE { get; set; } // Age in years

        [LoadColumn(5)]
        public float PAY_0 { get; set; } // Repayment status in September, 2005 (-1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; ...; 8 = payment delay for eight months; 9 = payment delay for nine months and above)

        [LoadColumn(6)]
        public float PAY_2 { get; set; } // Repayment status in August, 2005 (scale same as above)

        [LoadColumn(7)]
        public float PAY_3 { get; set; } // Repayment status in July, 2005 (scale same as above)

        [LoadColumn(8)]
        public float PAY_4 { get; set; } // Repayment status in June, 2005 (scale same as above)

        [LoadColumn(9)]
        public float PAY_5 { get; set; } // Repayment status in May, 2005 (scale same as above)

        [LoadColumn(10)]
        public float PAY_6 { get; set; } // Repayment status in April, 2005 (scale same as above)

        [LoadColumn(11)]
        public float BILL_AMT1 { get; set; } // Amount of bill statement in September, 2005 (NT dollar)

        [LoadColumn(12)]
        public float BILL_AMT2 { get; set; } // Amount of bill statement in August, 2005 (NT dollar)

        [LoadColumn(13)]
        public float BILL_AMT3 { get; set; } // Amount of bill statement in July, 2005 (NT dollar)

        [LoadColumn(14)]
        public float BILL_AMT4 { get; set; } // Amount of bill statement in June, 2005 (NT dollar)

        [LoadColumn(15)]
        public float BILL_AMT5 { get; set; } // Amount of bill statement in May, 2005 (NT dollar)

        [LoadColumn(16)]
        public float BILL_AMT6 { get; set; } // Amount of bill statement in April, 2005 (NT dollar)

        [LoadColumn(17)]


        public float PAY_AMT1 { get; set; } // Amount of previous payment in September, 2005 (NT dollar)

        [LoadColumn(18)]
        public float PAY_AMT2 { get; set; } // Amount of previous payment in August, 2005 (NT dollar)

        [LoadColumn(19)]
        public float PAY_AMT3 { get; set; } // Amount of previous payment in July, 2005 (NT dollar)

        [LoadColumn(20)]
        public float PAY_AMT4 { get; set; } // Amount of previous payment in June, 2005 (NT dollar)

        [LoadColumn(21)]
        public float PAY_AMT5 { get; set; } // Amount of previous payment in May, 2005 (NT dollar)

        [LoadColumn(22)]
        public float PAY_AMT6 { get; set; } // Amount of previous payment in April, 2005 (NT dollar)

        [LoadColumn(23), ColumnName("Label")]
        public bool DefaultPaymentNextMonth { get; set; } // Default payment (1 = yes; 0 = no)
    }

    // Class for predictions
    public class CreditCardPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool DefaultPaymentNextMonth { get; set; } // True for default, False for not default

        [ColumnName("Score")]
        public float Probability { get; set; } // Probability of the predicted outcome
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 1. Load the data
            MLContext mlContext = new MLContext();
            string dataPath = "credit_card_data.csv"; // Replace with your data file path
            IDataView dataView = mlContext.Data.LoadFromTextFile<CreditCardData>(dataPath, hasHeader: true, separatorChar: ',');

            // 2. Define the training pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                                                      "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                                                      "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                                                      "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Score", "Score"));

            // 3. Train the model
            ITransformer model = pipeline.Fit(dataView);

            // 4. Create a prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<CreditCardData, CreditCardPrediction>(model);

            // 5. Make a prediction
            CreditCardData newCustomer = new CreditCardData()
            {
                // Example customer data
                LIMIT_BAL = 100000,
                SEX = 1,
                EDUCATION = 2,
                MARRIAGE = 1,
                AGE = 35,
                //  other features
            };

            CreditCardPrediction prediction = predictionEngine.Predict(newCustomer);

            // 6. Display the prediction
            Console.WriteLine($"Predicted Default: {(prediction.DefaultPaymentNextMonth ? "Yes" : "No")}");
            Console.WriteLine($"Probability: {prediction.Probability}");

            Console.ReadKey();
        }
    }
}
