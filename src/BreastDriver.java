import java.util.*;
import java.io.*;

//no binning
//data imputation - we replaced each instance of "?" with a random value 1-10
//chunks for 10-fold cross validation are NOT shuffled in this class (but they were shuffled to get our experimental data)
public class BreastDriver {

    // Split the dataset into 10 chunks
    public static List<Object[][]> splitIntoChunks(Object[][] data, Object[] labels, int numChunks) {
        List<Object[]> dataset = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            Object[] combined = new Object[data[i].length + 1];
            System.arraycopy(data[i], 0, combined, 0, data[i].length);
            combined[combined.length - 1] = labels[i];
            dataset.add(combined);
        }

        // Shuffle the dataset to ensure randomness
        //Collections.shuffle(dataset);

        // Split into chunks
        int chunkSize = dataset.size() / numChunks;
        List<Object[][]> chunks = new ArrayList<>();

        for (int i = 0; i < numChunks; i++) {
            Object[][] chunk = new Object[chunkSize][];
            for (int j = 0; j < chunkSize; j++) {
                chunk[j] = dataset.get(i * chunkSize + j);
            }
            chunks.add(chunk);
        }

        return chunks;
    }

    public static void main(String[] args) throws IOException {
        // Assume data and labels are loaded as in your previous driver code
        String inputFile1 = "src/breast-cancer-wisconsin.data";
        try {
            FileInputStream fis = new FileInputStream(inputFile1);
            InputStreamReader isr = new InputStreamReader(fis);
            BufferedReader stdin = new BufferedReader(isr);

            // First, count the number of lines to determine the size of the arrays
            int lineCount = 0;
            while (stdin.readLine() != null) {
                lineCount++;
            }

            // Reset the reader to the beginning of the file
            stdin.close();
            fis = new FileInputStream(inputFile1);
            isr = new InputStreamReader(fis);
            stdin = new BufferedReader(isr);

            // Initialize the arrays with the known size
            Object[] labels = new Object[lineCount];
            Object[][] data = new Object[lineCount][9]; // Assuming 9 attributes (from column 2 to 10)

            String line;
            int lineNum = 0;

            // Read the file and fill the arrays
            while ((line = stdin.readLine()) != null) {
                String[] rawData = line.split(",");

                // Assign the label (first column)
                labels[lineNum] = Integer.parseInt(rawData[10]);

                // Fill the data array (columns 2 to 10)
                for (int i = 1; i <= 9; i++) {
                    if (rawData[i].equals("?")) {
                        data[lineNum][i - 1] = (int) (Math.random() * 10) + 1; // Handle missing values
                    } else {
                        data[lineNum][i - 1] = Integer.parseInt(rawData[i]);
                    }
                }

                lineNum++;
            }

            stdin.close();

            // Split into 10 chunks
            List<Object[][]> chunks = splitIntoChunks(data, labels, 10);

            // Loss instance variables
            double totalAccuracy = 0;
            double totalPrecision = 0;
            double totalRecall = 0;
            double totalF1 = 0;
            double total01loss = 0;

            // Perform 10-fold cross-validation
            for (int i = 0; i < 10; i++) {
                // Create training and testing sets
                List<Object[]> trainingData = new ArrayList<>();
                List<Object> trainingLabels = new ArrayList<>();

                Object[][] testData = chunks.get(i);
                Object[] testLabels = new Object[testData.length];
                for (int j = 0; j < testData.length; j++) {
                    testLabels[j] = testData[j][testData[j].length - 1]; // Last column is label
                }

                // Combine the other 9 chunks into the training set
                for (int j = 0; j < 10; j++) {
                    if (j != i) {
                        for (Object[] row : chunks.get(j)) {
                            trainingLabels.add(row[row.length - 1]);  // Last column is label
                            Object[] features = new Object[row.length - 1];
                            System.arraycopy(row, 0, features, 0, row.length - 1);
                            trainingData.add(features);
                        }
                    }
                }

                // Convert training data to array form
                Object[][] trainingArray = new Object[trainingData.size()][];
                trainingData.toArray(trainingArray);
                Object[] trainingLabelsArray = trainingLabels.toArray(new Object[0]);

                // Initialize and train the k-NN model
                int k = 3; // You can tune this value later
                KNN knn = new KNN(k);
                knn.fit(trainingData, trainingLabels);

                // Test the classifier
                int correctPredictions = 0;
                int truePositives = 0;
                int falsePositives = 0;
                int falseNegatives = 0;
                for (int j = 0; j < testData.length; j++) {
                    Object[] testInstance = new Object[testData[j].length - 1];
                    System.arraycopy(testData[j], 0, testInstance, 0, testData[j].length - 1);

                    Object predicted = classifier.classify(testInstance);
                    Object actual = testLabels[j];

                    if (i == 9) {
                        // Print the test data, predicted label, and actual label
                        System.out.print("Test Data: [ ");
                        for (Object feature : testInstance) {
                            System.out.print(feature + " ");
                        }
                        System.out.println("] Predicted: " + predicted + " Actual: " + actual);
                    }

                    if (predicted.equals(testLabels[j])) {
                        correctPredictions++;
                    }
                    // Check if the predicted class is 4 (positive class)
                    if (predicted.equals(4)) {
                        if (actual.equals(4)) {
                            truePositives++;  // Correctly predicted class 4 (True Positive)
                        } else {
                            falsePositives++;  // Incorrectly predicted class 4 (False Positive)
                        }
                    } else if (actual.equals(4)) {
                        falseNegatives++;  // Incorrectly predicted something else, but actual is class 4 (False Negative)
                    }
                }
                // Calculate precision and recall for class 4
                double precision = truePositives / (double) (truePositives + falsePositives);
                double recall = truePositives / (double) (truePositives + falseNegatives);
                totalPrecision += precision;
                totalRecall += recall;

                double f1Score = 2 * (precision * recall) / (precision + recall);
                totalF1 += f1Score;
                // Calculate accuracy for this fold
                double accuracy = (double) correctPredictions / testData.length;
                totalAccuracy += accuracy;
                // Calculate 0/1 loss
                double loss01 = 1.0 - (double) correctPredictions / testData.length;
                total01loss += loss01;

                if (i == 9) {
                    // Print loss info
                    System.out.println("Number of correct predictions: " + correctPredictions);
                    System.out.println("Number of test instances: " + testData.length);
                    System.out.println("Fold " + (i + 1) + " Accuracy: " + accuracy);
                    System.out.println("Fold " + (i + 1) + " 0/1 loss: " + loss01);
                    System.out.println("Precision for class 4 (fold " + (i + 1) + "): " + precision);
                    System.out.println("Recall for class 4 (fold " + (i + 1) + "): " + recall);
                    System.out.println("F1 Score for class 4 (fold " + (i + 1) + "): " + f1Score);
                }
            }

            // Average accuracy across all 10 folds
            double averageAccuracy = totalAccuracy / 10;
            double average01loss = total01loss / 10;
            double averagePrecision = totalPrecision / 10;
            double averageRecall = totalRecall / 10;
            double averageF1 = totalF1 / 10;
            //System.out.println("Average Accuracy: " + averageAccuracy);
            //System.out.println("Average 0/1 Loss: " + average01loss);
            //System.out.println("Average Precision for class 4: " + averagePrecision);
            //System.out.println("Average Recall for class 4: " + averageRecall);
            //System.out.println("Average F1 for class 4: " + averageF1);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

