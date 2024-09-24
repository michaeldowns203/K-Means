import java.util.*;
import java.io.*;

//binning
//no data imputation
//chunks for 10-fold cross validation ARE shuffled in this class
public class GlassDriver {

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
        Collections.shuffle(dataset);

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
        String inputFile1 = "src/glass.data";
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

                // Assign the label (last column)
                labels[lineNum] = Integer.parseInt(rawData[10]);

                // Fill the data array (columns 2 to 10)
                for (int i = 1; i <= 9; i++) {
                    if (Double.parseDouble(rawData[i]) < .001) {
                        data[lineNum][i - 1] = .001;
                    }
                    else if (Double.parseDouble(rawData[i]) < .2) {
                        data[lineNum][i - 1] = .2;
                    }
                    else if (Double.parseDouble(rawData[i]) < .5) {
                        data[lineNum][i - 1] = .5;
                    }
                    else if (Double.parseDouble(rawData[i]) < 1) {
                        data[lineNum][i - 1] = 1;
                    }
                    else if (Double.parseDouble(rawData[i]) < 1.515) {
                        data[lineNum][i - 1] = 1.515;
                    }
                    else if (Double.parseDouble(rawData[i]) < 1.517) {
                        data[lineNum][i - 1] = 1.517;
                    }
                    else if (Double.parseDouble(rawData[i]) < 1.519) {
                        data[lineNum][i - 1] = 1.519;
                    }
                    else if (Double.parseDouble(rawData[i]) < 1.525) {
                        data[lineNum][i - 1] = 1.525;
                    }
                    else if (Double.parseDouble(rawData[i]) < 2) {
                        data[lineNum][i - 1] = 2;
                    }
                    else if (Double.parseDouble(rawData[i]) < 2.75) {
                        data[lineNum][i - 1] = 2.75;
                    }
                    else if (Double.parseDouble(rawData[i]) < 3.5) {
                        data[lineNum][i - 1] = 3.5;
                    }
                    else if (Double.parseDouble(rawData[i]) < 3.75) {
                        data[lineNum][i - 1] = 3.75;
                    }
                    else if (Double.parseDouble(rawData[i]) < 4.5) {
                        data[lineNum][i - 1] = 4.5;
                    }
                    else if (Double.parseDouble(rawData[i]) < 8) {
                        data[lineNum][i - 1] = 8;
                    }
                    else if (Double.parseDouble(rawData[i]) < 9) {
                        data[lineNum][i - 1] = 9;
                    }
                    else if (Double.parseDouble(rawData[i]) < 10) {
                        data[lineNum][i - 1] = 10;
                    }
                    else if (Double.parseDouble(rawData[i]) < 13) {
                        data[lineNum][i - 1] = 13;
                    }
                    else if (Double.parseDouble(rawData[i]) < 14) {
                        data[lineNum][i - 1] = 14;
                    }
                    else if (Double.parseDouble(rawData[i]) < 15) {
                        data[lineNum][i - 1] = 15;
                    }
                    else if (Double.parseDouble(rawData[i]) < 70) {
                        data[lineNum][i - 1] = 70;
                    }
                    else if (Double.parseDouble(rawData[i]) < 72) {
                        data[lineNum][i - 1] = 72;
                    }
                    else if (Double.parseDouble(rawData[i]) < 72.5) {
                        data[lineNum][i - 1] = 72.5;
                    }
                    else if (Double.parseDouble(rawData[i]) < 73) {
                        data[lineNum][i - 1] = 73;
                    }
                    else if (Double.parseDouble(rawData[i]) < 73.5) {
                        data[lineNum][i - 1] = 73.5;
                    }
                    else if (Double.parseDouble(rawData[i]) < 75) {
                        data[lineNum][i - 1] = 75;
                    }
                    else if (Double.parseDouble(rawData[i]) < 76) {
                        data[lineNum][i - 1] = 76;
                    }
                }

                lineNum++;
            }

            // print the data to verify
            for (int i = 0; i < lineCount; i++) {
                System.out.print("Label: " + labels[i] + " Data: ");
                for (int j = 0; j < 9; j++) {
                    System.out.print(data[i][j] + " ");
                }
                System.out.println();
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
                    Object[] testInstance = new Object [testData[j].length - 1];
                    System.arraycopy(testData[j], 0, testInstance, 0, testData[j].length - 1);

                    Object predicted = classifier.classify(testInstance);
                    Object actual = testLabels[j];

                    // Print the test data, predicted label, and actual label
                    System.out.print("Test Data: [ ");
                    for (Object feature : testInstance) {
                        System.out.print(feature + " ");
                    }
                    System.out.println("] Predicted: " + predicted + " Actual: " + actual);


                    if (predicted.equals(testLabels[j])) {
                        correctPredictions++;
                    }
                    // Get true positives, false positives, and false negatives
                    if (predicted.equals(1)) {
                        if (actual.equals(1)) {
                            truePositives++;
                        } else {
                            falsePositives++;
                        }
                    } else if (actual.equals(1)) {
                        falseNegatives++;
                    }
                }
                // Calculate precision and recall
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
                // Print loss info
                System.out.println("Number of correct predictions: " + correctPredictions);
                System.out.println("Number of test instances: " + testData.length);
                System.out.println("Fold " + (i + 1) + " Accuracy: " + accuracy);
                System.out.println("Fold " + (i + 1) + " 0/1 loss: " + loss01);
                System.out.println("Precision for class 1 (fold " + (i + 1) + "): " + precision);
                System.out.println("Recall for class 1 (fold " + (i + 1) + "): " + recall);
                System.out.println("F1 Score for class 1 (fold " + (i + 1) + "): " + f1Score);

            }

            // Average accuracy across all 10 folds
            double averageAccuracy = totalAccuracy / 10;
            double average01loss = total01loss / 10;
            double averagePrecision = totalPrecision / 10;
            double averageRecall = totalRecall / 10;
            double averageF1 = totalF1 / 10;
            System.out.println("Average Accuracy: " + averageAccuracy);
            System.out.println("Average 0/1 Loss: " + average01loss);
            System.out.println("Average Precision for class 1: " + averagePrecision);
            System.out.println("Average Recall for class 1: " + averageRecall);
            System.out.println("Average F1 for class 1: " + averageF1);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}







