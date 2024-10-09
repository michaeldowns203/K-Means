import java.util.*;
import java.io.*;

// No binning
// No data imputation
// Chunks for 10-fold cross-validation ARE shuffled in this class
public class ComputerDriver {
    // Method to scale features using Min-Max Scaling
    public static List<List<Double>> minMaxScale(List<List<Double>> data) {
        int numFeatures = data.get(0).size();  // Process all columns as features
        List<Double> minValues = new ArrayList<>(Collections.nCopies(numFeatures, Double.MAX_VALUE));
        List<Double> maxValues = new ArrayList<>(Collections.nCopies(numFeatures, Double.MIN_VALUE));

        // Find the min and max values for each feature
        for (List<Double> row : data) {
            for (int i = 0; i < numFeatures; i++) {
                double value = row.get(i);
                if (value < minValues.get(i)) minValues.set(i, value);
                if (value > maxValues.get(i)) maxValues.set(i, value);
            }
        }

        // Scale the dataset based on min and max values
        List<List<Double>> scaledData = new ArrayList<>();
        for (List<Double> row : data) {
            List<Double> scaledRow = new ArrayList<>();
            for (int i = 0; i < numFeatures; i++) {
                double value = row.get(i);
                double scaledValue = (value - minValues.get(i)) / (maxValues.get(i) - minValues.get(i));
                scaledRow.add(scaledValue);
            }
            scaledData.add(scaledRow);  // Only include scaled features
        }

        return scaledData;
    }



    // Method to split the dataset into stratified chunks
    public static List<List<List<Double>>> splitIntoStratifiedChunks(List<List<Double>> data, List<String> labels, int numChunks) {
        // Combine data and labels into one dataset
        List<List<Double>> dataset = new ArrayList<>();
        for (int i = 0; i < data.size(); i++) {
            List<Double> combined = new ArrayList<>(data.get(i));
            combined.add(Double.parseDouble(labels.get(i))); // Add label as the last element
            dataset.add(combined);
        }

        // Sort dataset by the last element (target value)
        dataset.sort(Comparator.comparingDouble(a -> a.get(a.size() - 1)));

        // Split into chunks
        int chunkSize = dataset.size() / numChunks;
        List<List<List<Double>>> chunks = new ArrayList<>();

        for (int i = 0; i < numChunks; i++) {
            List<List<Double>> chunk = new ArrayList<>();
            for (int j = 0; j < chunkSize; j++) {
                // Ensure that we don't go out of bounds
                if (i * chunkSize + j < dataset.size()) {
                    chunk.add(dataset.get(i * chunkSize + j));
                }
            }
            chunks.add(chunk);
        }

        return chunks;
    }

    public static void main(String[] args) throws IOException {
        String inputFile1 = "src/machine.data";
        try {
            FileInputStream fis = new FileInputStream(inputFile1);
            InputStreamReader isr = new InputStreamReader(fis);
            BufferedReader stdin = new BufferedReader(isr);

            // First, count the number of lines to determine the size of the arrays
            int lineCount = 0;
            while (stdin.readLine() != null) {
                lineCount++;
            }

            System.out.println(lineCount);

            // Reset the reader to the beginning of the file
            stdin.close();
            fis = new FileInputStream(inputFile1);
            isr = new InputStreamReader(fis);
            stdin = new BufferedReader(isr);

            // Initialize the lists
            List<String> labels = new ArrayList<>();
            List<List<Double>> data = new ArrayList<>();

            String line;

            // Read the file and fill the lists
            while ((line = stdin.readLine()) != null) {
                String[] rawData = line.split(",");

                // Assign the label (last column)
                labels.add(rawData[8]);

                // Create a new row for the features
                List<Double> row = new ArrayList<>();
                for (int i = 0; i < 5; i++) {
                    row.add(Double.parseDouble(rawData[i + 2]));
                }
                data.add(row);
            }

            stdin.close();

            // Split into 10 chunks
            List<List<List<Double>>> chunks = splitIntoStratifiedChunks(data, labels, 10);

            // Loss instance variables
            double totalAccuracy = 0;
            double total01loss = 0;

            // Perform 10-fold cross-validation
            for (int i = 0; i < 10; i++) {
                // Create training and testing sets
                List<List<Double>> trainingData = new ArrayList<>();
                List<String> trainingLabels = new ArrayList<>();

                List<List<Double>> testData = chunks.get(i);  // Use List<List<Double>> for test data
                List<Double> testLabels = new ArrayList<>();  // Create a list for test labels

                // Extract labels from the test data (last column)
                for (List<Double> row : testData) {
                    testLabels.add(row.get(row.size() - 1));  // Last element is the label
                }

                // Combine the other 9 chunks into the training set
                for (int j = 0; j < 10; j++) {
                    if (j != i) {
                        for (List<Double> row : chunks.get(j)) {
                            trainingLabels.add(String.valueOf(row.get(row.size() - 1)));  // Last element is the label
                            trainingData.add(new ArrayList<>(row.subList(0, row.size() - 1)));  // All but the last element
                        }
                    }
                }

                List<List<Double>> scaledTrainingData = minMaxScale(trainingData);
                List<List<Double>> scaledTestData = minMaxScale(testData);

                // Initialize and train the k-NN model
                int k = 5; // You can tune this value later
                KNN knn = new KNN(k, .03, 10); // Set sigma and error threshold as needed
                knn.fit(scaledTrainingData, trainingLabels);

                // Test the classifier
                int correctPredictions = 0;
                for (int j = 0; j < testData.size(); j++) {
                    List<Double> testInstance = new ArrayList<>(scaledTestData.get(j).subList(0, scaledTestData.get(j).size() - 1));

                    double predicted = knn.predictValue(testInstance);
                    double actual = testLabels.get(j);

                    // Print the test data, predicted label, and actual label
                    System.out.print("Test Data: [ ");
                    for (Double feature : testInstance) {
                        System.out.print(feature + " ");
                    }
                    System.out.println("] Predicted: " + predicted + " Actual: " + actual);

                    if (knn.isPredictionCorrect(actual, predicted)) {
                        correctPredictions++;
                    }
                }

                // Calculate accuracy for this fold
                double accuracy = (double) correctPredictions / testData.size();
                totalAccuracy += accuracy;

                // Calculate 0/1 loss
                double loss01 = 1.0 - (double) correctPredictions / testData.size();
                total01loss += loss01;

                // Print loss info
                System.out.println("Number of correct predictions: " + correctPredictions);
                System.out.println("Number of test instances: " + testData.size());
                System.out.println("Fold " + (i + 1) + " Accuracy: " + accuracy);
                System.out.println("Fold " + (i + 1) + " 0/1 loss: " + loss01);
            }

            // Average accuracy across all 10 folds
            double averageAccuracy = totalAccuracy / 10;
            double average01loss = total01loss / 10;
            System.out.println("Average Accuracy: " + averageAccuracy);
            System.out.println("Average 0/1 Loss: " + average01loss);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}


