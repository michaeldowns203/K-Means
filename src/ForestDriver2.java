import java.util.*;
import java.io.*;

//normal 10 fold
public class ForestDriver2 {
    // Method to scale labels using Min-Max Scaling
    public static List<Double> minMaxScaleLabels(List<Double> labels) {
        // Scale the labels using Min-Max scaling
        double minLabel = Collections.min(labels);
        double maxLabel = Collections.max(labels);
        List<Double> scaledLabels = new ArrayList<>();
        for (double label : labels) {
            double scaledLabel = (label - minLabel) / (maxLabel - minLabel);
            scaledLabels.add(scaledLabel);
        }

        return scaledLabels;  // Return the scaled labels
    }


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

    public static List<List<Object>> extractTenPercent(List<List<Object>> dataset) {
        // Step 1: Sort the dataset based on the response value (last element)
        dataset.sort(Comparator.comparingDouble(row -> (Double) row.get(row.size() - 1)));

        // Step 2: Divide the sorted dataset into groups of ten consecutive examples
        List<List<List<Object>>> groups = new ArrayList<>();
        int groupSize = 10;

        for (int i = 0; i < dataset.size(); i += groupSize) {
            int end = Math.min(i + groupSize, dataset.size());
            groups.add(new ArrayList<>(dataset.subList(i, end)));
        }

        // Step 3: Create 10 datasets for stratified sampling
        List<List<Object>> tuningData = new ArrayList<>();
        int tuningSize = (int) Math.ceil(dataset.size() * 0.1);

        // Draw every item for each fold
        for (int fold = 0; fold < groupSize; fold++) {
            for (int i = fold; i < groups.size(); i += groupSize) {
                // Add the item to the tuning data, ensuring we do not exceed the tuning size
                if (i < groups.size() && fold < groups.get(i).size()) {
                    tuningData.add(groups.get(i).get(fold));
                }
                // Stop if we reached the required tuning size
                if (tuningData.size() >= tuningSize) {
                    break;
                }
            }
            // Break if we already have enough tuning data
            if (tuningData.size() >= tuningSize) {
                break;
            }
        }

        // Ensure we do not exceed the required tuning size
        return tuningData.subList(0, Math.min(tuningData.size(), tuningSize));
    }

    public static List<List<List<Object>>> splitIntoStratifiedChunks(List<List<Object>> dataset, int numChunks) {
        // Sort the dataset based on the response value (the last element in each list)
        dataset.sort(Comparator.comparingDouble(row -> (Double) row.get(row.size() - 1)));

        // Extract the tuning data (10%)
        //List<List<Object>> tuningData = extractTenPercent(dataset);

        // Remove the tuning data from the original dataset
        //List<List<Object>> remainingData = new ArrayList<>(dataset.subList(tuningData.size(), dataset.size()));

        // Create the chunks for each fold
        List<List<List<Object>>> chunks = new ArrayList<>();
        for (int i = 0; i < numChunks; i++) {
            chunks.add(new ArrayList<>());
        }

        // Break the remaining dataset into groups of 10 consecutive examples
        int groupSize = 10;
        List<List<Object>> group = new ArrayList<>();

        // Distribute each item into the corresponding chunk
        for (int i = 0; i < dataset.size(); i++) {
            group.add(dataset.get(i));  // Add item to the group

            // Once the group reaches the group size, distribute it across the chunks
            if (group.size() == groupSize || i == dataset.size() - 1) {
                for (int j = 0; j < group.size(); j++) {
                    int chunkIndex = j % numChunks;
                    chunks.get(chunkIndex).add(group.get(j));  // Distribute across the chunks
                }
                group.clear();  // Reset the group for the next batch
            }
        }

        return chunks;
    }



    public static void main(String[] args) throws IOException {
        String inputFile1 = "src/forestfires.data";
        try {
            FileInputStream fis = new FileInputStream(inputFile1);
            InputStreamReader isr = new InputStreamReader(fis);
            BufferedReader stdin = new BufferedReader(isr);

            // First, count the number of lines to determine the size of the lists
            int lineCount = 0;
            while (stdin.readLine() != null) {
                lineCount++;
            }
            // Reset the reader to the beginning of the file
            stdin.close();
            fis = new FileInputStream(inputFile1);
            isr = new InputStreamReader(fis);
            stdin = new BufferedReader(isr);

            // Initialize the lists
            List<List<Object>> dataset = new ArrayList<>();
            List<Object> labels = new ArrayList<>();

            String line;
            //instance variable; flag to skip the first line
            boolean firstLine = true;
            int lineNum = 0;

            // Read the file and fill the dataset
            while ((line = stdin.readLine()) != null) {
                //skips the first line as it includes headers not data
                if (firstLine) {
                    firstLine = false;
                    continue;
                }
                String[] rawData = line.split(",");
                List<Object> row = new ArrayList<>();

                // Assign the label (last column)
                labels.add(Double.parseDouble(rawData[12]));

                // Fill the data row
                for (int i = 0; i < 7; i++) {
                    row.add(Double.parseDouble(rawData[i + 4]));
                }
                row.add(labels.get(lineNum)); // Add the label to the row
                dataset.add(row);
                lineNum++;
            }

            stdin.close();

            // Extract stratified tuning data (10%)
            //List<List<Object>> testSet = extractTenPercent(dataset);

            // Split the remaining dataset into stratified chunks
            List<List<List<Object>>> chunks = splitIntoStratifiedChunks(dataset, 10);

            // Loss instance variables
            double totalMSE = 0;

            // Perform stratified 10-fold cross-validation
            for (int i = 0; i < 10; i++) {
                // Create training and testing sets
                List<List<Double>> trainingData = new ArrayList<>();
                List<String> trainingLabels = new ArrayList<>();

                List<Double> predictedList = new ArrayList<>();
                List<Double> actualList = new ArrayList<>();

                List<List<Object>> testData = chunks.get(i);
                List<String> testLabels = new ArrayList<>();
                for (List<Object> row : testData) {
                    testLabels.add(String.valueOf(row.get(row.size() - 1))); // Last column is label
                }

                // Combine the other 9 chunks into the training set
                for (int j = 0; j < 10; j++) {
                    if (j != i) {
                        for (List<Object> row : chunks.get(j)) {
                            trainingLabels.add(String.valueOf(row.get(row.size() - 1)));  // Last column is label
                            List<Double> features = new ArrayList<>();
                            for (int k = 0; k < row.size() - 1; k++) {
                                features.add((Double) row.get(k));
                            }
                            trainingData.add(features);
                        }
                    }
                }

                List<List<Double>> scaledTrainingData = minMaxScale(trainingData);
                List<List<Double>> doubleTestData = new ArrayList<>();
                for (List<Object> innerList : testData) {
                    List<Double> innerDoubleList = new ArrayList<>();
                    for (Object obj : innerList) {
                        innerDoubleList.add((Double) obj); // Cast to Double
                    }
                    doubleTestData.add(innerDoubleList);
                }
                List<List<Double>> scaledTestData = minMaxScale(doubleTestData);

                List<Double> doubleTrainingLabels = new ArrayList<>();
                for (String str : trainingLabels) {
                    doubleTrainingLabels.add(Double.parseDouble(str));
                }
                List<Double> scaledTrainingLabels = minMaxScaleLabels(doubleTrainingLabels);
                List<Double> doubleTestLabels = new ArrayList<>();
                for (String str : testLabels) {
                    doubleTestLabels.add(Double.parseDouble(str));
                }
                List<Double> scaledTestLabels = minMaxScaleLabels(doubleTestLabels);

                List<String> stringScaledTrainingLabels = new ArrayList<>();
                for (Double d : scaledTrainingLabels) {
                    stringScaledTrainingLabels.add(String.valueOf(d)); // or d.toString();
                }

                // Initialize and train the k-NN model
                int k = 10; // You can tune this value later
                KNN knn = new KNN(k, 1000, 25);
                knn.fit(trainingData, trainingLabels);
                //knn.editR();
                knn.kMeansAndReduceRegression(388, 1000);

                // Test the classifier
                for (int j = 0; j < testData.size(); j++) {
                    List<Double> testInstance = new ArrayList<>();
                    for (int l = 0; l < testData.get(j).size() - 1; l++) {
                        testInstance.add((Double) testData.get(j).get(l));
                    }

                    double predicted = knn.predictValue(testInstance);
                    double actual = Double.parseDouble(testLabels.get(j));
                    predictedList.add(predicted);
                    actualList.add(actual);

                    // Print the test data, predicted label, and actual label
                    System.out.print("Test Data: [ ");
                    for (Double feature : testInstance) {
                        System.out.print(feature + " ");
                    }
                    System.out.println("] Predicted: " + predicted + " Actual: " + actual);
                }

                double mse = knn.calculateMSE(actualList, predictedList);
                totalMSE += mse;

                // Print loss info
                System.out.println("Fold " + (i + 1) + " Mean Squared Error: " + mse);
            }

            // Average mse across all 10 folds
            double averageMSE = totalMSE / 10;
            System.out.println("Average Mean Squared Error: " + averageMSE);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}

