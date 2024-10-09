import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class BreastDriver2 {

    static class DataPoint {
        int id;
        double[] features;
        int label; // 4 for cancer, 2 for no cancer

        DataPoint(int id, double[] features, int label) {
            this.id = id;
            this.features = features;
            this.label = label;
        }
    }

    public static void main(String[] args) {
        String filePath = "src/breast-cancer-wisconsin.txt";
        List<DataPoint> dataset = new ArrayList<>();

        // Step 1: Read the data from the file
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                int id = Integer.parseInt(values[0]);
                double[] features = new double[9];

                for (int i = 0; i < 9; i++) {
                    if (values[i + 1].equals("?")) {
                        features[i] = new Random().nextInt(10) + 1; // Replace '?' with a random number between 1 and 10
                    } else {
                        features[i] = Double.parseDouble(values[i + 1]);
                    }
                }
                int label = Integer.parseInt(values[10]);
                dataset.add(new DataPoint(id, features, label));
            }
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        // Step 3: Count classes
        int cancerCount = 0;
        int noCancerCount = 0;
        for (DataPoint dp : dataset) {
            if (dp.label == 4) {
                cancerCount++;
            } else {
                noCancerCount++;
            }
        }

        // Step 4: Create training and test sets
        int totalDataPoints = dataset.size();
        int trainSize = (int) (totalDataPoints * 0.90); // 90% for training
        int testSize = totalDataPoints - trainSize;     // 10% for testing

        List<DataPoint>[] trainingSets = new List[10];
        for (int i = 0; i < 10; i++) {
            trainingSets[i] = new ArrayList<>();
        }
        List<DataPoint> tuningSet = new ArrayList<>();

        // Stratified sampling
        List<DataPoint> cancerSamples = new ArrayList<>();
        List<DataPoint> noCancerSamples = new ArrayList<>();

        for (DataPoint dp : dataset) {
            if (dp.label == 4) {
                cancerSamples.add(dp);
            } else {
                noCancerSamples.add(dp);
            }
        }

        // Fill the tuning set with 10% of the dataset
        int tuningSize = (int) (totalDataPoints * 0.10);
        Collections.shuffle(cancerSamples);
        Collections.shuffle(noCancerSamples);

        // Add samples for tuning
        int tuningCancerCount = (int) Math.round((double) cancerCount / totalDataPoints * tuningSize);
        int tuningNoCancerCount = tuningSize - tuningCancerCount;

        for (int i = 0; i < tuningCancerCount; i++) {
            tuningSet.add(cancerSamples.remove(0));
        }
        for (int i = 0; i < tuningNoCancerCount; i++) {
            tuningSet.add(noCancerSamples.remove(0));
        }

        // Prepare the remaining data for the training sets
        List<DataPoint> remainingData = new ArrayList<>(cancerSamples);
        remainingData.addAll(noCancerSamples);
        Collections.shuffle(remainingData);

        // Distribute remaining data into training sets
        int[] currentCancerCount = new int[10];
        int[] currentNoCancerCount = new int[10];

        for (DataPoint dp : remainingData) {
            int minIndex = findMinIndex(currentCancerCount, currentNoCancerCount, dp.label);
            trainingSets[minIndex].add(dp);
            if (dp.label == 4) {
                currentCancerCount[minIndex]++;
            } else {
                currentNoCancerCount[minIndex]++;
            }
        }

        // Step 5: Output results
        for (int i = 0; i < 10; i++) {
            System.out.println("Training Set F" + (i + 1) + " Size: " + trainingSets[i].size());
        }
        System.out.println("Tuning Set Size: " + tuningSet.size());
    }

    private static int findMinIndex(int[] cancerCount, int[] noCancerCount, int label) {
        int minIndex = -1;
        int minCount = Integer.MAX_VALUE;

        for (int i = 0; i < cancerCount.length; i++) {
            int currentCount = label == 4 ? cancerCount[i] : noCancerCount[i];
            if (currentCount < minCount) {
                minCount = currentCount;
                minIndex = i;
            }
        }
        return minIndex;
    }
}