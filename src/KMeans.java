import java.util.*;

public class KMeans {
    private List<DataPoint> dataset;   // The training dataset
    private int kc;                    // Number of clusters in k-means

    public KMeans(List<DataPoint> dataset) {
        this.dataset = dataset;
        this.kc = (int) Math.sqrt(dataset.size()); // Initial estimate for clusters (or adjust as needed)
    }

    // Method to run k-means clustering
    public List<DataPoint> kMeans() {
        // Initialize centroids randomly
        List<DataPoint> centroids = initializeCentroids(kc);
        List<DataPoint> previousCentroids;

        do {
            previousCentroids = new ArrayList<>(centroids);
            // Assign points to the nearest centroid
            List<List<DataPoint>> clusters = assignClusters(centroids);
            // Update centroids
            centroids = updateCentroids(clusters);
        } while (!centroidsEqual(previousCentroids, centroids));

        return centroids; // Return final centroids
    }

    // Method to initialize centroids randomly
    private List<DataPoint> initializeCentroids(int kc) {
        Collections.shuffle(dataset); // Shuffle dataset for randomness
        List<DataPoint> centroids = new ArrayList<>();
        for (int i = 0; i < kc; i++) {
            centroids.add(dataset.get(i)); // Take first kc points as centroids
        }
        return centroids;
    }

    // Method to assign each point to the nearest centroid
    private List<List<DataPoint>> assignClusters(List<DataPoint> centroids) {
        List<List<DataPoint>> clusters = new ArrayList<>(Collections.nCopies(centroids.size(), new ArrayList<>()));

        for (DataPoint point : dataset) {
            int closestCentroidIndex = 0;
            double minDistance = Double.MAX_VALUE;

            for (int i = 0; i < centroids.size(); i++) {
                double distance = euclideanDistance(point.features, centroids.get(i).features);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCentroidIndex = i;
                }
            }

            clusters.get(closestCentroidIndex).add(point); // Assign point to the closest centroid
        }

        return clusters;
    }

    // Method to update centroids based on assigned clusters
    private List<DataPoint> updateCentroids(List<List<DataPoint>> clusters) {
        List<DataPoint> newCentroids = new ArrayList<>();

        for (List<DataPoint> cluster : clusters) {
            if (cluster.isEmpty()) {
                continue; // Skip empty clusters
            }

            double[] newCentroidFeatures = new double[cluster.get(0).features.length];
            for (DataPoint point : cluster) {
                for (int i = 0; i < point.features.length; i++) {
                    newCentroidFeatures[i] += point.features[i];
                }
            }

            for (int i = 0; i < newCentroidFeatures.length; i++) {
                newCentroidFeatures[i] /= cluster.size(); // Calculate mean for each feature
            }

            newCentroids.add(new DataPoint(newCentroidFeatures, -1)); // Assign a dummy label
        }

        return newCentroids;
    }

    // Method to check if centroids have changed
    private boolean centroidsEqual(List<DataPoint> centroids1, List<DataPoint> centroids2) {
        for (int i = 0; i < centroids1.size(); i++) {
            if (!Arrays.equals(centroids1.get(i).features, centroids2.get(i).features)) {
                return false;
            }
        }
        return true;
    }

    // Calculate the Euclidean distance between two points
    private double euclideanDistance(double[] features1, double[] features2) {
        double sum = 0;
        for (int i = 0; i < features1.length; i++) {
            sum += Math.pow(features1[i] - features2[i], 2);
        }
        return Math.sqrt(sum);
    }

    // Method to get centroids for KNN
    public List<DataPoint> getCentroids() {
        return kMeans(); // Calculate and return centroids
    }
}
