import java.util.*;

// Define a class to represent a data point
class DataPoint {
    double[] features;  // Feature values
    int label;          // Class label

    public DataPoint(double[] features, int label) {
        this.features = features;
        this.label = label;
    }
}
