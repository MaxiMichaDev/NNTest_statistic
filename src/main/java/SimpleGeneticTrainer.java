import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class SimpleGeneticTrainer {
    public static void main(String[] args) {
        INDArray dummyWeights = Nd4j.arange(32).div(100); //INDArray von 0 bis 31, geteilt durch 100
        List<Double> dummyWeightList = Arrays.asList(ArrayUtils.toObject(dummyWeights.toDoubleVector())); //zu Liste konvertieren (Java sehr redundant hier xD)
        System.out.println("dummyWeightList = " + dummyWeightList);
        netTest test = new netTest();
        double fitness = test.fitness(dummyWeightList);
        System.out.println("fitness = " + fitness);
    }

}
