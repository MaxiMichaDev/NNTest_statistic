import org.deeplearning4j.nn.weights.WeightInitXavier;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CustomWeightInitializer extends WeightInitXavier {

    int counterLayers = 0;
    int counterStartIndex = 0;
    int numLayers;
    int[][] netShape;
    List<Double> weights;

    public CustomWeightInitializer(/*INDArray weightsL1, INDArray weightsL2, *//*INDArray weightsL3,*//* int numLayers*/ int[][] netShape, List<Double> weights) {
        this.weights = weights;
        this.netShape = netShape;
        this.numLayers = Array.getLength(netShape[0]);
    }

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
//        System.out.println("fanIn = " + fanIn + ", fanOut = " + fanOut + ", shape = " + Arrays.toString(shape) + ", order = " + order + ", paramView = " + paramView);

            INDArray result = Nd4j.create(weights.subList(counterStartIndex, counterStartIndex + netShape[counterLayers][0] * netShape[counterLayers][1]));
            counterStartIndex += netShape[counterLayers][0] * netShape[counterLayers][1];

//        System.out.println("-----");
//        System.out.println("Layer = " + counter);
//        System.out.println("result = " + result);
        counterLayers++;
        if (counterLayers == numLayers) {
            counterLayers = 0;
        }
        return result.reshape(order, shape);
    }
}
