import org.deeplearning4j.nn.weights.WeightInitXavier;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;

public class CustomWeightInitializer extends WeightInitXavier {

    int counter = 0;
    int numLayers;
    INDArray weights;

    public CustomWeightInitializer(INDArray weights, int numLayers) {
        this.weights = weights;
        this.numLayers = numLayers;
    }

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
//        System.out.println("fanIn = " + fanIn + ", fanOut = " + fanOut + ", shape = " + Arrays.toString(shape) + ", order = " + order + ", paramView = " + paramView);
        INDArray result = weights.get(NDArrayIndex.point(counter));
//        System.out.println("-----");
//        System.out.println("counter = " + counter);
//        System.out.println("result = " + result);
        counter++;
        if (counter == numLayers) {
            counter = 0;
        }
        return result.reshape(order, shape);
    }
}
