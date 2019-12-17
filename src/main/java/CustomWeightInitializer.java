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

    int layerCount;
    ArrayList<INDArray> weightINDArrays;

    public CustomWeightInitializer(ArrayList<INDArray> weightINDArrays) {
        this.weightINDArrays = weightINDArrays;
    }

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
//        System.out.println("fanIn = " + fanIn + ", fanOut = " + fanOut + ", shape = " + Arrays.toString(shape) + ", order = " + order + ", paramView = " + paramView);

        INDArray result = weightINDArrays.get(layerCount);

//        System.out.println("-----");
//        System.out.println("Layer = " + counter);
//        System.out.println("result = " + result);
        layerCount++;
        if (layerCount == netTest.NUM_LAYERS) {
            layerCount = 0;
        }
        return result.reshape(order, shape);
    }
}
