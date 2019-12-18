import org.deeplearning4j.nn.weights.WeightInitXavier;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;

public class CustomWeightInitializer extends WeightInitXavier {

    int layerCount;
    ArrayList<INDArray> weightINDArrays;

    public CustomWeightInitializer(ArrayList<INDArray> weightINDArrays) {
        this.weightINDArrays = weightINDArrays;
    }

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        INDArray result = weightINDArrays.get(layerCount);

        layerCount++;
        if (layerCount == NetTest.NUM_LAYERS) {
            layerCount = 0;
        }
        return result;//.reshape(order, shape);
    }
}
