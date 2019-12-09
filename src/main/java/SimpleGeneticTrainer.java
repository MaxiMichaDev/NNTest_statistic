import static io.jenetics.engine.EvolutionResult.toBestPhenotype;
import static io.jenetics.engine.Limits.bySteadyFitness;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;
import io.jenetics.DoubleGene;
import io.jenetics.MeanAlterer;
import io.jenetics.Mutator;
import io.jenetics.Optimize;
import io.jenetics.Phenotype;
import io.jenetics.engine.Engine;
import io.jenetics.engine.Codecs;
import io.jenetics.engine.EvolutionStatistics;
import io.jenetics.util.DoubleRange;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;



public class SimpleGeneticTrainer {
    private static final int R = 1;
    private static final int N = netTest.NUM_WEIGHTS;

    private static double fitnessFunction(final double[] weightArray){
        double value;
        List<Double> weightList = Arrays.asList(ArrayUtils.toObject(weightArray));

        netTest test = new netTest();
        value = test.fitness(weightList);
        System.out.println(value);
        return value;
    }
    public static void main(String[] args) {
//        INDArray dummyWeights = Nd4j.arange(32).div(100); //INDArray von 0 bis 31, geteilt durch 100
//        List<Double> dummyWeightList = Arrays.asList(ArrayUtils.toObject(dummyWeights.toDoubleVector())); //zu Liste konvertieren (Java sehr redundant hier xD)
//        System.out.println("dummyWeightList = " + dummyWeightList);
//        Double[] WeightList = new Double[32];
//        Arrays.fill(WeightList, 0.5);
//        List<Double> dummyWeightList = Arrays.asList(WeightList);
//        netTest test = new netTest();
//        double fitness = test.fitness(/*WeightList*/dummyWeightList);
//        System.out.println("fitness = " + fitness);

        final Engine<DoubleGene, Double> engine = Engine
                .builder(
                        SimpleGeneticTrainer::fitnessFunction,
                        Codecs.ofVector(DoubleRange.of(-R, R), N))
                .populationSize(1000)
                .optimize(Optimize.MINIMUM)
                .alterers(
                        new Mutator<>(0.01),
                        new MeanAlterer<>(0.5))
                .build();

        final EvolutionStatistics<Double, ?>
                statistics = EvolutionStatistics.ofNumber();

        final Phenotype<DoubleGene, Double> best = engine.stream()
//                .limit(bySteadyFitness(7))
                .limit(100)
                .peek(statistics)
                .collect(toBestPhenotype());

        System.out.println(statistics);
        System.out.println(best);
    }

}
