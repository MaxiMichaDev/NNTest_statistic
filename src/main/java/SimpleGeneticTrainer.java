import static io.jenetics.engine.Limits.bySteadyFitness;
import io.jenetics.*;
import io.jenetics.engine.Codecs;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionStatistics;
import io.jenetics.util.DoubleRange;
import io.jenetics.ext.SimulatedBinaryCrossover;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static io.jenetics.engine.EvolutionResult.toBestPhenotype;


public class SimpleGeneticTrainer {
    private static final int R = 1;
    private static final int N = NetTest.numWeights();

    private static double fitnessFunction(final double[] weightArray){
        double value;
        List<Double> weightList = Arrays.asList(ArrayUtils.toObject(weightArray));

        NetTest test = new NetTest();
        ArrayList<INDArray> weights = test.weightListToINDArray(weightList);
        value = test.fitness(weights, false);

        return value;
    }
    public static void main(String[] args) {
        final Engine<DoubleGene, Double> engine = Engine
                .builder(
                        SimpleGeneticTrainer::fitnessFunction,
                        Codecs.ofVector(DoubleRange.of(-R, R), N))
                .populationSize(100)
                .optimize(Optimize.MINIMUM)
//                .offspringFraction(0.05)
//                .survivorsFraction(0.04)
//                .survivorsSelector(new TruncationSelector<>(10))
//                .offspringSelector(new TournamentSelector<>(5))
                .alterers(
                        new Mutator<>(0.03),
                        new MeanAlterer<>(0.6),
//                        new SinglePointCrossover<>(0.8))
                        new SimulatedBinaryCrossover<>(0.8))
                .build();

        final EvolutionStatistics<Double, ?>
                statistics = EvolutionStatistics.ofNumber();

        final Phenotype<DoubleGene, Double> best = engine.stream()
//                .limit(bySteadyFitness(10))
                .limit(100)
                .peek(statistics)
                .collect(toBestPhenotype());

        System.out.println(statistics);
        ArrayList<Double> weightList = new ArrayList<>(N);
        for (DoubleGene doubleGene : best.getGenotype().getChromosome()) {
            weightList.add(doubleGene.getAllele());
        }
        System.out.println(best);
        ArrayList<INDArray> weights = NetTest.weightListToINDArray(weightList);
        NetTest.printWeights(weights);

        NetTest eval = new NetTest();
        System.out.println(eval.fitness(weights, true));
    }

}
