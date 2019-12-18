import io.jenetics.*;
import io.jenetics.engine.Codecs;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionStatistics;
import io.jenetics.util.DoubleRange;
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
        value = test.fitness(weightList);

        return value;
    }
    public static void main(String[] args) {
        final Engine<DoubleGene, Double> engine = Engine
                .builder(
                        SimpleGeneticTrainer::fitnessFunction,
                        Codecs.ofVector(DoubleRange.of(-R, R), N))
                .populationSize(50)
                .optimize(Optimize.MINIMUM)
                .survivorsSelector(new TournamentSelector<>(5))
                .offspringSelector(new RouletteWheelSelector<>())
                .alterers(
                        new Mutator<>(0.05),
                        new MeanAlterer<>(1),
                        new SinglePointCrossover<>(0.4))
                .build();

        final EvolutionStatistics<Double, ?>
                statistics = EvolutionStatistics.ofNumber();

        final Phenotype<DoubleGene, Double> best = engine.stream()
//                .limit(bySteadyFitness(7))
                .limit(50)
                .peek(statistics)
                .collect(toBestPhenotype());

        System.out.println(statistics);
        ArrayList<Double> weightList = new ArrayList<>(N);
        for (DoubleGene doubleGene : best.getGenotype().getChromosome()) {
            weightList.add(doubleGene.getAllele());
        }
        ArrayList<INDArray> weights = NetTest.weightListToINDArray(weightList);
        NetTest.printWeights(weights);
    }

}
