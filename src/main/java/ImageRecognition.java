import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.toIntExact;

public class ImageRecognition {
    protected static long SEED = 42;
    protected static Random RNG = new Random(SEED);
    protected static int MAX_PATHS_PER_LABEL = 18;
    protected static double SPLIT_TEST_AND_TRAIN = 0.8;

    public static String DATA_LOCAL_PATH;

    public static void main(String[] args) {

        int numLabels;
        DATA_LOCAL_PATH = "images/";

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(DATA_LOCAL_PATH);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, RNG);
        int numExamples = toIntExact(fileSplit.length());
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;
        BalancedPathFilter pathFilter = new BalancedPathFilter(RNG, labelMaker, numExamples, numLabels, MAX_PATHS_PER_LABEL);

        InputSplit[] inputSplits = fileSplit.sample(pathFilter, SPLIT_TEST_AND_TRAIN, 1 - SPLIT_TEST_AND_TRAIN);
        InputSplit trainData = inputSplits[0];
        InputSplit testData = inputSplits[1];

        ImageTransform flipTransform1 = new FlipImageTransform(RNG);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(RNG, 42);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(new Pair<>(flipTransform1,0.9),
                new Pair<>(flipTransform2,0.8),
                new Pair<>(warpTransform,0.5));

        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);


    }
}