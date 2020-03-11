/* *****************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.List;
import java.util.Random;

import org.datavec.image.transform.ColorConversionTransform;
import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Example transfer learning from a Tiny YOLO model pretrained on ImageNet and Pascal VOC
 * to perform object detection with bounding boxes on The Street View House Numbers (SVHN) Dataset.
 * <p>
 * References: <br>
 * - YOLO: Real-Time Object Detection: https://pjreddie.com/darknet/yolo/ <br>
 * - The Street View House Numbers (SVHN) Dataset: http://ufldl.stanford.edu/housenumbers/ <br>
 * <p>
 * Please note, cuDNN should be used to obtain reasonable performance: https://deeplearning4j.org/cudnn
 *
 * @author saudet
 */
public class Yolo {
//    private static final Logger log = LoggerFactory.getLogger(HouseNumberDetection.class);
    private String classes = System.getProperty("model.classes");
    // Enable different colour bounding box for different classes
    public static final Scalar RED = RGB(255.0, 0, 0);
    public static final Scalar GREEN = RGB(0, 255.0, 0);
    public static final Scalar BLUE = RGB(0, 0, 255.0);
    public static final Scalar YELLOW = RGB(255.0, 255.0, 0);
    public static final Scalar CYAN = RGB(0, 255.0, 255.0);
    public static final Scalar MAGENTA = RGB(255.0, 0.0, 255.0);
    public static final Scalar ORANGE = RGB(255.0, 128.0, 0);
    public static final Scalar PINK = RGB(255.0, 192.0, 203.0);
    public static final Scalar LIGHTBLUE = RGB(153.0, 204.0, 255.0);
    public static final Scalar VIOLET = RGB(238.0, 130.0, 238.0);

    public static void main(String[] args) throws java.lang.Exception {

        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;

        // number classes (digits) for the SVHN datasets
        int nClasses = 10;

        // parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 1.0;
        double[][] priorBoxes = {{2, 5}, {2.5, 6}, {3, 7}, {3.5, 8}, {4, 9}};
        double detectionThreshold = 0.6;

        // parameters for the training phase
        int batchSize = 10;
        int nEpochs = 1;
        double learningRate = 1e-4;

        int seed = 123;
        Random rng = new Random(seed);

//        SvhnDataFetcher fetcher = new SvhnDataFetcher();
//        File trainDir = fetcher.getDataSetPath(DataSetType.TRAIN);
//        File testDir = fetcher.getDataSetPath(DataSetType.TEST);


//        log.info("Load data...");

//        FileSplit trainData = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng);
//        FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);
//
//        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels,
//                        gridHeight, gridWidth, new SvhnLabelProvider(trainDir));
//        recordReaderTrain.initialize(trainData);
//
//        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels,
//                        gridHeight, gridWidth, new SvhnLabelProvider(testDir));
//        recordReaderTest.initialize(testData);
//
//         ObjectDetectionRecordReader performs regression, so we need to specify it here
//        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
//        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

//        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
//        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));


        ComputationGraph model;
        String modelFilename = "yolomodel.zip";

        if (new File(modelFilename).exists()) {
//            log.info("Load model...");

            model = ComputationGraph.load(new File(modelFilename), true);
        } else {
//            log.info("Build model...");

//            model = (ComputationGraph)TinyYOLO.builder().build().initPretrained();
            model = (ComputationGraph)YOLO2.builder().build().initPretrained();
            INDArray priors = Nd4j.create(priorBoxes);

//            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
//                    .seed(seed)
//                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
//                    .gradientNormalizationThreshold(1.0)
//                    .updater(new Adam.Builder().learningRate(learningRate).build())
//                    //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
//                    .l2(0.00001)
//                    .activation(Activation.IDENTITY)
//                    .trainingWorkspaceMode(WorkspaceMode.ENABLED)
//                    .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
//                    .build();
//
//            model = new TransferLearning.GraphBuilder(pretrained)
//                    .fineTuneConfiguration(fineTuneConf)
//                    .removeVertexKeepConnections("conv2d_9")
//                    .removeVertexKeepConnections("outputs")
//                    .addLayer("convolution2d_9",
//                            new ConvolutionLayer.Builder(1,1)
//                                    .nIn(1024)
//                                    .nOut(nBoxes * (5 + nClasses))
//                                    .stride(1,1)
//                                    .convolutionMode(ConvolutionMode.Same)
//                                    .weightInit(WeightInit.XAVIER)
//                                    .activation(Activation.IDENTITY)
//                                    .build(),
//                            "leaky_re_lu_8")
//                    .addLayer("outputs",
//                            new Yolo2OutputLayer.Builder()
//                                    .lambdaNoObj(lambdaNoObj)
//                                    .lambdaCoord(lambdaCoord)
//                                    .boundingBoxPriors(priors)
//                                    .build(),
//                            "convolution2d_9")
//                    .setOutputs("outputs")
//                    .build();
//            System.out.println(model.summary(InputType.convolutional(height, width, nChannels)));
//
//
//            log.info("Train model...");
//
//            model.setListeners(new ScoreIterationListener(1));
//            model.fit(train, nEpochs);
//
//            log.info("Save model...");
//            ModelSerializer.writeModel(model, modelFilename, true);
        }

        // visualize results on the test set
        NativeImageLoader imageLoader = new NativeImageLoader(width, height, 3);
        File pic = new File("C:\\Users\\maxi2\\Desktop\\index.jpg");
        ImageRecordReader imageRecordReader = new ImageRecordReader(height, width, nChannels);
        FileSplit fileSplit = new FileSplit(pic, NativeImageLoader.ALLOWED_FORMATS);
        ImageTransform imageTransform = new ColorConversionTransform(COLOR_BGR2RGB);
        imageRecordReader.initialize(fileSplit, imageTransform);
        RecordReaderDataSetIterator recordReaderDataSetIterator = new RecordReaderDataSetIterator.Builder(imageRecordReader, 1)
                .preProcessor(new ImagePreProcessingScaler(0, 1))
                .build();
//        recordReaderDataSetIterator.setPreProcessor(new ImagePreProcessingScaler(0, 1));
//        INDArray loadedImage = imageLoader.asImageMatrix(pic).getImage();
//        RecordMetaData recordMetaData = pic.
//        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame frame = new CanvasFrame("Frame");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout =
                (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer)model.getOutputLayer(0);
//        System.out.println("here 1");
//        List<String> labels = train.getLabels();
        String[] classes = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
                "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
                "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush" };
        recordReaderDataSetIterator.setCollectMetaData(true);
//        System.out.println("here 2");
//        Scalar[] colormap = {RED,BLUE,GREEN,CYAN,YELLOW,MAGENTA,ORANGE,PINK,LIGHTBLUE,VIOLET};

//        while (test.hasNext() && frame.isVisible()) {
//            System.out.println("here 3");
        org.nd4j.linalg.dataset.DataSet ds = recordReaderDataSetIterator.next();
//            RecordMetaDataImageURI metadata = (RecordMetaDataImageURI) ds.getExampleMetaData().get(0);
        INDArray features = ds.getFeatures();
        INDArray results = model.outputSingle(features);
        List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
//            File file = new File(metadata.getURI());
//        log.info(pic.getName() + ": " + objs);
//        System.out.println("here 4");

        Mat mat = imageLoader.asMat(features);
        Mat convertedMat = new Mat();
        mat.convertTo(convertedMat, CV_8U, 255, 0);
//            int w = metadata.getOrigW() * 2;
//            int h = metadata.getOrigH() * 2;
        int w = width;
        int h = height;
        Mat image = new Mat();
        resize(convertedMat, image, new Size(w, h));
        for (DetectedObject obj : objs) {
//                System.out.println("here 5");
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            String label = classes[obj.getPredictedClass()];
            int x1 = (int) Math.round(w * xy1[0] / gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / gridHeight);
            rectangle(image, new Point(x1, y1), new Point(x2, y2), RED);
            putText(image, label + obj.getConfidence(), new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, RED);
            System.out.println(obj.getPredictedClass());
        }
//        System.out.println("here 6");
        frame.setTitle(pic.getName() + " - Frame");
        frame.setCanvasSize(w, h);
        frame.showImage(converter.convert(image));
        frame.waitKey();
//        System.out.println("here 7");
//        }
        frame.dispose();
    }
}
