package uk.ac.cam.js2428.mvqa;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Debug;
import android.os.Trace;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Map;

import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import io.jhdf.api.Node;
import uk.ac.cam.js2428.mvqa.ml.Vgg19Quant8;
import uk.ac.cam.js2428.mvqa.ml.Vqa;

public class CnnLstmModel extends VqaModel {
    private final Context context;
    private Vgg19Quant8 cnn;
    private Vqa model;
    private TensorBuffer imageFeature;
    private TensorBuffer questionFeature;
    private TensorBuffer cnnImageFeature;

    private static final int packetCount = 20;

    /*
    This code is the preprocessing code written by Stackoverflow user Tom
    https://stackoverflow.com/questions/56034981/how-to-fix-the-image-preprocessing-difference-between-tensorflow-and-android-stu
    */
    private static float[][][] preprocessImageInput(Bitmap bitmap) {
        final float[] imagenet_means_caffe = new float[]{103.939f, 116.779f, 123.68f};

        float[][][] result = new float[bitmap.getHeight()][bitmap.getWidth()][3];// Assuming rgb
        for (int y = 0; y < bitmap.getHeight(); y++) {
            for (int x = 0; x < bitmap.getWidth(); x++) {
                final int px = bitmap.getPixel(x, y);
                // rgb-->bgr, then subtract means.  no scaling
                result[y][x][0] = (Color.blue(px) - imagenet_means_caffe[0]);
                result[y][x][1] = (Color.green(px) - imagenet_means_caffe[1]);
                result[y][x][2] = (Color.red(px) - imagenet_means_caffe[2]);
            }
        }
        return result;
    }

    private static float[] flatten3D(float[][][] f) {
        float[] result = new float[f.length * f[0].length * f[0][0].length];
        for (int i = 0; i < f.length; i++) {
            for (int j = 0; j < f[0].length; j++) {
                for (int k = 0; k < f[0][0].length; k++) {
                    result[f.length * f[0].length * i + f[0].length * j + k] = f[i][j][k];
                }
            }
        }
        return result;
    }

    @Override
    public void setImage(String imageLocation) throws IOException {
        AssetManager assetManager = context.getAssets();
        InputStream is = assetManager.open(imageLocation);
        Bitmap bitmap = BitmapFactory.decodeStream(is);
        Bitmap resizedBitmap =
                Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        float[][][] imagePreprocessed = preprocessImageInput(resizedBitmap);
        imageFeature.loadArray(flatten3D(imagePreprocessed));
    }

    @Override
    public String runInference(String question) throws QuestionException {
        questionFeature.loadArray(parseQuestion(question));
        Vgg19Quant8.Outputs cnnOutputs = cnn.process(imageFeature);
        cnnImageFeature = cnnOutputs.getOutputFeature0AsTensorBuffer();
        Vqa.Outputs vqaOutputs = model.process(questionFeature, cnnImageFeature);
        TensorBuffer answerFeature = vqaOutputs.getOutputFeature0AsTensorBuffer();
        return decodeAnswer(answerFeature.getIntArray()[0]);
    }

    public void testQuestionOnly() {
        try {
            long[] elapsedTime = new long[packetCount];
            int[] matches = new int[packetCount];
            for (int i = 0; i < packetCount; i++) {
                InputStream io = context.getAssets().open("packets/test_packet" + i + ".h5");
                HdfFile hdfFile = HdfFile.fromInputStream(io);
                Dataset datasetQuestions = hdfFile.getDatasetByPath("ques");
                Dataset datasetImageFeatures = hdfFile.getDatasetByPath("img_feats");
                Dataset datasetAnswers = hdfFile.getDatasetByPath("answers");
                float[][] questions = (float[][])datasetQuestions.getData();
                float[][] imageFeatures = (float[][])datasetImageFeatures.getData();
                int[] answers = (int[])datasetAnswers.getData();

                long startTime = System.currentTimeMillis();
                int match = 0;
                for (int j = 0; j < 10; j++) {
                    cnnImageFeature.loadArray(imageFeatures[j]);
                    questionFeature.loadArray(questions[j]);
                    Vqa.Outputs vqaOutputs = model.process(questionFeature, cnnImageFeature);
                    TensorBuffer answerFeature = vqaOutputs.getOutputFeature0AsTensorBuffer();
                    float[] probs = answerFeature.getFloatArray();
                    float maxProb = 0;
                    int answer = -1;
                    for (int k = 0; k < probs.length; k++) {
                       if (maxProb < probs[k]) {
                           answer = k;
                           maxProb = probs[k];
                       }
                    }
                    if (answers[i] == answer) {
                        match++;
                    }
                }
                hdfFile.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public CnnLstmModel (Context context) {
        super(context);
        this.context = context;
        /*
        try {
            cnn = Vgg19Quant8.newInstance(context);
            imageFeature =
                    TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
        } catch (IOException e) {
            System.err.println("Problem initialising TensorFlow VQA model");
            e.printStackTrace();
        }*/

        // Initialising the VQA model
        try {
            model = Vqa.newInstance(context);
            questionFeature =
                    TensorBuffer.createFixedSize(new int[]{1, 26}, DataType.FLOAT32);
            cnnImageFeature =
                    TensorBuffer.createFixedSize(new int[]{1, 1000}, DataType.FLOAT32);
        } catch (IOException e) {
            System.err.println("Problem initialising TensorFlow VQA model");
            e.printStackTrace();
        }
        testQuestionOnly();
    }

}