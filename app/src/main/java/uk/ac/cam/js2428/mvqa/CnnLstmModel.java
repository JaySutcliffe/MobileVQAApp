package uk.ac.cam.js2428.mvqa;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;

import uk.ac.cam.js2428.mvqa.ml.Cnn;
import uk.ac.cam.js2428.mvqa.ml.Vqa;



public class CnnLstmModel extends VqaModel {
    private final Context context;
    private Cnn cnn;
    private Vqa model;
    private TensorBuffer imageFeature;
    private TensorBuffer questionFeature;
    private TensorBuffer cnnImageFeature;

    private ImageProcessor imageProcessor;
    private TensorImage tImage;

    private static final int PACKET_COUNT = 20;
    private static final int PACKET_SIZE = 20;

    @Override
    public void setImage(String imageLocation) throws IOException {
        AssetManager assetManager = context.getAssets();
        InputStream is = assetManager.open(imageLocation);
        Bitmap bitmap = BitmapFactory.decodeStream(is);
        tImage.load(bitmap);
        imageFeature = imageProcessor.process(tImage).getTensorBuffer();
        cnnImageFeature = cnn.process(imageFeature).getOutputFeature0AsTensorBuffer();
    }

    @Override
    public String runInference(String question) throws QuestionException {
        questionFeature.loadArray(parseQuestion(question));
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
        return decodeAnswer(answer);
    }

    @Override
    public EvaluationOutput evaluateQuestionOnly() {
        try {
            long[] elapsedTime = new long[PACKET_COUNT];
            int[] matches = new int[PACKET_COUNT];
            for (int i = 0; i < PACKET_COUNT; i++) {
                float[][] questions = new float[PACKET_SIZE][maxQuestionLength];
                float[][] imageFeatures = new float[PACKET_SIZE][4096];
                int[] answers = new int[PACKET_SIZE];
                InputStream is = context.getAssets().open(
                        "packets/test_packet" + i + ".bin");
                DataInputStream dis = new DataInputStream(is);
                for (int n = 0; n < PACKET_SIZE; n++) {
                    for (int m = 0; m < maxQuestionLength; m++) {
                        questions[n][m] = dis.readFloat();
                    }
                }
                for (int n = 0; n < PACKET_SIZE; n++) {
                    for (int m = 0; m < 4096; m++) {
                        imageFeatures[n][m] = dis.readFloat();
                    }
                }
                for (int n = 0; n < PACKET_SIZE; n++) {
                    answers[n] = (int)dis.readFloat();
                }
                dis.close();

                long startTime = System.currentTimeMillis();
                for (int j = 0; j < PACKET_SIZE; j++) {
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
                    if (answers[j] == answer) {
                        matches[i]++;
                    }
                }
                elapsedTime[i] = System.currentTimeMillis() - startTime;
            }
            return new EvaluationOutput(PACKET_SIZE, matches, elapsedTime);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public CnnLstmModel (Context context) {
        super(context);
        this.context = context;

        // https://www.tensorflow.org/lite/convert/metadata
        Model.Options options;
        CompatibilityList compatList = new CompatibilityList();

        if (compatList.isDelegateSupportedOnThisDevice()){
            // if the device has a supported GPU, add the GPU delegate
            options = new Model.Options.Builder().setDevice(Model.Device.GPU).build();
        } else {
            // if the GPU is not supported, run on 4 threads
            options = new Model.Options.Builder().setNumThreads(4).build();
        }

        try {
            imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                            .add(new NormalizeOp(127.5f, 127.5f))
                            .build();
            tImage = new TensorImage(DataType.FLOAT32);
            cnn = Cnn.newInstance(context, options);
        } catch (IOException e) {
            System.err.println("Problem initialising TensorFlow VQA model");
            e.printStackTrace();
        }

        // Initialising the VQA model
        try {
            model = Vqa.newInstance(context, options);
            questionFeature =
                    TensorBuffer.createFixedSize(new int[]{1, 26}, DataType.FLOAT32);
            cnnImageFeature =
                    TensorBuffer.createFixedSize(new int[]{1, 4096}, DataType.FLOAT32);
        } catch (IOException e) {
            System.err.println("Problem initialising TensorFlow VQA model");
            e.printStackTrace();
        }
    }

}