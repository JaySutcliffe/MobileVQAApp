package uk.ac.cam.js2428.mvqa.models;

import android.content.Context;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;

import uk.ac.cam.js2428.mvqa.ml.FullAttentionVqaF16;
import uk.ac.cam.js2428.mvqa.ml.Mobilenet3by3;

public class FullAttentionModel extends CnnLstmModelBase {
    private Mobilenet3by3 cnn;
    private FullAttentionVqaF16 model;

    @Override
    protected void setCnnImageFeature() {
        cnnImageFeature = cnn.process(imageFeature).getOutputFeature0AsTensorBuffer();
    }

    @Override
    protected float[] getAnswer() {
        FullAttentionVqaF16.Outputs vqaOutputs = model.process(questionFeature, cnnImageFeature);
        return vqaOutputs.getOutputFeature0AsTensorBuffer().getFloatArray();
    }

    public FullAttentionModel(Context context) {
        super(context);
        maxQuestionLength = 14;

        Model.Options options1;
        Model.Options options2;
        CompatibilityList compatList = new CompatibilityList();

        if (compatList.isDelegateSupportedOnThisDevice()){
            options1 = new Model.Options.Builder().setDevice(Model.Device.GPU).build();
        } else {
            options1 = new Model.Options.Builder().setNumThreads(4).build();
        }

        options2 = new Model.Options.Builder().setNumThreads(4).build();

        try {
            cnn = Mobilenet3by3.newInstance(context, options1);
        } catch (IOException e) {
            System.err.println("Problem initialising TensorFlow VQA model");
            e.printStackTrace();
        }

        // Initialising the VQA model
        try {
            model = FullAttentionVqaF16.newInstance(context, options2);
            questionFeature =
                    TensorBuffer.createFixedSize(new int[]{1, 14}, DataType.FLOAT32);
            cnnImageFeature =
                    TensorBuffer.createFixedSize(new int[]{1, 3, 3, 1280}, DataType.FLOAT32);
        } catch (IOException e) {
            System.err.println("Problem initialising TensorFlow VQA model");
            e.printStackTrace();
        }
    }
}
