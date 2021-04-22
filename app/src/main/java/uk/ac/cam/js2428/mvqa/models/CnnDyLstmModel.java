package uk.ac.cam.js2428.mvqa.models;

import android.content.Context;

public class CnnDyLstmModel extends CnnLstmModelBase {

    @Override
    protected void setCnnImageFeature() {

    }

    @Override
    protected float[] getAnswer() {
        return new float[0];
    }

    public CnnDyLstmModel(Context context) {
        super(context);
    }
}
