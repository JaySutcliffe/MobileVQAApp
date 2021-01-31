package uk.ac.cam.js2428.mvqa.models;

import android.content.Context;

public class CnnI8LstmModel extends CnnLstmModelBase {

    @Override
    protected void setCnnImageFeature() {

    }

    @Override
    protected float[] getAnswer() {
        return new float[0];
    }

    public CnnI8LstmModel(Context context) {
        super(context);
    }
}
