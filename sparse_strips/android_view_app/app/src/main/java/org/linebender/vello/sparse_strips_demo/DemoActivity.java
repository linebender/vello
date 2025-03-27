package org.linebender.vello.sparse_strips_demo;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.FrameLayout;

public final class DemoActivity extends Activity {
    static {
         System.loadLibrary("main");
    }

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        View view = new DemoView(this);
        view.setLayoutParams(
                new FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT));
        view.setFocusable(true);
        view.setFocusableInTouchMode(true);
        FrameLayout layout = new FrameLayout(this);
        layout.addView(view);
        setContentView(layout);
        view.requestFocus();
    }
}
