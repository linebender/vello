package org.linebender.vello.sparse_strips_demo;

import android.content.Context;

import org.linebender.android.rustview.RustView;

public final class DemoView extends RustView {
    @Override
    protected native long newViewPeer(Context context);

    public DemoView(Context context) {
        super(context);
    }
}
