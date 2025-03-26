# Demo of Android View with Vello Hybrid

[Android View](https://github.com/mwcampbell/android-view) is the upcoming new way to make Android apps with Rust.

To run this, you can use the following commands.

```sh
cd sparse_strips/android_view_app/
cargo ndk -t arm64-v8a -o app/src/main/jniLibs/ build -p sparse_strips_android [--release]
./gradlew build
./gradlew installDebug
```

You then should open the app "Vello Sparse Strips Demo".
To view logs and tracing output, run:

```sh
adb shell run-as org.linebender.vello.sparse_strips_demo logcat -v color
```
