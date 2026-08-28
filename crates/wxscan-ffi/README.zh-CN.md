# wxscan-ffi

[English](README.md) · **简体中文**

[`wxscan`](../wxscan) 的 C ABI，供 C、C++、Swift、Kotlin、Python 以及其它语言的
调用方使用。

`include/wxscan.h` 由这些源码用 cbindgen 生成并提交进仓库，所以使用方既不需要
cbindgen 也不需要 Rust 工具链。

```c
#include "wxscan.h"

// 两个模型都可以是 NULL，那样就是在没有 CNN 阶段的模式下解码。
// 权重加载失败时句柄为零。
WxScanScannerId scanner = wxscan_scanner_new(detect, detect_len, sr, sr_len);

// 一张正立的、紧密排列的灰度图。
WxScanResults *out = wxscan_scan_gray(scanner, gray, width, height);
for (size_t i = 0; i < out->results_len; i++) {
    printf("%s\n", out->results[i].text);
}
wxscan_results_free(out);

wxscan_scanner_release(scanner);
```

对磁盘上的图片，`wxscan_scan_path` 自己读取并解码文件，所以手上只有一个路径的调用方
永远不必把像素落地：一张 1200 万像素的照片作为 RGBA 是 48 MB，而一个把它跨线程或跨
isolate 传递的调用方要为这块缓冲付不止一次代价。

```c
WxScanStatus status;
WxScanResults *out = wxscan_scan_path(scanner, "/tmp/photo.jpg", &status);
// status 区分「文件读不了」和「一张里面没有码的图片」；后者是 Ok，
// 只是结果集为空。
```

文件里记录的方向会被应用上，所以一张把手机横过来拍的照片扫描时是正立的，它的坐标也
和当时屏幕上看到的对得上。PNG、JPEG 和 GIF 会被解码，那是相册选择器会写出的全部格式；
HEIC 不会，必须读 HEIC 的调用方需要平台自己的解码器配合 `wxscan_scan_pixels`。这个
入口在 `image-io` feature 之后，默认开启；它要占 436 KB 的解码器，所以用不上它的构建
可以把它关掉。

对相机帧，`wxscan_scan_frame` 额外接受行跨距、旋转，以及一个镜像返回 x 坐标的标志。
帧本身从不被镜像，因为检测器是在非镜像输入上训练的；这个标志存在，是为了让坐标和
一个镜像显示的预览对得上——前置摄像头的预览通常就是镜像的。

结果是普通的 C 结构体。序列化——如果调用方需要的话——属于那个调用方的绑定层。

扫描器是一个显式的句柄而不是全局单例，所以多个扫描器可以带着不同的模型共存，调用之间
也不会去争同一把锁。一个实例同一时刻扫一帧。

### 句柄不是指针

`WxScanScannerId` 是一个由本库发放、并在它自己的一张表里查找的数字。一个已被释放的、
从未存在过的，或者调用方凭空编出来的句柄，查不到任何东西，于是作为一次普通的失败
返回——一个零、一个 NULL 结果、`WxScanStatus::BadArgument`。换成地址的话就会被解引用，
而崩溃会落在离出错点很远的地方。

句柄从不复用，所以一个陈旧的句柄绝不会在之后变成指向另一个扫描器。

它同时还是引用计数的，因为两边同时持有一个扫描器是常态，而且谁也看不见对方的生命
周期——一个托管应用为静态图片持有它，同时一个相机绑定用同一个句柄解码帧：

```c
wxscan_scanner_retain(scanner);   // 相机绑定现在也是一个持有者
...
wxscan_scanner_release(scanner);  // 然后还回去；最后一个走的人负责释放
```

只有单一持有者的调用方从不调用 retain，而是像对待任何其它分配那样，把 `new` 和
`release` 配成一对。释放一个已经不存在的句柄什么也不做，debug 构建会在 stderr 上
说一声——但在另一个持有者还在的时候多释放一次，拿走的是**那个持有者的**引用，而且是
静默的。这件事无法被检测：两次释放彼此无法区分。取了几次引用，就释放几次。

## 链接

本 crate 构建为静态库和 rlib。它有意不产出 cdylib：那会要求在构建时就解析 TFLite 的
符号，而这里的设计是把它们留给宿主构建系统的最终链接步骤。想要一个共享库的调用方，
应该用自己的一个 crate 把本 crate 包起来，并在那里提供搜索路径——参见
[wxscan](https://github.com/wilinz/wxscan) 里的 `wxscan_core` 包，那是一个覆盖五个
平台的完整例子。

[wxscan-rs](https://github.com/wilinz/wxscan-rs) 的一部分。Apache-2.0。
