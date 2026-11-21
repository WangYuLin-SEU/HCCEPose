<p align="center">
  <img src="docs/assets/hccepose-readme-banner.webp" width="100%" alt="HCCEPose">
</p>

<p align="center">
  <a href="https://wangyulin-seu.github.io/hccepose/?lang=en"><img src="https://img.shields.io/badge/Website-HCCEPose-1F6FEB.svg?logo=googlechrome&amp;logoColor=white" height="20" alt="Website"></a>
  <a href="https://wangyulin-seu.github.io/hccepose/docs/index.html?lang=en"><img src="https://img.shields.io/badge/Docs-guide-1F6FEB.svg?logo=readthedocs&amp;logoColor=white" height="20" alt="Docs"></a>
  <a href="https://arxiv.org/abs/2510.10177"><img src="https://img.shields.io/badge/arXiv-2510.10177-B31B1B.svg?logo=arxiv" height="20" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/SEU-WYL/HccePose"><img src="https://img.shields.io/badge/HuggingFace-HccePose-FFD21E.svg?logo=huggingface&amp;logoColor=white" height="20" alt="Hugging Face"></a>
</p>

<p align="center">English · <a href="README_CN.md">中文</a></p>

<h3 align="center">6D pose estimation from a single RGB image</h3>
<p align="center">Predict front and back surface coordinates, then solve the object-to-camera pose from ultra-dense 2D–3D correspondences.</p>

<p align="center">
  <a href="show_vis/fig2.jpg"><img src="docs/assets/readme-showcase/method-en.svg" width="100%" alt="HccePose(BF) paper overview: front/back surface prediction, 2D–3D correspondences and pose estimation"></a>
</p>
<p align="center">
  <img src="docs/assets/readme-showcase/features-en.svg" width="100%" alt="RGB image · Front and back surfaces · Ultra-dense 2D–3D matches · Object-to-camera pose">
</p>
<p align="center"><a href="docs/en/method.md">Explore the method →</a></p>

## Models and example data

Pretrained weights, object models and example inputs are hosted on **[Hugging Face](https://huggingface.co/datasets/SEU-WYL/HccePose)**.

<table align="center">
  <thead>
    <tr><th align="center">Resource</th><th align="center">Download / instructions</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Bin-picking models and detector/pose weights</td><td align="center"><a href="https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/demo-bin-picking">Hugging Face files</a></td></tr>
    <tr><td align="center">RGB-D images, depth and camera intrinsics</td><td align="center"><a href="https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_imgs_RGBD">RGB-D frames</a></td></tr>
    <tr><td align="center">RGB images and input videos</td><td align="center"><a href="https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_imgs">Images</a> · <a href="https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_videos">Videos</a></td></tr>
    <tr><td align="center">Seven core BOP datasets: pretrained pose weights</td><td align="center"><a href="docs/en/bop-evaluation.md#pretrained-weights">Weights and evaluation guide</a></td></tr>
  </tbody>
</table>

For a first run, follow [Models and inputs](docs/en/assets.md#download-one-rgb-d-example) to download the matching minimal set and prepare the camera matrix. The full resource repository also contains training data that this example does not need.

## Find your starting point

<table align="center">
  <thead>
    <tr><th align="center">Goal</th><th align="center">Documentation</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Run pretrained inference</td><td align="center"><a href="docs/en/installation.md">Installation</a> → <a href="docs/en/assets.md">Models and inputs</a> → <a href="docs/en/quickstart.md">First prediction</a></td></tr>
    <tr><td align="center">Process images or video</td><td align="center"><a href="docs/en/rgb-inference.md">RGB images</a> · <a href="docs/en/video.md">Video frames</a></td></tr>
    <tr><td align="center">Estimate or refine with depth</td><td align="center"><a href="docs/en/rgbd-wapr.md">WAPR</a> · <a href="docs/en/rgbd-megapose.md">MegaPose</a> · <a href="docs/en/depth-comparison.md">Depth comparison</a></td></tr>
    <tr><td align="center">Train new objects</td><td align="center"><a href="docs/en/object-preparation.md">Meshes and symmetry</a> → <a href="docs/en/rendering.md">PBR rendering</a> → <a href="docs/en/detector-training.md">Detector</a> → <a href="docs/en/coordinate-labels.md">Surface labels</a> → <a href="docs/en/pose-training.md">Pose training</a></td></tr>
    <tr><td align="center">Integrate into an application</td><td align="center"><a href="docs/en/tester-api.md">Tester</a> · <a href="docs/en/results-api.md">Results and coordinates</a> · <a href="docs/en/refinement-api.md">Refinement</a> · <a href="docs/en/data-api.md">Data loading</a> · <a href="docs/en/utilities-api.md">Utilities</a></td></tr>
    <tr><td align="center">Accelerate, evaluate or explore</td><td align="center"><a href="docs/en/acceleration.md">ONNX/TensorRT</a> · <a href="docs/en/bop-evaluation.md">BOP evaluation</a> · <a href="docs/en/changelog.md">Public changelog</a> · <a href="docs/en/project-structure.md">Project structure</a></td></tr>
  </tbody>
</table>

## Get started

The project provides RGB inference, WAPR integration, and an optional interface to an external MegaPose installation. Prepare matching detector and object pose weights, object meshes, camera intrinsics, and input images, then run an inference example. See [installation](docs/en/installation.md), [resource conventions](docs/en/assets.md) and [your first prediction](docs/en/quickstart.md).

```bash
python examples/inference/predict_image.py
```

Run this command from the project root after editing the resource paths and object IDs at the end of the entry script. Outputs are saved to `outputs/rgb/`.

## WAPR integration

Install [WAPR](https://github.com/WangYuLin-SEU/WAPR/tree/4e591be318a828c91980c58ef8bbdc34e6599881) separately. The example restores instance masks predicted by HccePose for the current image to the full frame, then estimates poses using registered depth, camera intrinsics, and each object’s metric mesh and diameter. Each instance produces `pose_4x4_m` (object to camera, translation in meters) and a `score_6d` computed for that input. Input instance order is preserved. This API estimates poses from masks without HccePose pose seeds. See [WAPR usage](docs/en/rgbd-wapr.md).

Optional score filtering and per-object CAD rendered-mask NMS are provided in [the postprocessing tutorial](docs/en/wapr-postprocessing.md) and `examples/inference/postprocess_wapr.py`. The raw inference JSON is kept separately.

<p align="center"><sub>Saved outputs · WBPS ≥ 0.75 · Scores are not probabilities.</sub></p>

<table align="center">
<tr><th align="center">RGB · saved pose projections</th><th align="center">Depth samples + CAD + spatial frame</th></tr>
<tr><td align="center" width="50%"><a href="https://wangyulin-seu.github.io/hccepose/docs/rgbd-wapr.html?lang=en"><img src="docs/assets/readme-3d/rgb-projection-en.svg" width="420" alt="RGB · saved pose projections"></a></td><td align="center" width="50%">

```stl
solid hccepose_retained_with_depth_samples
facet normal -.993968 -.103567 -.036063
outer loop
vertex -.128282 .135053 -.336622
vertex -.128034 .131467 -.333146
vertex -.129991 .145252 -.318788
endloop
endfacet
facet normal .993969 .103562 .0360662
outer loop
vertex -.0686434 .141267 -.334458
vertex -.064729 .113697 -.363174
vertex -.0695929 .155024 -.347796
endloop
endfacet
facet normal .0978607 -.689227 -.717906
outer loop
vertex -.0672035 .149553 -.397935
vertex -.0873303 .151067 -.402132
vertex -.0674509 .153138 -.401411
endloop
endfacet
facet normal .993969 .103562 .0360657
outer loop
vertex -.0288846 .145409 -.333015
vertex -.0305944 .155608 -.315181
vertex -.0286372 .141824 -.329539
endloop
endfacet
facet normal -.0494901 .717105 -.695206
outer loop
vertex -.0335638 .198634 -.356894
vertex -.0674509 .153138 -.401411
vertex -.0610279 .195773 -.35789
endloop
endfacet
facet normal .049485 -.71711 .695201
outer loop
vertex -.0870829 .147481 -.398656
vertex -.0876857 .151727 -.394234
vertex -.117366 .163725 -.379744
endloop
endfacet
facet normal .097859 -.689232 -.717902
outer loop
vertex -.0846084 .111626 -.363896
vertex -.0644816 .110112 -.359698
vertex -.0843609 .108041 -.36042
endloop
endfacet
facet normal .993969 .103563 .0360638
outer loop
vertex -.0313591 .181264 -.367776
vertex -.0316066 .18485 -.371252
vertex -.0335638 .198634 -.356894
endloop
endfacet
facet normal -.993969 -.103562 -.0360661
outer loop
vertex -.0858631 .120463 -.354691
vertex -.0909973 .175051 -.36994
vertex -.0876857 .151727 -.394234
endloop
endfacet
facet normal .0494895 -.717105 .695206
outer loop
vertex -.0672035 .149553 -.397935
vertex -.0313591 .181264 -.367776
vertex -.0711179 .177122 -.369218
endloop
endfacet
facet normal .0494901 -.717105 .695206
outer loop
vertex -.117366 .163725 -.379744
vertex -.0909973 .175051 -.36994
vertex -.130756 .170908 -.371382
endloop
endfacet
facet normal -.633645 -.560586 -.533139
outer loop
vertex -.0843609 .108041 -.36042
vertex -.128034 .131467 -.333146
vertex -.114075 .127432 -.345494
endloop
endfacet
facet normal .993969 .103564 .036067
outer loop
vertex -.031966 .175483 -.334449
vertex -.0305944 .155608 -.315181
vertex -.0313591 .181264 -.367776
endloop
endfacet
facet normal -.0978599 .689228 .717905
outer loop
vertex -.129991 .145252 -.318788
vertex -.0305944 .155608 -.315181
vertex -.031966 .175483 -.334449
endloop
endfacet
facet normal .09786 -.689229 -.717905
outer loop
vertex -.128282 .135053 -.336622
vertex -.0909973 .175051 -.36994
vertex -.0887709 .14279 -.338665
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.0286372 .141824 -.329539
vertex -.0305944 .155608 -.315181
vertex -.129991 .145252 -.318788
endloop
endfacet
facet normal -.993969 -.103562 -.036065
outer loop
vertex -.130756 .170908 -.371382
vertex -.132961 .188278 -.3605
vertex -.131003 .174494 -.374858
endloop
endfacet
facet normal .0978606 -.689228 -.717905
outer loop
vertex -.0873303 .151067 -.402132
vertex -.0672035 .149553 -.397935
vertex -.0870829 .147481 -.398656
endloop
endfacet
facet normal .09786 -.689229 -.717905
outer loop
vertex -.0658106 .129369 -.378368
vertex -.0846084 .111626 -.363896
vertex -.0870829 .147481 -.398656
endloop
endfacet
facet normal .0494896 -.717105 .695206
outer loop
vertex -.0909973 .175051 -.36994
vertex -.117366 .163725 -.379744
vertex -.0876857 .151727 -.394234
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.0885228 .139195 -.335179
vertex -.0858631 .120463 -.354691
vertex -.114075 .127432 -.345494
endloop
endfacet
facet normal -.0494897 .717105 -.695206
outer loop
vertex -.0610279 .195773 -.35789
vertex -.131003 .174494 -.374858
vertex -.132961 .188278 -.3605
endloop
endfacet
facet normal .993969 .103562 .0360659
outer loop
vertex -.0711179 .177122 -.369218
vertex -.0695929 .155024 -.347796
vertex -.0658106 .129369 -.378368
endloop
endfacet
facet normal .0494894 -.717105 .695206
outer loop
vertex -.0534583 .119864 -.350424
vertex -.129991 .145252 -.318788
vertex -.0644816 .110112 -.359698
endloop
endfacet
facet normal -.993969 -.103562 -.0360661
outer loop
vertex -.0885228 .139195 -.335179
vertex -.0887709 .14279 -.338665
vertex -.0858631 .120463 -.354691
endloop
endfacet
facet normal .09786 -.689228 -.717906
outer loop
vertex -.0846084 .111626 -.363896
vertex -.064729 .113697 -.363174
vertex -.0644816 .110112 -.359698
endloop
endfacet
facet normal .77204 -.41413 -.482132
outer loop
vertex -.064729 .113697 -.363174
vertex -.0534583 .119864 -.350424
vertex -.0644816 .110112 -.359698
endloop
endfacet
facet normal -.993969 -.103562 -.0360661
outer loop
vertex -.0909973 .175051 -.36994
vertex -.0858631 .120463 -.354691
vertex -.0887709 .14279 -.338665
endloop
endfacet
facet normal -.633645 -.560588 -.533137
outer loop
vertex -.117366 .163725 -.379744
vertex -.131003 .174494 -.374858
vertex -.0873303 .151067 -.402132
endloop
endfacet
facet normal -.0494899 .717105 -.695206
outer loop
vertex -.0674509 .153138 -.401411
vertex -.0335638 .198634 -.356894
vertex -.0316066 .18485 -.371252
endloop
endfacet
facet normal -.0494901 .717104 -.695207
outer loop
vertex -.0858631 .120463 -.354691
vertex -.0846084 .111626 -.363896
vertex -.114075 .127432 -.345494
endloop
endfacet
facet normal -.633644 -.560587 -.533139
outer loop
vertex -.114075 .127432 -.345494
vertex -.128034 .131467 -.333146
vertex -.128282 .135053 -.336622
endloop
endfacet
facet normal .097859 -.689225 -.717909
outer loop
vertex -.128282 .135053 -.336622
vertex -.0887709 .14279 -.338665
vertex -.0885228 .139195 -.335179
endloop
endfacet
facet normal .77204 -.414129 -.482133
outer loop
vertex -.0286372 .141824 -.329539
vertex -.0534583 .119864 -.350424
vertex -.0288846 .145409 -.333015
endloop
endfacet
facet normal .993969 .103562 .0360661
outer loop
vertex -.0658106 .129369 -.378368
vertex -.0672035 .149553 -.397935
vertex -.0711179 .177122 -.369218
endloop
endfacet
facet normal .0494904 -.717104 .695207
outer loop
vertex -.0644816 .110112 -.359698
vertex -.129991 .145252 -.318788
vertex -.0843609 .108041 -.36042
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.128282 .135053 -.336622
vertex -.129991 .145252 -.318788
vertex -.130756 .170908 -.371382
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.131003 .174494 -.374858
vertex -.0674509 .153138 -.401411
vertex -.0873303 .151067 -.402132
endloop
endfacet
facet normal .0978606 -.689228 -.717905
outer loop
vertex -.0658106 .129369 -.378368
vertex -.0870829 .147481 -.398656
vertex -.0672035 .149553 -.397935
endloop
endfacet
facet normal .0978601 -.689229 -.717905
outer loop
vertex -.0658106 .129369 -.378368
vertex -.064729 .113697 -.363174
vertex -.0846084 .111626 -.363896
endloop
endfacet
facet normal -.633645 -.560589 -.533136
outer loop
vertex -.114075 .127432 -.345494
vertex -.0846084 .111626 -.363896
vertex -.0843609 .108041 -.36042
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.132961 .188278 -.3605
vertex -.129991 .145252 -.318788
vertex -.0610279 .195773 -.35789
endloop
endfacet
facet normal .77204 -.41413 -.482132
outer loop
vertex -.0534583 .119864 -.350424
vertex -.064729 .113697 -.363174
vertex -.0288846 .145409 -.333015
endloop
endfacet
facet normal .0978602 -.689228 -.717905
outer loop
vertex -.0288846 .145409 -.333015
vertex -.0686434 .141267 -.334458
vertex -.0695929 .155024 -.347796
endloop
endfacet
facet normal .0978602 -.689229 -.717905
outer loop
vertex -.0711179 .177122 -.369218
vertex -.0313591 .181264 -.367776
vertex -.0695929 .155024 -.347796
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.0313591 .181264 -.367776
vertex -.0305944 .155608 -.315181
vertex -.0288846 .145409 -.333015
endloop
endfacet
facet normal .09786 -.689229 -.717905
outer loop
vertex -.0313591 .181264 -.367776
vertex -.0288846 .145409 -.333015
vertex -.0695929 .155024 -.347796
endloop
endfacet
facet normal .993969 .103562 .0360659
outer loop
vertex -.064729 .113697 -.363174
vertex -.0658106 .129369 -.378368
vertex -.0695929 .155024 -.347796
endloop
endfacet
facet normal .993969 .103561 .0360666
outer loop
vertex -.0313591 .181264 -.367776
vertex -.0335638 .198634 -.356894
vertex -.031966 .175483 -.334449
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.0534583 .119864 -.350424
vertex -.0286372 .141824 -.329539
vertex -.129991 .145252 -.318788
endloop
endfacet
facet normal -.0494897 .717105 -.695206
outer loop
vertex -.131003 .174494 -.374858
vertex -.0610279 .195773 -.35789
vertex -.0674509 .153138 -.401411
endloop
endfacet
facet normal -.0494906 .717105 -.695205
outer loop
vertex -.0885228 .139195 -.335179
vertex -.114075 .127432 -.345494
vertex -.128282 .135053 -.336622
endloop
endfacet
facet normal .0978598 -.689229 -.717905
outer loop
vertex -.0909973 .175051 -.36994
vertex -.128282 .135053 -.336622
vertex -.130756 .170908 -.371382
endloop
endfacet
facet normal -.993969 -.103563 -.0360678
outer loop
vertex -.0876857 .151727 -.394234
vertex -.0870829 .147481 -.398656
vertex -.0846084 .111626 -.363896
endloop
endfacet
facet normal -.993969 -.103562 -.0360663
outer loop
vertex -.0876857 .151727 -.394234
vertex -.0846084 .111626 -.363896
vertex -.0858631 .120463 -.354691
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.0843609 .108041 -.36042
vertex -.129991 .145252 -.318788
vertex -.128034 .131467 -.333146
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.132961 .188278 -.3605
vertex -.130756 .170908 -.371382
vertex -.129991 .145252 -.318788
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.0610279 .195773 -.35789
vertex -.129991 .145252 -.318788
vertex -.0335638 .198634 -.356894
endloop
endfacet
facet normal -.09786 .689229 .717905
outer loop
vertex -.0335638 .198634 -.356894
vertex -.129991 .145252 -.318788
vertex -.031966 .175483 -.334449
endloop
endfacet
facet normal -.0494903 .717105 -.695206
outer loop
vertex -.0288846 .145409 -.333015
vertex -.064729 .113697 -.363174
vertex -.0686434 .141267 -.334458
endloop
endfacet
facet normal .77204 -.41413 -.482132
outer loop
vertex -.0316066 .18485 -.371252
vertex -.0313591 .181264 -.367776
vertex -.0672035 .149553 -.397935
endloop
endfacet
facet normal .77204 -.41413 -.482132
outer loop
vertex -.0674509 .153138 -.401411
vertex -.0316066 .18485 -.371252
vertex -.0672035 .149553 -.397935
endloop
endfacet
facet normal -.633645 -.560589 -.533137
outer loop
vertex -.0873303 .151067 -.402132
vertex -.0870829 .147481 -.398656
vertex -.117366 .163725 -.379744
endloop
endfacet
facet normal -.633645 -.560587 -.533139
outer loop
vertex -.131003 .174494 -.374858
vertex -.117366 .163725 -.379744
vertex -.130756 .170908 -.371382
endloop
endfacet
facet normal .0183974 -.702605 .711343
outer loop
vertex -.12393 .0446068 -.355535
vertex -.126451 .0415024 -.358536
vertex -.109183 .0345501 -.36585
endloop
endfacet
facet normal -.0184015 .702601 -.711346
outer loop
vertex -.125034 .0867629 -.398216
vertex -.159569 .100668 -.383589
vertex -.115361 .098675 -.3867
endloop
endfacet
facet normal -.86339 .347617 .365677
outer loop
vertex -.134359 .131712 -.353577
vertex -.13147 .120765 -.336349
vertex -.131838 .134817 -.350576
endloop
endfacet
facet normal -.0184011 .702602 -.711345
outer loop
vertex -.12577 .114867 -.426669
vertex -.111023 .10481 -.436984
vertex -.128291 .111763 -.429671
endloop
endfacet
facet normal .504199 .620898 .600224
outer loop
vertex -.080771 .142064 -.400971
vertex -.131838 .134817 -.350576
vertex -.0802625 .122651 -.381316
endloop
endfacet
facet normal -.504205 -.6209 -.600217
outer loop
vertex -.133991 .11766 -.33935
vertex -.128673 .115519 -.341603
vertex -.109534 .0885316 -.329763
endloop
endfacet
facet normal -.863393 .347611 .365675
outer loop
vertex -.159201 .0866155 -.369362
vertex -.16209 .097563 -.38659
vertex -.161722 .083511 -.372363
endloop
endfacet
facet normal -.0184008 .702603 -.711344
outer loop
vertex -.10056 .145912 -.396658
vertex -.0980388 .149016 -.393657
vertex -.080771 .142064 -.400971
endloop
endfacet
facet normal .0184015 -.702602 .711346
outer loop
vertex -.148131 .0821584 -.37405
vertex -.0994557 .103756 -.353978
vertex -.128673 .115519 -.341603
endloop
endfacet
facet normal -.504199 -.620899 -.600223
outer loop
vertex -.134359 .131712 -.353577
vertex -.10056 .145912 -.396658
vertex -.0998237 .117808 -.368204
endloop
endfacet
facet normal -.5042 -.620898 -.600223
outer loop
vertex -.109534 .0885316 -.329763
vertex -.0994557 .103756 -.353978
vertex -.0987197 .0756518 -.325524
endloop
endfacet
facet normal -.597497 -.251012 .761571
outer loop
vertex -.161722 .083511 -.372363
vertex -.126451 .0415024 -.358536
vertex -.135403 .0582719 -.360033
endloop
endfacet
facet normal -.0183992 .702601 -.711346
outer loop
vertex -.0970489 .122019 -.420349
vertex -.111023 .10481 -.436984
vertex -.10056 .145912 -.396658
endloop
endfacet
facet normal .863391 -.347615 -.365677
outer loop
vertex -.109183 .0345501 -.36585
vertex -.111023 .10481 -.436984
vertex -.0970489 .122019 -.420349
endloop
endfacet
facet normal -.863391 .347615 .365677
outer loop
vertex -.12393 .0446068 -.355535
vertex -.0994557 .103756 -.353978
vertex -.122138 .0758236 -.38098
endloop
endfacet
facet normal -.504199 -.620898 -.600223
outer loop
vertex -.128291 .111763 -.429671
vertex -.111023 .10481 -.436984
vertex -.109183 .0345501 -.36585
endloop
endfacet
facet normal .0184013 -.702602 .711345
outer loop
vertex -.0987197 .0756518 -.325524
vertex -.0789308 .071804 -.329836
vertex -.0961986 .0787563 -.322523
endloop
endfacet
facet normal -.863391 .347616 .365677
outer loop
vertex -.13147 .120765 -.336349
vertex -.134359 .131712 -.353577
vertex -.133991 .11766 -.33935
endloop
endfacet
facet normal -.863391 .347615 .365677
outer loop
vertex -.14855 .114237 -.370471
vertex -.159201 .0866155 -.369362
vertex -.133991 .11766 -.33935
endloop
endfacet
facet normal -.504199 -.620899 -.600223
outer loop
vertex -.0994557 .103756 -.353978
vertex -.109534 .0885316 -.329763
vertex -.128673 .115519 -.341603
endloop
endfacet
facet normal .504199 .620898 .600224
outer loop
vertex -.124666 .0727109 -.383989
vertex -.148131 .0821584 -.37405
vertex -.135403 .0582719 -.360033
endloop
endfacet
facet normal .5042 .620898 .600223
outer loop
vertex -.0802625 .122651 -.381316
vertex -.0961986 .0787563 -.322523
vertex -.0789308 .071804 -.329836
endloop
endfacet
facet normal -.0184015 .702602 -.711345
outer loop
vertex -.0998237 .117808 -.368204
vertex -.115361 .098675 -.3867
vertex -.14855 .114237 -.370471
endloop
endfacet
facet normal -.5042 -.620898 -.600223
outer loop
vertex -.151696 .10193 -.399838
vertex -.109183 .0345501 -.36585
vertex -.16209 .097563 -.38659
endloop
endfacet
facet normal .0184014 -.702602 .711346
outer loop
vertex -.124666 .0727109 -.383989
vertex -.122138 .0758236 -.38098
vertex -.148131 .0821584 -.37405
endloop
endfacet
facet normal -.863391 .347616 .365677
outer loop
vertex -.159201 .0866155 -.369362
vertex -.159569 .100668 -.383589
vertex -.16209 .097563 -.38659
endloop
endfacet
facet normal -.623522 .742615 -.244425
outer loop
vertex -.159569 .100668 -.383589
vertex -.151696 .10193 -.399838
vertex -.16209 .097563 -.38659
endloop
endfacet
facet normal .0184015 -.702602 .711346
outer loop
vertex -.0994557 .103756 -.353978
vertex -.148131 .0821584 -.37405
vertex -.122138 .0758236 -.38098
endloop
endfacet
facet normal -.597498 -.251014 .761569
outer loop
vertex -.109534 .0885316 -.329763
vertex -.0961986 .0787563 -.322523
vertex -.13147 .120765 -.336349
endloop
endfacet
facet normal .504199 .620898 .600224
outer loop
vertex -.131838 .134817 -.350576
vertex -.080771 .142064 -.400971
vertex -.0980388 .149016 -.393657
endloop
endfacet
facet normal .504199 .620898 .600224
outer loop
vertex -.148131 .0821584 -.37405
vertex -.159201 .0866155 -.369362
vertex -.135403 .0582719 -.360033
endloop
endfacet
facet normal -.597498 -.251013 .76157
outer loop
vertex -.135403 .0582719 -.360033
vertex -.126451 .0415024 -.358536
vertex -.12393 .0446068 -.355535
endloop
endfacet
facet normal -.863389 .347618 .36568
outer loop
vertex -.12393 .0446068 -.355535
vertex -.122138 .0758236 -.38098
vertex -.124666 .0727109 -.383989
endloop
endfacet
facet normal -.623522 .742615 -.244425
outer loop
vertex -.128291 .111763 -.429671
vertex -.151696 .10193 -.399838
vertex -.12577 .114867 -.426669
endloop
endfacet
facet normal -.0184015 .702601 -.711346
outer loop
vertex -.14855 .114237 -.370471
vertex -.134359 .131712 -.353577
vertex -.0998237 .117808 -.368204
endloop
endfacet
facet normal -.504199 -.620898 -.600224
outer loop
vertex -.16209 .097563 -.38659
vertex -.109183 .0345501 -.36585
vertex -.161722 .083511 -.372363
endloop
endfacet
facet normal .0184016 -.702602 .711346
outer loop
vertex -.12393 .0446068 -.355535
vertex -.109183 .0345501 -.36585
vertex -.0987197 .0756518 -.325524
endloop
endfacet
facet normal .504199 .620898 .600224
outer loop
vertex -.0961986 .0787563 -.322523
vertex -.131838 .134817 -.350576
vertex -.13147 .120765 -.336349
endloop
endfacet
facet normal -.863391 .347616 .365676
outer loop
vertex -.14855 .114237 -.370471
vertex -.133991 .11766 -.33935
vertex -.134359 .131712 -.353577
endloop
endfacet
facet normal -.863391 .347615 .365676
outer loop
vertex -.14855 .114237 -.370471
vertex -.159569 .100668 -.383589
vertex -.159201 .0866155 -.369362
endloop
endfacet
facet normal -.597499 -.251015 .761568
outer loop
vertex -.135403 .0582719 -.360033
vertex -.159201 .0866155 -.369362
vertex -.161722 .083511 -.372363
endloop
endfacet
facet normal .863391 -.347615 -.365677
outer loop
vertex -.0789308 .071804 -.329836
vertex -.109183 .0345501 -.36585
vertex -.0802625 .122651 -.381316
endloop
endfacet
facet normal -.623522 .742615 -.244425
outer loop
vertex -.151696 .10193 -.399838
vertex -.159569 .100668 -.383589
vertex -.12577 .114867 -.426669
endloop
endfacet
facet normal -.863391 .347615 .365677
outer loop
vertex -.12577 .114867 -.426669
vertex -.125034 .0867629 -.398216
vertex -.115361 .098675 -.3867
endloop
endfacet
facet normal -.863391 .347615 .365676
outer loop
vertex -.0998237 .117808 -.368204
vertex -.10056 .145912 -.396658
vertex -.115361 .098675 -.3867
endloop
endfacet
facet normal -.0184015 .702602 -.711345
outer loop
vertex -.10056 .145912 -.396658
vertex -.111023 .10481 -.436984
vertex -.12577 .114867 -.426669
endloop
endfacet
facet normal -.863391 .347615 .365677
outer loop
vertex -.10056 .145912 -.396658
vertex -.12577 .114867 -.426669
vertex -.115361 .098675 -.3867
endloop
endfacet
facet normal -.0184015 .702602 -.711345
outer loop
vertex -.159569 .100668 -.383589
vertex -.14855 .114237 -.370471
vertex -.115361 .098675 -.3867
endloop
endfacet
facet normal -.0184017 .702601 -.711346
outer loop
vertex -.10056 .145912 -.396658
vertex -.080771 .142064 -.400971
vertex -.0970489 .122019 -.420349
endloop
endfacet
facet normal -.504199 -.620898 -.600224
outer loop
vertex -.151696 .10193 -.399838
vertex -.128291 .111763 -.429671
vertex -.109183 .0345501 -.36585
endloop
endfacet
facet normal .504199 .620898 .600223
outer loop
vertex -.0961986 .0787563 -.322523
vertex -.0802625 .122651 -.381316
vertex -.131838 .134817 -.350576
endloop
endfacet
facet normal .5042 .620897 .600223
outer loop
vertex -.124666 .0727109 -.383989
vertex -.135403 .0582719 -.360033
vertex -.12393 .0446068 -.355535
endloop
endfacet
facet normal -.863391 .347615 .365677
outer loop
vertex -.0994557 .103756 -.353978
vertex -.12393 .0446068 -.355535
vertex -.0987197 .0756518 -.325524
endloop
endfacet
facet normal .0183996 -.702601 .711346
outer loop
vertex -.128673 .115519 -.341603
vertex -.133991 .11766 -.33935
vertex -.159201 .0866155 -.369362
endloop
endfacet
facet normal .0184012 -.702601 .711346
outer loop
vertex -.128673 .115519 -.341603
vertex -.159201 .0866155 -.369362
vertex -.148131 .0821584 -.37405
endloop
endfacet
facet normal -.504199 -.620898 -.600224
outer loop
vertex -.161722 .083511 -.372363
vertex -.109183 .0345501 -.36585
vertex -.126451 .0415024 -.358536
endloop
endfacet
facet normal .0184016 -.702602 .711346
outer loop
vertex -.0789308 .071804 -.329836
vertex -.0987197 .0756518 -.325524
vertex -.109183 .0345501 -.36585
endloop
endfacet
facet normal .863391 -.347615 -.365677
outer loop
vertex -.0802625 .122651 -.381316
vertex -.109183 .0345501 -.36585
vertex -.080771 .142064 -.400971
endloop
endfacet
facet normal .863391 -.347615 -.365677
outer loop
vertex -.080771 .142064 -.400971
vertex -.109183 .0345501 -.36585
vertex -.0970489 .122019 -.420349
endloop
endfacet
facet normal .504199 .620898 .600224
outer loop
vertex -.12577 .114867 -.426669
vertex -.159569 .100668 -.383589
vertex -.125034 .0867629 -.398216
endloop
endfacet
facet normal -.623522 .742614 -.244426
outer loop
vertex -.0980388 .149016 -.393657
vertex -.10056 .145912 -.396658
vertex -.134359 .131712 -.353577
endloop
endfacet
facet normal -.623523 .742614 -.244426
outer loop
vertex -.131838 .134817 -.350576
vertex -.0980388 .149016 -.393657
vertex -.134359 .131712 -.353577
endloop
endfacet
facet normal -.597499 -.251015 .761568
outer loop
vertex -.13147 .120765 -.336349
vertex -.133991 .11766 -.33935
vertex -.109534 .0885316 -.329763
endloop
endfacet
facet normal -.597498 -.251013 .76157
outer loop
vertex -.0961986 .0787563 -.322523
vertex -.109534 .0885316 -.329763
vertex -.0987197 .0756518 -.325524
endloop
endfacet
facet normal .937979 .0041826 .346666
outer loop
vertex .00160398 .215859 -.38076
vertex .00287104 .212406 -.384146
vertex .00760289 .226869 -.397124
endloop
endfacet
facet normal -.937978 -.0041877 -.346669
outer loop
vertex -.0546747 .215608 -.40156
vertex -.0641383 .186681 -.375605
vertex -.0595366 .228858 -.388565
endloop
endfacet
facet normal -.236592 -.723174 .648878
outer loop
vertex -.0768093 .221213 -.341738
vertex -.0593168 .22475 -.331418
vertex -.0780764 .224666 -.338351
endloop
endfacet
facet normal -.937978 -.00418718 -.346668
outer loop
vertex -.0921938 .21544 -.415427
vertex -.0861949 .22645 -.431791
vertex -.0909267 .211987 -.418813
endloop
endfacet
facet normal -.253419 .690652 .677332
outer loop
vertex -.1014 .267889 -.391151
vertex -.0780764 .224666 -.338351
vertex -.0754831 .268005 -.381572
endloop
endfacet
facet normal .253423 -.690657 -.677326
outer loop
vertex -.0580498 .221297 -.334805
vertex -.0565924 .225752 -.338802
vertex -.0254719 .241471 -.343187
endloop
endfacet
facet normal -.236589 -.723178 .648875
outer loop
vertex -.0453788 .186764 -.368671
vertex -.0628712 .183227 -.378991
vertex -.0441117 .183311 -.372058
endloop
endfacet
facet normal -.937979 -.00418633 -.346666
outer loop
vertex -.104865 .249973 -.38156
vertex -.106132 .253426 -.378173
vertex -.1014 .267889 -.391151
endloop
endfacet
facet normal .937978 .00418761 .346669
outer loop
vertex -.0423452 .196037 -.376991
vertex -.0485861 .250224 -.36076
vertex -.0565924 .225752 -.338802
endloop
endfacet
facet normal .25342 -.690652 -.677332
outer loop
vertex -.0768093 .221213 -.341738
vertex -.104865 .249973 -.38156
vertex -.0673457 .25014 -.367693
endloop
endfacet
facet normal .253419 -.690652 -.677332
outer loop
vertex -.0254719 .241471 -.343187
vertex -.0485861 .250224 -.36076
vertex -.011067 .250391 -.346893
endloop
endfacet
facet normal .495955 -.5084 .703959
outer loop
vertex -.0441117 .183311 -.372058
vertex .00287104 .212406 -.384146
vertex -.0136791 .206395 -.376827
endloop
endfacet
facet normal -.937978 -.00418544 -.346669
outer loop
vertex -.0932185 .245592 -.413018
vertex -.0861949 .22645 -.431791
vertex -.104865 .249973 -.38156
endloop
endfacet
facet normal .236591 .723175 -.648878
outer loop
vertex .00760289 .226869 -.397124
vertex -.0861949 .22645 -.431791
vertex -.0932185 .245592 -.413018
endloop
endfacet
facet normal -.236591 -.723176 .648877
outer loop
vertex .00160398 .215859 -.38076
vertex -.0485861 .250224 -.36076
vertex -.0371856 .219154 -.391231
endloop
endfacet
facet normal .253419 -.690652 -.677332
outer loop
vertex -.0909267 .211987 -.418813
vertex -.0861949 .22645 -.431791
vertex .00760289 .226869 -.397124
endloop
endfacet
facet normal .937979 .00418713 .346668
outer loop
vertex -.011067 .250391 -.346893
vertex -.00760228 .268308 -.356484
vertex -.0123341 .253845 -.343506
endloop
endfacet
facet normal -.236592 -.723175 .648878
outer loop
vertex -.0593168 .22475 -.331418
vertex -.0768093 .221213 -.341738
vertex -.0580498 .221297 -.334805
endloop
endfacet
facet normal -.236591 -.723176 .648877
outer loop
vertex -.0696766 .201774 -.360802
vertex -.0453788 .186764 -.368671
vertex -.0580498 .221297 -.334805
endloop
endfacet
facet normal .25342 -.690652 -.677332
outer loop
vertex -.0485861 .250224 -.36076
vertex -.0254719 .241471 -.343187
vertex -.0565924 .225752 -.338802
endloop
endfacet
facet normal -.253419 .690652 .677332
outer loop
vertex -.0359152 .215691 -.394626
vertex -.0423452 .196037 -.376991
vertex -.0136791 .206395 -.376827
endloop
endfacet
facet normal -.25342 .690652 .677332
outer loop
vertex -.0754831 .268005 -.381572
vertex -.0123341 .253845 -.343506
vertex -.00760228 .268308 -.356484
endloop
endfacet
facet normal -.937978 -.00418765 -.346669
outer loop
vertex -.0673457 .25014 -.367693
vertex -.0595366 .228858 -.388565
vertex -.0696766 .201774 -.360802
endloop
endfacet
facet normal .25342 -.690652 -.677332
outer loop
vertex -.0714992 .192072 -.391238
vertex .00760289 .226869 -.397124
vertex -.0628712 .183227 -.378991
endloop
endfacet
facet normal .937978 .00418757 .346669
outer loop
vertex -.0359152 .215691 -.394626
vertex -.0371856 .219154 -.391231
vertex -.0423452 .196037 -.376991
endloop
endfacet
facet normal -.236591 -.723175 .648878
outer loop
vertex -.0453788 .186764 -.368671
vertex -.0641383 .186681 -.375605
vertex -.0628712 .183227 -.378991
endloop
endfacet
facet normal -.830546 -.514324 .213693
outer loop
vertex -.0641383 .186681 -.375605
vertex -.0714992 .192072 -.391238
vertex -.0628712 .183227 -.378991
endloop
endfacet
facet normal .937978 .0041876 .346669
outer loop
vertex -.0485861 .250224 -.36076
vertex -.0423452 .196037 -.376991
vertex -.0371856 .219154 -.391231
endloop
endfacet
facet normal .495956 -.508402 .703957
outer loop
vertex -.0254719 .241471 -.343187
vertex -.0123341 .253845 -.343506
vertex -.0593168 .22475 -.331418
endloop
endfacet
facet normal -.253419 .690652 .677332
outer loop
vertex -.0780764 .224666 -.338351
vertex -.1014 .267889 -.391151
vertex -.106132 .253426 -.378173
endloop
endfacet
facet normal -.253419 .690651 .677333
outer loop
vertex -.0423452 .196037 -.376991
vertex -.0453788 .186764 -.368671
vertex -.0136791 .206395 -.376827
endloop
endfacet
facet normal .495955 -.508401 .703958
outer loop
vertex -.0136791 .206395 -.376827
vertex .00287104 .212406 -.384146
vertex .00160398 .215859 -.38076
endloop
endfacet
facet normal -.236592 -.723172 .648881
outer loop
vertex .00160398 .215859 -.38076
vertex -.0371856 .219154 -.391231
vertex -.0359152 .215691 -.394626
endloop
endfacet
facet normal -.830546 -.514324 .213694
outer loop
vertex -.0909267 .211987 -.418813
vertex -.0714992 .192072 -.391238
vertex -.0921938 .21544 -.415427
endloop
endfacet
facet normal -.937978 -.00418768 -.346669
outer loop
vertex -.0696766 .201774 -.360802
vertex -.0768093 .221213 -.341738
vertex -.0673457 .25014 -.367693
endloop
endfacet
facet normal .253419 -.690651 -.677333
outer loop
vertex -.0628712 .183227 -.378991
vertex .00760289 .226869 -.397124
vertex -.0441117 .183311 -.372058
endloop
endfacet
facet normal .937978 .00418769 .346669
outer loop
vertex .00160398 .215859 -.38076
vertex .00760289 .226869 -.397124
vertex -.011067 .250391 -.346893
endloop
endfacet
facet normal -.25342 .690652 .677332
outer loop
vertex -.0123341 .253845 -.343506
vertex -.0780764 .224666 -.338351
vertex -.0593168 .22475 -.331418
endloop
endfacet
facet normal -.236592 -.723175 .648877
outer loop
vertex -.0696766 .201774 -.360802
vertex -.0580498 .221297 -.334805
vertex -.0768093 .221213 -.341738
endloop
endfacet
facet normal -.236591 -.723176 .648877
outer loop
vertex -.0696766 .201774 -.360802
vertex -.0641383 .186681 -.375605
vertex -.0453788 .186764 -.368671
endloop
endfacet
facet normal .495957 -.508402 .703956
outer loop
vertex -.0136791 .206395 -.376827
vertex -.0453788 .186764 -.368671
vertex -.0441117 .183311 -.372058
endloop
endfacet
facet normal .236591 .723175 -.648877
outer loop
vertex -.00760228 .268308 -.356484
vertex .00760289 .226869 -.397124
vertex -.0754831 .268005 -.381572
endloop
endfacet
facet normal -.830546 -.514324 .213693
outer loop
vertex -.0714992 .192072 -.391238
vertex -.0641383 .186681 -.375605
vertex -.0921938 .21544 -.415427
endloop
endfacet
facet normal -.236591 -.723175 .648878
outer loop
vertex -.0921938 .21544 -.415427
vertex -.0546747 .215608 -.40156
vertex -.0595366 .228858 -.388565
endloop
endfacet
facet normal -.236591 -.723176 .648877
outer loop
vertex -.0673457 .25014 -.367693
vertex -.104865 .249973 -.38156
vertex -.0595366 .228858 -.388565
endloop
endfacet
facet normal -.937978 -.0041876 -.346669
outer loop
vertex -.104865 .249973 -.38156
vertex -.0861949 .22645 -.431791
vertex -.0921938 .21544 -.415427
endloop
endfacet
facet normal -.236591 -.723176 .648877
outer loop
vertex -.104865 .249973 -.38156
vertex -.0921938 .21544 -.415427
vertex -.0595366 .228858 -.388565
endloop
endfacet
facet normal -.937978 -.0041876 -.346668
outer loop
vertex -.0641383 .186681 -.375605
vertex -.0696766 .201774 -.360802
vertex -.0595366 .228858 -.388565
endloop
endfacet
facet normal -.937978 -.00418798 -.346669
outer loop
vertex -.104865 .249973 -.38156
vertex -.1014 .267889 -.391151
vertex -.0932185 .245592 -.413018
endloop
endfacet
facet normal .253419 -.690652 -.677332
outer loop
vertex -.0714992 .192072 -.391238
vertex -.0909267 .211987 -.418813
vertex .00760289 .226869 -.397124
endloop
endfacet
facet normal -.25342 .690652 .677332
outer loop
vertex -.0123341 .253845 -.343506
vertex -.0754831 .268005 -.381572
vertex -.0780764 .224666 -.338351
endloop
endfacet
facet normal -.253419 .690653 .677332
outer loop
vertex -.0359152 .215691 -.394626
vertex -.0136791 .206395 -.376827
vertex .00160398 .215859 -.38076
endloop
endfacet
facet normal -.236591 -.723175 .648877
outer loop
vertex -.0485861 .250224 -.36076
vertex .00160398 .215859 -.38076
vertex -.011067 .250391 -.346893
endloop
endfacet
facet normal .937978 .0041861 .34667
outer loop
vertex -.0565924 .225752 -.338802
vertex -.0580498 .221297 -.334805
vertex -.0453788 .186764 -.368671
endloop
endfacet
facet normal .937978 .00418736 .346669
outer loop
vertex -.0565924 .225752 -.338802
vertex -.0453788 .186764 -.368671
vertex -.0423452 .196037 -.376991
endloop
endfacet
facet normal .253419 -.690652 -.677332
outer loop
vertex -.0441117 .183311 -.372058
vertex .00760289 .226869 -.397124
vertex .00287104 .212406 -.384146
endloop
endfacet
facet normal .937978 .00418774 .346669
outer loop
vertex -.00760228 .268308 -.356484
vertex -.011067 .250391 -.346893
vertex .00760289 .226869 -.397124
endloop
endfacet
facet normal .236591 .723175 -.648877
outer loop
vertex -.0754831 .268005 -.381572
vertex .00760289 .226869 -.397124
vertex -.1014 .267889 -.391151
endloop
endfacet
facet normal .236591 .723176 -.648877
outer loop
vertex -.1014 .267889 -.391151
vertex .00760289 .226869 -.397124
vertex -.0932185 .245592 -.413018
endloop
endfacet
facet normal -.253419 .690652 .677332
outer loop
vertex -.0921938 .21544 -.415427
vertex -.0641383 .186681 -.375605
vertex -.0546747 .215608 -.40156
endloop
endfacet
facet normal -.830546 -.514324 .213693
outer loop
vertex -.106132 .253426 -.378173
vertex -.104865 .249973 -.38156
vertex -.0768093 .221213 -.341738
endloop
endfacet
facet normal -.830545 -.514325 .213693
outer loop
vertex -.0780764 .224666 -.338351
vertex -.106132 .253426 -.378173
vertex -.0768093 .221213 -.341738
endloop
endfacet
facet normal .495956 -.508403 .703956
outer loop
vertex -.0593168 .22475 -.331418
vertex -.0580498 .221297 -.334805
vertex -.0254719 .241471 -.343187
endloop
endfacet
facet normal .495955 -.508401 .703958
outer loop
vertex -.0123341 .253845 -.343506
vertex -.0254719 .241471 -.343187
vertex -.011067 .250391 -.346893
endloop
endfacet
facet normal -.37623 .0578118 .924721
outer loop
vertex -.0293924 .202128 -.306593
vertex -.0326592 .205583 -.308138
vertex -.0457976 .191174 -.312583
endloop
endfacet
facet normal .376227 -.0578067 -.924722
outer loop
vertex -.00681868 .198659 -.362076
vertex .0194579 .227478 -.353187
vertex .0057167 .185401 -.356147
endloop
endfacet
facet normal .656917 .720468 .22223
outer loop
vertex .0521273 .192924 -.337735
vertex .0478697 .190625 -.317696
vertex .0553942 .189469 -.33619
endloop
endfacet
facet normal .376228 -.0578073 -.924722
outer loop
vertex .00823042 .196347 -.399065
vertex -.00817483 .185393 -.405055
vertex .00496347 .199802 -.40061
endloop
endfacet
facet normal .653387 -.691074 .309035
outer loop
vertex .0310284 .143929 -.386513
vertex .0553942 .189469 -.33619
vertex .020633 .145526 -.360962
endloop
endfacet
facet normal -.653385 .691079 -.309028
outer loop
vertex .0446028 .19408 -.319241
vertex .0405562 .189642 -.32061
vertex .0159475 .175701 -.299756
endloop
endfacet
facet normal .656912 .720472 .22223
outer loop
vertex .0119334 .228634 -.334692
vertex .016191 .230933 -.354732
vertex .00866642 .232089 -.336238
endloop
endfacet
facet normal .37623 -.0578081 -.924721
outer loop
vertex .0408998 .161793 -.383613
vertex .0441667 .158338 -.382068
vertex .0310284 .143929 -.386513
endloop
endfacet
facet normal -.376227 .0578068 .924722
outer loop
vertex .00351044 .219396 -.337542
vertex .0183261 .165262 -.32813
vertex .0405562 .189642 -.32061
endloop
endfacet
facet normal -.653387 .691074 -.309034
outer loop
vertex .0521273 .192924 -.337735
vertex .0408998 .161793 -.383613
vertex .0258507 .164106 -.346624
endloop
endfacet
facet normal -.653387 .691074 -.309034
outer loop
vertex .0159475 .175701 -.299756
vertex .0183261 .165262 -.32813
vertex .00327703 .167574 -.291141
endloop
endfacet
facet normal .198478 .550323 .811019
outer loop
vertex .00866642 .232089 -.336238
vertex -.0326592 .205583 -.308138
vertex -.0159495 .21075 -.315733
endloop
endfacet
facet normal .376227 -.057809 -.924723
outer loop
vertex .00993396 .16624 -.39649
vertex -.00817483 .185393 -.405055
vertex .0408998 .161793 -.383613
endloop
endfacet
facet normal -.656915 -.720469 -.222231
outer loop
vertex -.0457976 .191174 -.312583
vertex -.00817483 .185393 -.405055
vertex .00993396 .16624 -.39649
endloop
endfacet
facet normal .656915 .720469 .222231
outer loop
vertex -.0293924 .202128 -.306593
vertex .0183261 .165262 -.32813
vertex -.0110677 .196351 -.342032
endloop
endfacet
facet normal -.653387 .691074 -.309034
outer loop
vertex .00496347 .199802 -.40061
vertex -.00817483 .185393 -.405055
vertex -.0457976 .191174 -.312583
endloop
endfacet
facet normal -.376228 .0578073 .924722
outer loop
vertex .00327703 .167574 -.291141
vertex -.00659434 .149709 -.29404
vertex .00654396 .164119 -.289596
endloop
endfacet
facet normal .656916 .720468 .22223
outer loop
vertex .0478697 .190625 -.317696
vertex .0521273 .192924 -.337735
vertex .0446028 .19408 -.319241
endloop
endfacet
facet normal .656915 .720469 .222231
outer loop
vertex .0337372 .212375 -.346433
vertex .0119334 .228634 -.334692
vertex .0446028 .19408 -.319241
endloop
endfacet
facet normal -.653387 .691074 -.309034
outer loop
vertex .0183261 .165262 -.32813
vertex .0159475 .175701 -.299756
vertex .0405562 .189642 -.32061
endloop
endfacet
facet normal .653387 -.691074 .309034
outer loop
vertex -.0143432 .199815 -.343582
vertex .00351044 .219396 -.337542
vertex -.0159495 .21075 -.315733
endloop
endfacet
facet normal .653387 -.691074 .309034
outer loop
vertex .020633 .145526 -.360962
vertex .00654396 .164119 -.289596
vertex -.00659434 .149709 -.29404
endloop
endfacet
facet normal .376228 -.0578068 -.924722
outer loop
vertex .0258507 .164106 -.346624
vertex .0057167 .185401 -.356147
vertex .0337372 .212375 -.346433
endloop
endfacet
facet normal -.653387 .691074 -.309034
outer loop
vertex .0127382 .22136 -.368841
vertex -.0457976 .191174 -.312583
vertex .016191 .230933 -.354732
endloop
endfacet
facet normal -.376227 .0578069 .924722
outer loop
vertex -.0143432 .199815 -.343582
vertex -.0110677 .196351 -.342032
vertex .00351044 .219396 -.337542
endloop
endfacet
facet normal .656916 .720468 .222231
outer loop
vertex .0119334 .228634 -.334692
vertex .0194579 .227478 -.353187
vertex .016191 .230933 -.354732
endloop
endfacet
facet normal .730542 .468574 -.496737
outer loop
vertex .0194579 .227478 -.353187
vertex .0127382 .22136 -.368841
vertex .016191 .230933 -.354732
endloop
endfacet
facet normal -.376227 .0578068 .924722
outer loop
vertex .0183261 .165262 -.32813
vertex .00351044 .219396 -.337542
vertex -.0110677 .196351 -.342032
endloop
endfacet
facet normal .198476 .550325 .811018
outer loop
vertex .0159475 .175701 -.299756
vertex .00654396 .164119 -.289596
vertex .0478697 .190625 -.317696
endloop
endfacet
facet normal .653387 -.691074 .309035
outer loop
vertex .0553942 .189469 -.33619
vertex .0310284 .143929 -.386513
vertex .0441667 .158338 -.382068
endloop
endfacet
facet normal .653388 -.691073 .309035
outer loop
vertex .00351044 .219396 -.337542
vertex .0119334 .228634 -.334692
vertex -.0159495 .21075 -.315733
endloop
endfacet
facet normal .198477 .550324 .811018
outer loop
vertex -.0159495 .21075 -.315733
vertex -.0326592 .205583 -.308138
vertex -.0293924 .202128 -.306593
endloop
endfacet
facet normal .656919 .720466 .222233
outer loop
vertex -.0293924 .202128 -.306593
vertex -.0110677 .196351 -.342032
vertex -.0143432 .199815 -.343582
endloop
endfacet
facet normal .730542 .468573 -.496737
outer loop
vertex .00496347 .199802 -.40061
vertex .0127382 .22136 -.368841
vertex .00823042 .196347 -.399065
endloop
endfacet
facet normal .376227 -.0578068 -.924722
outer loop
vertex .0337372 .212375 -.346433
vertex .0521273 .192924 -.337735
vertex .0258507 .164106 -.346624
endloop
endfacet
facet normal -.653387 .691073 -.309035
outer loop
vertex .016191 .230933 -.354732
vertex -.0457976 .191174 -.312583
vertex .00866642 .232089 -.336238
endloop
endfacet
facet normal -.376227 .0578067 .924722
outer loop
vertex -.0293924 .202128 -.306593
vertex -.0457976 .191174 -.312583
vertex .00327703 .167574 -.291141
endloop
endfacet
facet normal .653387 -.691073 .309034
outer loop
vertex .00654396 .164119 -.289596
vertex .0553942 .189469 -.33619
vertex .0478697 .190625 -.317696
endloop
endfacet
facet normal .656916 .720469 .22223
outer loop
vertex .0337372 .212375 -.346433
vertex .0446028 .19408 -.319241
vertex .0521273 .192924 -.337735
endloop
endfacet
facet normal .656915 .720469 .22223
outer loop
vertex .0337372 .212375 -.346433
vertex .0194579 .227478 -.353187
vertex .0119334 .228634 -.334692
endloop
endfacet
facet normal .198475 .550326 .811018
outer loop
vertex -.0159495 .21075 -.315733
vertex .0119334 .228634 -.334692
vertex .00866642 .232089 -.336238
endloop
endfacet
facet normal -.656915 -.720469 -.222231
outer loop
vertex -.00659434 .149709 -.29404
vertex -.0457976 .191174 -.312583
vertex .020633 .145526 -.360962
endloop
endfacet
facet normal .730542 .468574 -.496737
outer loop
vertex .0127382 .22136 -.368841
vertex .0194579 .227478 -.353187
vertex .00823042 .196347 -.399065
endloop
endfacet
facet normal .656916 .720469 .22223
outer loop
vertex .00823042 .196347 -.399065
vertex -.00681868 .198659 -.362076
vertex .0057167 .185401 -.356147
endloop
endfacet
facet normal .656915 .720469 .22223
outer loop
vertex .0258507 .164106 -.346624
vertex .0408998 .161793 -.383613
vertex .0057167 .185401 -.356147
endloop
endfacet
facet normal .376227 -.0578068 -.924722
outer loop
vertex .0408998 .161793 -.383613
vertex -.00817483 .185393 -.405055
vertex .00823042 .196347 -.399065
endloop
endfacet
facet normal .656915 .720469 .222231
outer loop
vertex .0408998 .161793 -.383613
vertex .00823042 .196347 -.399065
vertex .0057167 .185401 -.356147
endloop
endfacet
facet normal .376228 -.0578068 -.924722
outer loop
vertex .0194579 .227478 -.353187
vertex .0337372 .212375 -.346433
vertex .0057167 .185401 -.356147
endloop
endfacet
facet normal .376227 -.0578065 -.924723
outer loop
vertex .0408998 .161793 -.383613
vertex .0310284 .143929 -.386513
vertex .00993396 .16624 -.39649
endloop
endfacet
facet normal -.653387 .691073 -.309035
outer loop
vertex .0127382 .22136 -.368841
vertex .00496347 .199802 -.40061
vertex -.0457976 .191174 -.312583
endloop
endfacet
facet normal .653387 -.691074 .309034
outer loop
vertex .00654396 .164119 -.289596
vertex .020633 .145526 -.360962
vertex .0553942 .189469 -.33619
endloop
endfacet
facet normal .653386 -.691074 .309035
outer loop
vertex -.0143432 .199815 -.343582
vertex -.0159495 .21075 -.315733
vertex -.0293924 .202128 -.306593
endloop
endfacet
facet normal .656915 .720469 .222231
outer loop
vertex .0183261 .165262 -.32813
vertex -.0293924 .202128 -.306593
vertex .00327703 .167574 -.291141
endloop
endfacet
facet normal -.376226 .0578083 .924723
outer loop
vertex .0405562 .189642 -.32061
vertex .0446028 .19408 -.319241
vertex .0119334 .228634 -.334692
endloop
endfacet
facet normal -.376227 .0578071 .924722
outer loop
vertex .0405562 .189642 -.32061
vertex .0119334 .228634 -.334692
vertex .00351044 .219396 -.337542
endloop
endfacet
facet normal -.653387 .691073 -.309035
outer loop
vertex .00866642 .232089 -.336238
vertex -.0457976 .191174 -.312583
vertex -.0326592 .205583 -.308138
endloop
endfacet
facet normal -.376227 .0578067 .924722
outer loop
vertex -.00659434 .149709 -.29404
vertex .00327703 .167574 -.291141
vertex -.0457976 .191174 -.312583
endloop
endfacet
facet normal -.656915 -.720469 -.222231
outer loop
vertex .020633 .145526 -.360962
vertex -.0457976 .191174 -.312583
vertex .0310284 .143929 -.386513
endloop
endfacet
facet normal -.656915 -.720469 -.222231
outer loop
vertex .0310284 .143929 -.386513
vertex -.0457976 .191174 -.312583
vertex .00993396 .16624 -.39649
endloop
endfacet
facet normal .653387 -.691074 .309035
outer loop
vertex .00823042 .196347 -.399065
vertex .0194579 .227478 -.353187
vertex -.00681868 .198659 -.362076
endloop
endfacet
facet normal .730541 .468574 -.496737
outer loop
vertex .0441667 .158338 -.382068
vertex .0408998 .161793 -.383613
vertex .0521273 .192924 -.337735
endloop
endfacet
facet normal .730541 .468574 -.496737
outer loop
vertex .0553942 .189469 -.33619
vertex .0441667 .158338 -.382068
vertex .0521273 .192924 -.337735
endloop
endfacet
facet normal .198475 .550326 .811018
outer loop
vertex .0478697 .190625 -.317696
vertex .0446028 .19408 -.319241
vertex .0159475 .175701 -.299756
endloop
endfacet
facet normal .198477 .550324 .811018
outer loop
vertex .00654396 .164119 -.289596
vertex .0159475 .175701 -.299756
vertex .00327703 .167574 -.291141
endloop
endfacet
facet normal .399829 -.0642066 -.914338
outer loop
vertex .0007178 .111665 -.43883
vertex -.00253143 .108049 -.439997
vertex -.0154594 .121799 -.446616
endloop
endfacet
facet normal -.399832 .0642014 .914337
outer loop
vertex -.0232721 .115517 -.38397
vertex .0025838 .0880176 -.370732
vertex -.0108045 .129395 -.379492
endloop
endfacet
facet normal .646398 -.687495 .330938
outer loop
vertex .0350763 .124185 -.359063
vertex .0463222 .126517 -.376183
vertex .0383256 .127802 -.357896
endloop
endfacet
facet normal -.399831 .0642019 .914338
outer loop
vertex -.0392653 .118086 -.347396
vertex -.0554425 .128219 -.355182
vertex -.0425146 .114469 -.348563
endloop
endfacet
facet normal .649851 .723344 .233383
outer loop
vertex -.0164515 .171619 -.341179
vertex .0383256 .127802 -.357896
vertex -.00540385 .169845 -.366443
endloop
endfacet
facet normal -.649844 -.723349 -.233386
outer loop
vertex .043073 .122901 -.37735
vertex .0390912 .127136 -.379389
vertex .0362342 .140188 -.411889
endloop
endfacet
facet normal .646395 -.6875 .330935
outer loop
vertex .0105804 .0867335 -.389019
vertex -.000665439 .0844009 -.371899
vertex .00733116 .0831168 -.390186
endloop
endfacet
facet normal -.399829 .0642029 .914338
outer loop
vertex -.0067728 .154253 -.335727
vertex -.00352353 .157869 -.33456
vertex -.0164515 .171619 -.341179
endloop
endfacet
facet normal .399831 -.0642015 -.914337
outer loop
vertex .00229236 .0955486 -.393262
vertex .0172171 .150401 -.390587
vertex .0390912 .127136 -.379389
endloop
endfacet
facet normal -.649851 -.723344 -.233383
outer loop
vertex .0350763 .124185 -.359063
vertex -.0067728 .154253 -.335727
vertex .00922047 .151685 -.372301
endloop
endfacet
facet normal -.64985 -.723345 -.233383
outer loop
vertex .0362342 .140188 -.411889
vertex .0172171 .150401 -.390587
vertex .0332103 .147833 -.427161
endloop
endfacet
facet normal .739796 -.53153 -.412526
outer loop
vertex .00733116 .0831168 -.390186
vertex -.00253143 .108049 -.439997
vertex .00392603 .103555 -.422627
endloop
endfacet
facet normal -.399832 .0642036 .914337
outer loop
vertex -.0374317 .148266 -.348714
vertex -.0554425 .128219 -.355182
vertex -.0067728 .154253 -.335727
endloop
endfacet
facet normal -.646397 .687497 -.330937
outer loop
vertex -.0154594 .121799 -.446616
vertex -.0554425 .128219 -.355182
vertex -.0374317 .148266 -.348714
endloop
endfacet
facet normal .646397 -.687497 .330937
outer loop
vertex .0007178 .111665 -.43883
vertex .0172171 .150401 -.390587
vertex -.0120176 .11786 -.401087
endloop
endfacet
facet normal -.649851 -.723344 -.233383
outer loop
vertex -.0425146 .114469 -.348563
vertex -.0554425 .128219 -.355182
vertex -.0154594 .121799 -.446616
endloop
endfacet
facet normal .39983 -.064202 -.914338
outer loop
vertex .0332103 .147833 -.427161
vertex .0235317 .165199 -.432613
vertex .0364596 .151449 -.425994
endloop
endfacet
facet normal .646397 -.687496 .330938
outer loop
vertex .0463222 .126517 -.376183
vertex .0350763 .124185 -.359063
vertex .043073 .122901 -.37735
endloop
endfacet
facet normal .646397 -.687497 .330937
outer loop
vertex .0167858 .103826 -.365632
vertex .0105804 .0867335 -.389019
vertex .043073 .122901 -.37735
endloop
endfacet
facet normal -.649851 -.723344 -.233383
outer loop
vertex .0172171 .150401 -.390587
vertex .0362342 .140188 -.411889
vertex .0390912 .127136 -.379389
endloop
endfacet
facet normal .649851 .723344 .233383
outer loop
vertex -.0152754 .114233 -.402257
vertex .00229236 .0955486 -.393262
vertex .00392603 .103555 -.422627
endloop
endfacet
facet normal .649851 .723344 .233383
outer loop
vertex -.00540385 .169845 -.366443
vertex .0364596 .151449 -.425994
vertex .0235317 .165199 -.432613
endloop
endfacet
facet normal -.399831 .0642015 .914338
outer loop
vertex .00922047 .151685 -.372301
vertex -.0108045 .129395 -.379492
vertex .0167858 .103826 -.365632
endloop
endfacet
facet normal -.64985 -.723344 -.233383
outer loop
vertex -.0135353 .0936477 -.364723
vertex -.0154594 .121799 -.446616
vertex -.000665439 .0844009 -.371899
endloop
endfacet
facet normal .399831 -.0642015 -.914337
outer loop
vertex -.0152754 .114233 -.402257
vertex -.0120176 .11786 -.401087
vertex .00229236 .0955486 -.393262
endloop
endfacet
facet normal .646397 -.687496 .330937
outer loop
vertex .0105804 .0867335 -.389019
vertex .0025838 .0880176 -.370732
vertex -.000665439 .0844009 -.371899
endloop
endfacet
facet normal .174348 -.440737 .880542
outer loop
vertex .0025838 .0880176 -.370732
vertex -.0135353 .0936477 -.364723
vertex -.000665439 .0844009 -.371899
endloop
endfacet
facet normal .399831 -.0642015 -.914337
outer loop
vertex .0172171 .150401 -.390587
vertex .00229236 .0955486 -.393262
vertex -.0120176 .11786 -.401087
endloop
endfacet
facet normal .739795 -.531531 -.412527
outer loop
vertex .0362342 .140188 -.411889
vertex .0364596 .151449 -.425994
vertex .0463222 .126517 -.376183
endloop
endfacet
facet normal .649851 .723344 .233383
outer loop
vertex .0383256 .127802 -.357896
vertex -.0164515 .171619 -.341179
vertex -.00352353 .157869 -.33456
endloop
endfacet
facet normal .649851 .723343 .233383
outer loop
vertex .00229236 .0955486 -.393262
vertex .0105804 .0867335 -.389019
vertex .00392603 .103555 -.422627
endloop
endfacet
facet normal .739796 -.531531 -.412526
outer loop
vertex .00392603 .103555 -.422627
vertex -.00253143 .108049 -.439997
vertex .0007178 .111665 -.43883
endloop
endfacet
facet normal .646401 -.687493 .330938
outer loop
vertex .0007178 .111665 -.43883
vertex -.0120176 .11786 -.401087
vertex -.0152754 .114233 -.402257
endloop
endfacet
facet normal .174348 -.440737 .880542
outer loop
vertex -.0425146 .114469 -.348563
vertex -.0135353 .0936477 -.364723
vertex -.0392653 .118086 -.347396
endloop
endfacet
facet normal -.399831 .0642014 .914337
outer loop
vertex .0167858 .103826 -.365632
vertex .0350763 .124185 -.359063
vertex .00922047 .151685 -.372301
endloop
endfacet
facet normal -.649851 -.723343 -.233383
outer loop
vertex -.000665439 .0844009 -.371899
vertex -.0154594 .121799 -.446616
vertex .00733116 .0831168 -.390186
endloop
endfacet
facet normal .399831 -.0642014 -.914337
outer loop
vertex .0007178 .111665 -.43883
vertex -.0154594 .121799 -.446616
vertex .0332103 .147833 -.427161
endloop
endfacet
facet normal .649851 .723344 .233383
outer loop
vertex .0364596 .151449 -.425994
vertex .0383256 .127802 -.357896
vertex .0463222 .126517 -.376183
endloop
endfacet
facet normal .646397 -.687496 .330938
outer loop
vertex .0167858 .103826 -.365632
vertex .043073 .122901 -.37735
vertex .0350763 .124185 -.359063
endloop
endfacet
facet normal .646397 -.687497 .330937
outer loop
vertex .0167858 .103826 -.365632
vertex .0025838 .0880176 -.370732
vertex .0105804 .0867335 -.389019
endloop
endfacet
facet normal .739794 -.531532 -.412527
outer loop
vertex .00392603 .103555 -.422627
vertex .0105804 .0867335 -.389019
vertex .00733116 .0831168 -.390186
endloop
endfacet
facet normal -.646397 .687497 -.330937
outer loop
vertex .0235317 .165199 -.432613
vertex -.0154594 .121799 -.446616
vertex -.00540385 .169845 -.366443
endloop
endfacet
facet normal .174348 -.440737 .880542
outer loop
vertex -.0135353 .0936477 -.364723
vertex .0025838 .0880176 -.370732
vertex -.0392653 .118086 -.347396
endloop
endfacet
facet normal .646397 -.687496 .330938
outer loop
vertex -.0392653 .118086 -.347396
vertex -.0232721 .115517 -.38397
vertex -.0108045 .129395 -.379492
endloop
endfacet
facet normal .646397 -.687497 .330937
outer loop
vertex .00922047 .151685 -.372301
vertex -.0067728 .154253 -.335727
vertex -.0108045 .129395 -.379492
endloop
endfacet
facet normal -.399831 .0642015 .914337
outer loop
vertex -.0067728 .154253 -.335727
vertex -.0554425 .128219 -.355182
vertex -.0392653 .118086 -.347396
endloop
endfacet
facet normal .646397 -.687497 .330937
outer loop
vertex -.0067728 .154253 -.335727
vertex -.0392653 .118086 -.347396
vertex -.0108045 .129395 -.379492
endloop
endfacet
facet normal -.399831 .0642015 .914338
outer loop
vertex .0025838 .0880176 -.370732
vertex .0167858 .103826 -.365632
vertex -.0108045 .129395 -.379492
endloop
endfacet
facet normal -.399832 .0642011 .914337
outer loop
vertex -.0067728 .154253 -.335727
vertex -.0164515 .171619 -.341179
vertex -.0374317 .148266 -.348714
endloop
endfacet
facet normal -.649851 -.723344 -.233383
outer loop
vertex -.0135353 .0936477 -.364723
vertex -.0425146 .114469 -.348563
vertex -.0154594 .121799 -.446616
endloop
endfacet
facet normal .649851 .723344 .233383
outer loop
vertex .0364596 .151449 -.425994
vertex -.00540385 .169845 -.366443
vertex .0383256 .127802 -.357896
endloop
endfacet
facet normal .64985 .723345 .233382
outer loop
vertex -.0152754 .114233 -.402257
vertex .00392603 .103555 -.422627
vertex .0007178 .111665 -.43883
endloop
endfacet
facet normal .646397 -.687497 .330937
outer loop
vertex .0172171 .150401 -.390587
vertex .0007178 .111665 -.43883
vertex .0332103 .147833 -.427161
endloop
endfacet
facet normal .399833 -.0642029 -.914337
outer loop
vertex .0390912 .127136 -.379389
vertex .043073 .122901 -.37735
vertex .0105804 .0867335 -.389019
endloop
endfacet
facet normal .399832 -.0642017 -.914337
outer loop
vertex .0390912 .127136 -.379389
vertex .0105804 .0867335 -.389019
vertex .00229236 .0955486 -.393262
endloop
endfacet
facet normal -.649851 -.723344 -.233383
outer loop
vertex .00733116 .0831168 -.390186
vertex -.0154594 .121799 -.446616
vertex -.00253143 .108049 -.439997
endloop
endfacet
facet normal .399831 -.0642014 -.914337
outer loop
vertex .0235317 .165199 -.432613
vertex .0332103 .147833 -.427161
vertex -.0154594 .121799 -.446616
endloop
endfacet
facet normal -.646397 .687497 -.330937
outer loop
vertex -.00540385 .169845 -.366443
vertex -.0154594 .121799 -.446616
vertex -.0164515 .171619 -.341179
endloop
endfacet
facet normal -.646397 .687497 -.330937
outer loop
vertex -.0164515 .171619 -.341179
vertex -.0154594 .121799 -.446616
vertex -.0374317 .148266 -.348714
endloop
endfacet
facet normal .649851 .723344 .233383
outer loop
vertex -.0392653 .118086 -.347396
vertex .0025838 .0880176 -.370732
vertex -.0232721 .115517 -.38397
endloop
endfacet
facet normal .174347 -.440737 .880542
outer loop
vertex -.00352353 .157869 -.33456
vertex -.0067728 .154253 -.335727
vertex .0350763 .124185 -.359063
endloop
endfacet
facet normal .174347 -.440738 .880542
outer loop
vertex .0383256 .127802 -.357896
vertex -.00352353 .157869 -.33456
vertex .0350763 .124185 -.359063
endloop
endfacet
facet normal .739794 -.531532 -.412527
outer loop
vertex .0463222 .126517 -.376183
vertex .043073 .122901 -.37735
vertex .0362342 .140188 -.411889
endloop
endfacet
facet normal .739796 -.531531 -.412526
outer loop
vertex .0364596 .151449 -.425994
vertex .0362342 .140188 -.411889
vertex .0332103 .147833 -.427161
endloop
endfacet
facet normal -.207869 .374559 -.903601
outer loop
vertex -.085364 .0940517 -.429342
vertex -.0886171 .090338 -.430133
vertex -.103225 .101438 -.422171
endloop
endfacet
facet normal .207866 -.374565 .9036
outer loop
vertex -.072892 .0715778 -.375126
vertex -.043676 .0493773 -.391049
vertex -.0604097 .0858275 -.37209
endloop
endfacet
facet normal .730402 -.555013 -.398089
outer loop
vertex -.011145 .0865143 -.383138
vertex -.0120492 .0977192 -.400419
vertex -.00789193 .090228 -.382347
endloop
endfacet
facet normal .207867 -.374564 .9036
outer loop
vertex -.0645774 .0565953 -.338982
vertex -.0824385 .0639818 -.331811
vertex -.0678305 .0528816 -.339773
endloop
endfacet
facet normal .650621 .742739 .158213
outer loop
vertex -.0434013 .108546 -.322318
vertex -.00789193 .090228 -.382347
vertex -.0491448 .118896 -.347285
endloop
endfacet
facet normal -.650617 -.742741 -.158221
outer loop
vertex -.0153023 .0940056 -.40121
vertex -.0198016 .0974244 -.398758
vertex -.0413261 .119788 -.415231
endloop
endfacet
facet normal .730397 -.555016 -.398092
outer loop
vertex -.0478334 .0568686 -.409121
vertex -.0469291 .0456636 -.39184
vertex -.0510865 .0531549 -.409912
endloop
endfacet
facet normal .207869 -.374563 .9036
outer loop
vertex -.0320464 .0937322 -.331071
vertex -.0287932 .0974459 -.33028
vertex -.0434013 .108546 -.322318
endloop
endfacet
facet normal -.207866 .374565 -.9036
outer loop
vertex -.0571985 .063985 -.404017
vertex -.0445183 .116206 -.385287
vertex -.0198016 .0974244 -.398758
endloop
endfacet
facet normal -.650621 -.742739 -.158213
outer loop
vertex -.011145 .0865143 -.383138
vertex -.0320464 .0937322 -.331071
vertex -.040361 .108715 -.367215
endloop
endfacet
facet normal -.65062 -.74274 -.158213
outer loop
vertex -.0413261 .119788 -.415231
vertex -.0445183 .116206 -.385287
vertex -.052833 .131189 -.421431
endloop
endfacet
facet normal .369489 -.127596 -.920433
outer loop
vertex -.0510865 .0531549 -.409912
vertex -.0886171 .090338 -.430133
vertex -.0731556 .0819564 -.422764
endloop
endfacet
facet normal .207865 -.374563 .903601
outer loop
vertex -.0644064 .0845671 -.327426
vertex -.0824385 .0639818 -.331811
vertex -.0320464 .0937322 -.331071
endloop
endfacet
facet normal -.7304 .555013 .39809
outer loop
vertex -.103225 .101438 -.422171
vertex -.0824385 .0639818 -.331811
vertex -.0644064 .0845671 -.327426
endloop
endfacet
facet normal .7304 -.555014 -.39809
outer loop
vertex -.085364 .0940517 -.429342
vertex -.0445183 .116206 -.385287
vertex -.0737877 .0827926 -.392404
endloop
endfacet
facet normal -.650621 -.742739 -.158213
outer loop
vertex -.0678305 .0528816 -.339773
vertex -.0824385 .0639818 -.331811
vertex -.103225 .101438 -.422171
endloop
endfacet
facet normal -.207867 .374564 -.9036
outer loop
vertex -.052833 .131189 -.421431
vertex -.0641879 .146003 -.412678
vertex -.0495799 .134902 -.42064
endloop
endfacet
facet normal .730401 -.555013 -.398089
outer loop
vertex -.0120492 .0977192 -.400419
vertex -.011145 .0865143 -.383138
vertex -.0153023 .0940056 -.40121
endloop
endfacet
facet normal .7304 -.555014 -.39809
outer loop
vertex -.0294572 .0656093 -.387591
vertex -.0478334 .0568686 -.409121
vertex -.0153023 .0940056 -.40121
endloop
endfacet
facet normal -.650621 -.742739 -.158213
outer loop
vertex -.0445183 .116206 -.385287
vertex -.0413261 .119788 -.415231
vertex -.0198016 .0974244 -.398758
endloop
endfacet
facet normal .650621 .742739 .158213
outer loop
vertex -.0770494 .0790691 -.393198
vertex -.0571985 .063985 -.404017
vertex -.0731556 .0819564 -.422764
endloop
endfacet
facet normal .650621 .742739 .158213
outer loop
vertex -.0491448 .118896 -.347285
vertex -.0495799 .134902 -.42064
vertex -.0641879 .146003 -.412678
endloop
endfacet
facet normal .207867 -.374565 .9036
outer loop
vertex -.040361 .108715 -.367215
vertex -.0604097 .0858275 -.37209
vertex -.0294572 .0656093 -.387591
endloop
endfacet
facet normal -.65062 -.742739 -.158214
outer loop
vertex -.0533569 .0478834 -.375828
vertex -.103225 .101438 -.422171
vertex -.0469291 .0456636 -.39184
endloop
endfacet
facet normal -.207866 .374565 -.9036
outer loop
vertex -.0770494 .0790691 -.393198
vertex -.0737877 .0827926 -.392404
vertex -.0571985 .063985 -.404017
endloop
endfacet
facet normal .730401 -.555013 -.39809
outer loop
vertex -.0478334 .0568686 -.409121
vertex -.043676 .0493773 -.391049
vertex -.0469291 .0456636 -.39184
endloop
endfacet
facet normal .663454 -.657312 .357449
outer loop
vertex -.043676 .0493773 -.391049
vertex -.0533569 .0478834 -.375828
vertex -.0469291 .0456636 -.39184
endloop
endfacet
facet normal -.207866 .374565 -.9036
outer loop
vertex -.0445183 .116206 -.385287
vertex -.0571985 .063985 -.404017
vertex -.0737877 .0827926 -.392404
endloop
endfacet
facet normal .369487 -.127597 -.920434
outer loop
vertex -.0413261 .119788 -.415231
vertex -.0495799 .134902 -.42064
vertex -.0120492 .0977192 -.400419
endloop
endfacet
facet normal .650621 .742739 .158213
outer loop
vertex -.00789193 .090228 -.382347
vertex -.0434013 .108546 -.322318
vertex -.0287932 .0974459 -.33028
endloop
endfacet
facet normal .650621 .742739 .158213
outer loop
vertex -.0571985 .063985 -.404017
vertex -.0478334 .0568686 -.409121
vertex -.0731556 .0819564 -.422764
endloop
endfacet
facet normal .369488 -.127597 -.920433
outer loop
vertex -.0731556 .0819564 -.422764
vertex -.0886171 .090338 -.430133
vertex -.085364 .0940517 -.429342
endloop
endfacet
facet normal .730403 -.555009 -.39809
outer loop
vertex -.085364 .0940517 -.429342
vertex -.0737877 .0827926 -.392404
vertex -.0770494 .0790691 -.393198
endloop
endfacet
facet normal .663454 -.657311 .35745
outer loop
vertex -.0678305 .0528816 -.339773
vertex -.0533569 .0478834 -.375828
vertex -.0645774 .0565953 -.338982
endloop
endfacet
facet normal .207866 -.374565 .9036
outer loop
vertex -.0294572 .0656093 -.387591
vertex -.011145 .0865143 -.383138
vertex -.040361 .108715 -.367215
endloop
endfacet
facet normal -.650621 -.742739 -.158212
outer loop
vertex -.0469291 .0456636 -.39184
vertex -.103225 .101438 -.422171
vertex -.0510865 .0531549 -.409912
endloop
endfacet
facet normal -.207866 .374565 -.9036
outer loop
vertex -.085364 .0940517 -.429342
vertex -.103225 .101438 -.422171
vertex -.052833 .131189 -.421431
endloop
endfacet
facet normal .650621 .742739 .158213
outer loop
vertex -.0495799 .134902 -.42064
vertex -.00789193 .090228 -.382347
vertex -.0120492 .0977192 -.400419
endloop
endfacet
facet normal .730401 -.555014 -.398089
outer loop
vertex -.0294572 .0656093 -.387591
vertex -.0153023 .0940056 -.40121
vertex -.011145 .0865143 -.383138
endloop
endfacet
facet normal .7304 -.555014 -.39809
outer loop
vertex -.0294572 .0656093 -.387591
vertex -.043676 .0493773 -.391049
vertex -.0478334 .0568686 -.409121
endloop
endfacet
facet normal .369486 -.127598 -.920434
outer loop
vertex -.0731556 .0819564 -.422764
vertex -.0478334 .0568686 -.409121
vertex -.0510865 .0531549 -.409912
endloop
endfacet
facet normal -.7304 .555014 .39809
outer loop
vertex -.0641879 .146003 -.412678
vertex -.103225 .101438 -.422171
vertex -.0491448 .118896 -.347285
endloop
endfacet
facet normal .663454 -.657312 .357449
outer loop
vertex -.0533569 .0478834 -.375828
vertex -.043676 .0493773 -.391049
vertex -.0645774 .0565953 -.338982
endloop
endfacet
facet normal .730401 -.555013 -.398089
outer loop
vertex -.0645774 .0565953 -.338982
vertex -.072892 .0715778 -.375126
vertex -.0604097 .0858275 -.37209
endloop
endfacet
facet normal .7304 -.555014 -.39809
outer loop
vertex -.040361 .108715 -.367215
vertex -.0320464 .0937322 -.331071
vertex -.0604097 .0858275 -.37209
endloop
endfacet
facet normal .207866 -.374565 .9036
outer loop
vertex -.0320464 .0937322 -.331071
vertex -.0824385 .0639818 -.331811
vertex -.0645774 .0565953 -.338982
endloop
endfacet
facet normal .7304 -.555014 -.39809
outer loop
vertex -.0320464 .0937322 -.331071
vertex -.0645774 .0565953 -.338982
vertex -.0604097 .0858275 -.37209
endloop
endfacet
facet normal .207867 -.374565 .9036
outer loop
vertex -.043676 .0493773 -.391049
vertex -.0294572 .0656093 -.387591
vertex -.0604097 .0858275 -.37209
endloop
endfacet
facet normal .207866 -.374565 .9036
outer loop
vertex -.0320464 .0937322 -.331071
vertex -.0434013 .108546 -.322318
vertex -.0644064 .0845671 -.327426
endloop
endfacet
facet normal -.650621 -.742739 -.158213
outer loop
vertex -.0533569 .0478834 -.375828
vertex -.0678305 .0528816 -.339773
vertex -.103225 .101438 -.422171
endloop
endfacet
facet normal .650621 .742739 .158213
outer loop
vertex -.0495799 .134902 -.42064
vertex -.0491448 .118896 -.347285
vertex -.00789193 .090228 -.382347
endloop
endfacet
facet normal .65062 .74274 .158213
outer loop
vertex -.0770494 .0790691 -.393198
vertex -.0731556 .0819564 -.422764
vertex -.085364 .0940517 -.429342
endloop
endfacet
facet normal .7304 -.555014 -.39809
outer loop
vertex -.0445183 .116206 -.385287
vertex -.085364 .0940517 -.429342
vertex -.052833 .131189 -.421431
endloop
endfacet
facet normal -.207865 .374563 -.903601
outer loop
vertex -.0198016 .0974244 -.398758
vertex -.0153023 .0940056 -.40121
vertex -.0478334 .0568686 -.409121
endloop
endfacet
facet normal -.207866 .374564 -.9036
outer loop
vertex -.0198016 .0974244 -.398758
vertex -.0478334 .0568686 -.409121
vertex -.0571985 .063985 -.404017
endloop
endfacet
facet normal -.650621 -.742739 -.158213
outer loop
vertex -.0510865 .0531549 -.409912
vertex -.103225 .101438 -.422171
vertex -.0886171 .090338 -.430133
endloop
endfacet
facet normal -.207866 .374565 -.9036
outer loop
vertex -.0641879 .146003 -.412678
vertex -.052833 .131189 -.421431
vertex -.103225 .101438 -.422171
endloop
endfacet
facet normal -.7304 .555014 .39809
outer loop
vertex -.0491448 .118896 -.347285
vertex -.103225 .101438 -.422171
vertex -.0434013 .108546 -.322318
endloop
endfacet
facet normal -.7304 .555014 .39809
outer loop
vertex -.0434013 .108546 -.322318
vertex -.103225 .101438 -.422171
vertex -.0644064 .0845671 -.327426
endloop
endfacet
facet normal .650621 .742739 .158213
outer loop
vertex -.0645774 .0565953 -.338982
vertex -.043676 .0493773 -.391049
vertex -.072892 .0715778 -.375126
endloop
endfacet
facet normal .663454 -.657312 .357449
outer loop
vertex -.0287932 .0974459 -.33028
vertex -.0320464 .0937322 -.331071
vertex -.011145 .0865143 -.383138
endloop
endfacet
facet normal .663453 -.657312 .357449
outer loop
vertex -.00789193 .090228 -.382347
vertex -.0287932 .0974459 -.33028
vertex -.011145 .0865143 -.383138
endloop
endfacet
facet normal .369486 -.127598 -.920434
outer loop
vertex -.0120492 .0977192 -.400419
vertex -.0153023 .0940056 -.40121
vertex -.0413261 .119788 -.415231
endloop
endfacet
facet normal .369488 -.127596 -.920434
outer loop
vertex -.0495799 .134902 -.42064
vertex -.0413261 .119788 -.415231
vertex -.052833 .131189 -.421431
endloop
endfacet
facet normal -.675181 .493942 .547861
outer loop
vertex -.163209 .193526 -.377492
vertex -.1665 .189833 -.378219
vertex -.159846 .180655 -.361743
endloop
endfacet
facet normal .675176 -.493946 -.547863
outer loop
vertex -.122698 .163889 -.410364
vertex -.136007 .182246 -.443316
vertex -.110067 .178058 -.407572
endloop
endfacet
facet normal -.332732 .458927 -.823817
outer loop
vertex -.10309 .219172 -.436041
vertex -.113302 .232743 -.424356
vertex -.0997982 .222864 -.435314
endloop
endfacet
facet normal .675177 -.493946 -.547863
outer loop
vertex -.095691 .144131 -.432278
vertex -.0923281 .13126 -.416529
vertex -.0989828 .140439 -.433006
endloop
endfacet
facet normal .65835 .738515 .145504
outer loop
vertex -.0528271 .175571 -.407799
vertex -.0997982 .222864 -.435314
vertex -.0714827 .189219 -.392661
endloop
endfacet
facet normal -.658356 -.738509 -.145506
outer loop
vertex -.116593 .229051 -.425084
vertex -.114544 .226224 -.420009
vertex -.126091 .230022 -.387039
endloop
endfacet
facet normal -.332737 .458924 -.823817
outer loop
vertex -.149511 .192125 -.432359
vertex -.139299 .178554 -.444044
vertex -.152803 .188433 -.433086
endloop
endfacet
facet normal .675178 -.493944 -.547863
outer loop
vertex -.0627735 .181057 -.425003
vertex -.0594818 .18475 -.424275
vertex -.0528271 .175571 -.407799
endloop
endfacet
facet normal -.675176 .493946 .547863
outer loop
vertex -.145245 .186241 -.421796
vertex -.103284 .210694 -.392131
vertex -.114544 .226224 -.420009
endloop
endfacet
facet normal -.65835 -.738515 -.145504
outer loop
vertex -.10309 .219172 -.436041
vertex -.0627735 .181057 -.425003
vertex -.0897806 .200815 -.403088
endloop
endfacet
facet normal -.65835 -.738514 -.145505
outer loop
vertex -.126091 .230022 -.387039
vertex -.103284 .210694 -.392131
vertex -.130291 .230452 -.370217
endloop
endfacet
facet normal -.712699 .673784 -.195129
outer loop
vertex -.152803 .188433 -.433086
vertex -.1665 .189833 -.378219
vertex -.158753 .19307 -.39534
endloop
endfacet
facet normal .675178 -.493946 -.547861
outer loop
vertex -.0740818 .151728 -.412497
vertex -.0923281 .13126 -.416529
vertex -.0627735 .181057 -.425003
endloop
endfacet
facet normal .332734 -.458927 .823817
outer loop
vertex -.159846 .180655 -.361743
vertex -.0923281 .13126 -.416529
vertex -.0740818 .151728 -.412497
endloop
endfacet
facet normal -.332734 .458926 -.823817
outer loop
vertex -.163209 .193526 -.377492
vertex -.103284 .210694 -.392131
vertex -.132901 .17747 -.398677
endloop
endfacet
facet normal -.65835 -.738515 -.145504
outer loop
vertex -.0989828 .140439 -.433006
vertex -.0923281 .13126 -.416529
vertex -.159846 .180655 -.361743
endloop
endfacet
facet normal -.675177 .493945 .547863
outer loop
vertex -.130291 .230452 -.370217
vertex -.120345 .224966 -.353013
vertex -.126999 .234144 -.369489
endloop
endfacet
facet normal -.332733 .458927 -.823817
outer loop
vertex -.113302 .232743 -.424356
vertex -.10309 .219172 -.436041
vertex -.116593 .229051 -.425084
endloop
endfacet
facet normal -.332734 .458926 -.823817
outer loop
vertex -.12162 .198386 -.440136
vertex -.149511 .192125 -.432359
vertex -.116593 .229051 -.425084
endloop
endfacet
facet normal -.65835 -.738515 -.145504
outer loop
vertex -.103284 .210694 -.392131
vertex -.126091 .230022 -.387039
vertex -.114544 .226224 -.420009
endloop
endfacet
facet normal .65835 .738515 .145504
outer loop
vertex -.136202 .173768 -.399406
vertex -.145245 .186241 -.421796
vertex -.158753 .19307 -.39534
endloop
endfacet
facet normal .65835 .738514 .145504
outer loop
vertex -.0714827 .189219 -.392661
vertex -.126999 .234144 -.369489
vertex -.120345 .224966 -.353013
endloop
endfacet
facet normal .675176 -.493946 -.547863
outer loop
vertex -.0897806 .200815 -.403088
vertex -.110067 .178058 -.407572
vertex -.12162 .198386 -.440136
endloop
endfacet
facet normal -.658351 -.738514 -.145504
outer loop
vertex -.126901 .166832 -.440649
vertex -.159846 .180655 -.361743
vertex -.139299 .178554 -.444044
endloop
endfacet
facet normal -.675177 .493946 .547863
outer loop
vertex -.136202 .173768 -.399406
vertex -.132901 .17747 -.398677
vertex -.145245 .186241 -.421796
endloop
endfacet
facet normal -.332733 .458927 -.823817
outer loop
vertex -.149511 .192125 -.432359
vertex -.136007 .182246 -.443316
vertex -.139299 .178554 -.444044
endloop
endfacet
facet normal .242143 -.0247633 -.969925
outer loop
vertex -.136007 .182246 -.443316
vertex -.126901 .166832 -.440649
vertex -.139299 .178554 -.444044
endloop
endfacet
facet normal -.675176 .493946 .547863
outer loop
vertex -.103284 .210694 -.392131
vertex -.145245 .186241 -.421796
vertex -.132901 .17747 -.398677
endloop
endfacet
facet normal -.712701 .673782 -.195129
outer loop
vertex -.126091 .230022 -.387039
vertex -.126999 .234144 -.369489
vertex -.113302 .232743 -.424356
endloop
endfacet
facet normal .65835 .738515 .145504
outer loop
vertex -.0997982 .222864 -.435314
vertex -.0528271 .175571 -.407799
vertex -.0594818 .18475 -.424275
endloop
endfacet
facet normal .65835 .738515 .145503
outer loop
vertex -.145245 .186241 -.421796
vertex -.149511 .192125 -.432359
vertex -.158753 .19307 -.39534
endloop
endfacet
facet normal -.7127 .673783 -.195129
outer loop
vertex -.158753 .19307 -.39534
vertex -.1665 .189833 -.378219
vertex -.163209 .193526 -.377492
endloop
endfacet
facet normal -.332731 .458931 -.823816
outer loop
vertex -.163209 .193526 -.377492
vertex -.132901 .17747 -.398677
vertex -.136202 .173768 -.399406
endloop
endfacet
facet normal .242143 -.0247629 -.969924
outer loop
vertex -.0989828 .140439 -.433006
vertex -.126901 .166832 -.440649
vertex -.095691 .144131 -.432278
endloop
endfacet
facet normal .675176 -.493946 -.547863
outer loop
vertex -.12162 .198386 -.440136
vertex -.10309 .219172 -.436041
vertex -.0897806 .200815 -.403088
endloop
endfacet
facet normal -.658349 -.738515 -.145504
outer loop
vertex -.139299 .178554 -.444044
vertex -.159846 .180655 -.361743
vertex -.152803 .188433 -.433086
endloop
endfacet
facet normal -.675176 .493946 .547863
outer loop
vertex -.163209 .193526 -.377492
vertex -.159846 .180655 -.361743
vertex -.130291 .230452 -.370217
endloop
endfacet
facet normal .65835 .738515 .145504
outer loop
vertex -.126999 .234144 -.369489
vertex -.0997982 .222864 -.435314
vertex -.113302 .232743 -.424356
endloop
endfacet
facet normal -.332733 .458926 -.823817
outer loop
vertex -.12162 .198386 -.440136
vertex -.116593 .229051 -.425084
vertex -.10309 .219172 -.436041
endloop
endfacet
facet normal -.332734 .458926 -.823817
outer loop
vertex -.12162 .198386 -.440136
vertex -.136007 .182246 -.443316
vertex -.149511 .192125 -.432359
endloop
endfacet
facet normal -.712702 .673781 -.195129
outer loop
vertex -.158753 .19307 -.39534
vertex -.149511 .192125 -.432359
vertex -.152803 .188433 -.433086
endloop
endfacet
facet normal .332734 -.458926 .823817
outer loop
vertex -.120345 .224966 -.353013
vertex -.159846 .180655 -.361743
vertex -.0714827 .189219 -.392661
endloop
endfacet
facet normal .242143 -.0247633 -.969925
outer loop
vertex -.126901 .166832 -.440649
vertex -.136007 .182246 -.443316
vertex -.095691 .144131 -.432278
endloop
endfacet
facet normal -.332733 .458927 -.823817
outer loop
vertex -.095691 .144131 -.432278
vertex -.122698 .163889 -.410364
vertex -.110067 .178058 -.407572
endloop
endfacet
facet normal -.332734 .458926 -.823817
outer loop
vertex -.0897806 .200815 -.403088
vertex -.0627735 .181057 -.425003
vertex -.110067 .178058 -.407572
endloop
endfacet
facet normal .675177 -.493946 -.547863
outer loop
vertex -.0627735 .181057 -.425003
vertex -.0923281 .13126 -.416529
vertex -.095691 .144131 -.432278
endloop
endfacet
facet normal -.332734 .458926 -.823817
outer loop
vertex -.0627735 .181057 -.425003
vertex -.095691 .144131 -.432278
vertex -.110067 .178058 -.407572
endloop
endfacet
facet normal .675177 -.493946 -.547863
outer loop
vertex -.136007 .182246 -.443316
vertex -.12162 .198386 -.440136
vertex -.110067 .178058 -.407572
endloop
endfacet
facet normal .675176 -.493947 -.547863
outer loop
vertex -.0627735 .181057 -.425003
vertex -.0528271 .175571 -.407799
vertex -.0740818 .151728 -.412497
endloop
endfacet
facet normal -.65835 -.738515 -.145504
outer loop
vertex -.126901 .166832 -.440649
vertex -.0989828 .140439 -.433006
vertex -.159846 .180655 -.361743
endloop
endfacet
facet normal .65835 .738515 .145504
outer loop
vertex -.126999 .234144 -.369489
vertex -.0714827 .189219 -.392661
vertex -.0997982 .222864 -.435314
endloop
endfacet
facet normal .65835 .738514 .145505
outer loop
vertex -.136202 .173768 -.399406
vertex -.158753 .19307 -.39534
vertex -.163209 .193526 -.377492
endloop
endfacet
facet normal -.332734 .458926 -.823817
outer loop
vertex -.103284 .210694 -.392131
vertex -.163209 .193526 -.377492
vertex -.130291 .230452 -.370217
endloop
endfacet
facet normal -.675177 .493947 .547861
outer loop
vertex -.114544 .226224 -.420009
vertex -.116593 .229051 -.425084
vertex -.149511 .192125 -.432359
endloop
endfacet
facet normal -.675177 .493946 .547863
outer loop
vertex -.114544 .226224 -.420009
vertex -.149511 .192125 -.432359
vertex -.145245 .186241 -.421796
endloop
endfacet
facet normal -.65835 -.738515 -.145504
outer loop
vertex -.152803 .188433 -.433086
vertex -.159846 .180655 -.361743
vertex -.1665 .189833 -.378219
endloop
endfacet
facet normal -.675176 .493946 .547863
outer loop
vertex -.120345 .224966 -.353013
vertex -.130291 .230452 -.370217
vertex -.159846 .180655 -.361743
endloop
endfacet
facet normal .332734 -.458926 .823817
outer loop
vertex -.0714827 .189219 -.392661
vertex -.159846 .180655 -.361743
vertex -.0528271 .175571 -.407799
endloop
endfacet
facet normal .332734 -.458926 .823817
outer loop
vertex -.0528271 .175571 -.407799
vertex -.159846 .180655 -.361743
vertex -.0740818 .151728 -.412497
endloop
endfacet
facet normal .65835 .738515 .145504
outer loop
vertex -.095691 .144131 -.432278
vertex -.136007 .182246 -.443316
vertex -.122698 .163889 -.410364
endloop
endfacet
facet normal .242143 -.0247636 -.969925
outer loop
vertex -.0594818 .18475 -.424275
vertex -.0627735 .181057 -.425003
vertex -.10309 .219172 -.436041
endloop
endfacet
facet normal .242142 -.024764 -.969925
outer loop
vertex -.0997982 .222864 -.435314
vertex -.0594818 .18475 -.424275
vertex -.10309 .219172 -.436041
endloop
endfacet
facet normal -.712702 .673781 -.195129
outer loop
vertex -.113302 .232743 -.424356
vertex -.116593 .229051 -.425084
vertex -.126091 .230022 -.387039
endloop
endfacet
facet normal -.7127 .673783 -.195129
outer loop
vertex -.126999 .234144 -.369489
vertex -.126091 .230022 -.387039
vertex -.130291 .230452 -.370217
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0715003 .310009 -.397155
vertex .0697682 .309234 -.395606
vertex .0697682 .311558 -.39638
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0715003 .310009 -.397155
vertex .0697682 .311558 -.39638
vertex .0715003 .310783 -.394831
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0715003 .310009 -.397155
vertex .0715003 .310783 -.394831
vertex .0697682 .309234 -.395606
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0697682 .311558 -.39638
vertex .0697682 .309234 -.395606
vertex .0715003 .310783 -.394831
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0569296 .309738 -.40288
vertex .0551975 .308964 -.401331
vertex .0551975 .311287 -.402105
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0569296 .309738 -.40288
vertex .0551975 .311287 -.402105
vertex .0569296 .310513 -.400556
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0569296 .309738 -.40288
vertex .0569296 .310513 -.400556
vertex .0551975 .308964 -.401331
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0551975 .311287 -.402105
vertex .0551975 .308964 -.401331
vertex .0569296 .310513 -.400556
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0345313 .311247 -.41107
vertex .0327992 .310472 -.409521
vertex .0327992 .312796 -.410295
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0345313 .311247 -.41107
vertex .0327992 .312796 -.410295
vertex .0345313 .312022 -.408746
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0345313 .311247 -.41107
vertex .0345313 .312022 -.408746
vertex .0327992 .310472 -.409521
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0327992 .312796 -.410295
vertex .0327992 .310472 -.409521
vertex .0345313 .312022 -.408746
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0638056 .302625 -.399729
vertex .0620735 .30185 -.398179
vertex .0620735 .304174 -.398954
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0638056 .302625 -.399729
vertex .0620735 .304174 -.398954
vertex .0638056 .303399 -.397405
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0638056 .302625 -.399729
vertex .0638056 .303399 -.397405
vertex .0620735 .30185 -.398179
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0620735 .304174 -.398954
vertex .0620735 .30185 -.398179
vertex .0638056 .303399 -.397405
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0306481 .307774 -.412807
vertex .028916 .306999 -.411257
vertex .028916 .309323 -.412032
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0306481 .307774 -.412807
vertex .028916 .309323 -.412032
vertex .0306481 .308548 -.410483
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0306481 .307774 -.412807
vertex .0306481 .308548 -.410483
vertex .028916 .306999 -.411257
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .028916 .309323 -.412032
vertex .028916 .306999 -.411257
vertex .0306481 .308548 -.410483
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0599096 .298639 -.400604
vertex .0581775 .297864 -.399054
vertex .0581775 .300188 -.399829
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0599096 .298639 -.400604
vertex .0581775 .300188 -.399829
vertex .0599096 .299413 -.39828
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0599096 .298639 -.400604
vertex .0599096 .299413 -.39828
vertex .0581775 .297864 -.399054
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0581775 .300188 -.399829
vertex .0581775 .297864 -.399054
vertex .0599096 .299413 -.39828
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0100731 .257333 -.35082
vertex -.0118051 .256558 -.349271
vertex -.0118051 .258882 -.350046
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0100731 .257333 -.35082
vertex -.0118051 .258882 -.350046
vertex -.0100731 .258107 -.348497
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0100731 .257333 -.35082
vertex -.0100731 .258107 -.348497
vertex -.0118051 .256558 -.349271
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0118051 .258882 -.350046
vertex -.0118051 .256558 -.349271
vertex -.0100731 .258107 -.348497
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0597893 .294666 -.401472
vertex .0580573 .293892 -.399923
vertex .0580573 .296215 -.400697
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0597893 .294666 -.401472
vertex .0580573 .296215 -.400697
vertex .0597893 .295441 -.399148
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0597893 .294666 -.401472
vertex .0597893 .295441 -.399148
vertex .0580573 .293892 -.399923
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0580573 .296215 -.400697
vertex .0580573 .293892 -.399923
vertex .0597893 .295441 -.399148
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0195381 .249633 -.345726
vertex -.0212702 .248858 -.344177
vertex -.0212702 .251182 -.344952
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0195381 .249633 -.345726
vertex -.0212702 .251182 -.344952
vertex -.0195381 .250407 -.343402
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0195381 .249633 -.345726
vertex -.0195381 .250407 -.343402
vertex -.0212702 .248858 -.344177
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0212702 .251182 -.344952
vertex -.0212702 .248858 -.344177
vertex -.0195381 .250407 -.343402
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0450165 .293089 -.405615
vertex .0432845 .292314 -.404066
vertex .0432845 .294638 -.40484
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0450165 .293089 -.405615
vertex .0432845 .294638 -.40484
vertex .0450165 .293863 -.403291
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0450165 .293089 -.405615
vertex .0450165 .293863 -.403291
vertex .0432845 .292314 -.404066
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0432845 .294638 -.40484
vertex .0432845 .292314 -.404066
vertex .0450165 .293863 -.403291
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0733213 .28535 -.39495
vertex .0715893 .284575 -.3934
vertex .0715893 .286899 -.394175
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0733213 .28535 -.39495
vertex .0715893 .286899 -.394175
vertex .0733213 .286125 -.392626
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0733213 .28535 -.39495
vertex .0733213 .286125 -.392626
vertex .0715893 .284575 -.3934
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0715893 .286899 -.394175
vertex .0715893 .284575 -.3934
vertex .0733213 .286125 -.392626
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0259206 .246161 -.346344
vertex -.0276526 .245387 -.344795
vertex -.0276526 .247711 -.345569
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0259206 .246161 -.346344
vertex -.0276526 .247711 -.345569
vertex -.0259206 .246936 -.34402
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0259206 .246161 -.346344
vertex -.0259206 .246936 -.34402
vertex -.0276526 .245387 -.344795
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0276526 .247711 -.345569
vertex -.0276526 .245387 -.344795
vertex -.0259206 .246936 -.34402
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .044927 .289116 -.406483
vertex .0431949 .288342 -.404934
vertex .0431949 .290665 -.405708
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .044927 .289116 -.406483
vertex .0431949 .290665 -.405708
vertex .044927 .289891 -.404159
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .044927 .289116 -.406483
vertex .044927 .289891 -.404159
vertex .0431949 .288342 -.404934
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0431949 .290665 -.405708
vertex .0431949 .288342 -.404934
vertex .044927 .289891 -.404159
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0731704 .281467 -.395773
vertex .0714384 .280692 -.394224
vertex .0714384 .283016 -.394999
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0731704 .281467 -.395773
vertex .0714384 .283016 -.394999
vertex .0731704 .282241 -.393449
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0731704 .281467 -.395773
vertex .0731704 .282241 -.393449
vertex .0714384 .280692 -.394224
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0714384 .283016 -.394999
vertex .0714384 .280692 -.394224
vertex .0731704 .282241 -.393449
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0299237 .249683 -.356882
vertex -.0316558 .248908 -.355332
vertex -.0316558 .251232 -.356107
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0299237 .249683 -.356882
vertex -.0316558 .251232 -.356107
vertex -.0299237 .250457 -.354558
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0299237 .249683 -.356882
vertex -.0299237 .250457 -.354558
vertex -.0316558 .248908 -.355332
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0316558 .251232 -.356107
vertex -.0316558 .248908 -.355332
vertex -.0299237 .250457 -.354558
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.003555 .251427 -.359363
vertex -.00528705 .250653 -.357814
vertex -.00528705 .252976 -.358589
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.003555 .251427 -.359363
vertex -.00528705 .252976 -.358589
vertex -.003555 .252202 -.35704
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.003555 .251427 -.359363
vertex -.003555 .252202 -.35704
vertex -.00528705 .250653 -.357814
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00528705 .252976 -.358589
vertex -.00528705 .250653 -.357814
vertex -.003555 .252202 -.35704
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0693512 .277597 -.39659
vertex .0676191 .276823 -.395041
vertex .0676191 .279147 -.395815
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0693512 .277597 -.39659
vertex .0676191 .279147 -.395815
vertex .0693512 .278372 -.394266
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0693512 .277597 -.39659
vertex .0693512 .278372 -.394266
vertex .0676191 .276823 -.395041
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0676191 .279147 -.395815
vertex .0676191 .276823 -.395041
vertex .0693512 .278372 -.394266
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0407705 .233513 -.339252
vertex -.0425025 .232738 -.337702
vertex -.0425025 .235062 -.338477
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0407705 .233513 -.339252
vertex -.0425025 .235062 -.338477
vertex -.0407705 .234288 -.336928
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0407705 .233513 -.339252
vertex -.0407705 .234288 -.336928
vertex -.0425025 .232738 -.337702
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0425025 .235062 -.338477
vertex -.0425025 .232738 -.337702
vertex -.0407705 .234288 -.336928
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0166422 .245582 -.356696
vertex -.0183743 .244807 -.355147
vertex -.0183743 .247131 -.355921
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0166422 .245582 -.356696
vertex -.0183743 .247131 -.355921
vertex -.0166422 .246356 -.354372
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0166422 .245582 -.356696
vertex -.0166422 .246356 -.354372
vertex -.0183743 .244807 -.355147
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0183743 .247131 -.355921
vertex -.0183743 .244807 -.355147
vertex -.0166422 .246356 -.354372
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0553535 .277765 -.403214
vertex .0536215 .27699 -.401665
vertex .0536215 .279314 -.40244
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0553535 .277765 -.403214
vertex .0536215 .279314 -.40244
vertex .0553535 .278539 -.400891
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0553535 .277765 -.403214
vertex .0553535 .278539 -.400891
vertex .0536215 .27699 -.401665
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0536215 .279314 -.40244
vertex .0536215 .27699 -.401665
vertex .0553535 .278539 -.400891
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0649393 .22788 -.336478
vertex -.0666713 .227106 -.334929
vertex -.0666713 .229429 -.335703
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0649393 .22788 -.336478
vertex -.0666713 .229429 -.335703
vertex -.0649393 .228655 -.334154
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0649393 .22788 -.336478
vertex -.0649393 .228655 -.334154
vertex -.0666713 .227106 -.334929
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0666713 .229429 -.335703
vertex -.0666713 .227106 -.334929
vertex -.0649393 .228655 -.334154
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0334615 .245483 -.362335
vertex -.0351935 .244708 -.360786
vertex -.0351935 .247032 -.361561
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0334615 .245483 -.362335
vertex -.0351935 .247032 -.361561
vertex -.0334615 .246258 -.360012
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0334615 .245483 -.362335
vertex -.0334615 .246258 -.360012
vertex -.0351935 .244708 -.360786
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0351935 .247032 -.361561
vertex -.0351935 .244708 -.360786
vertex -.0334615 .246258 -.360012
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0059782 .226177 -.333975
vertex .00424615 .225402 -.332426
vertex .00424615 .227726 -.333201
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0059782 .226177 -.333975
vertex .00424615 .227726 -.333201
vertex .0059782 .226951 -.331652
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0059782 .226177 -.333975
vertex .0059782 .226951 -.331652
vertex .00424615 .225402 -.332426
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00424615 .227726 -.333201
vertex .00424615 .225402 -.332426
vertex .0059782 .226951 -.331652
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0721138 .267628 -.394866
vertex .0703817 .266854 -.393317
vertex .0703817 .269178 -.394092
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0721138 .267628 -.394866
vertex .0703817 .269178 -.394092
vertex .0721138 .268403 -.392542
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0721138 .267628 -.394866
vertex .0721138 .268403 -.392542
vertex .0703817 .266854 -.393317
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0703817 .269178 -.394092
vertex .0703817 .266854 -.393317
vertex .0721138 .268403 -.392542
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0586209 .22456 -.33702
vertex -.0603529 .223785 -.335471
vertex -.0603529 .226109 -.336245
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0586209 .22456 -.33702
vertex -.0603529 .226109 -.336245
vertex -.0586209 .225335 -.334696
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0586209 .22456 -.33702
vertex -.0586209 .225335 -.334696
vertex -.0603529 .223785 -.335471
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0603529 .226109 -.336245
vertex -.0603529 .223785 -.335471
vertex -.0586209 .225335 -.334696
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0202298 .209975 -.315244
vertex -.0219619 .2092 -.313694
vertex -.0219619 .211524 -.314469
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0202298 .209975 -.315244
vertex -.0219619 .211524 -.314469
vertex -.0202298 .210749 -.31292
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0202298 .209975 -.315244
vertex -.0202298 .210749 -.31292
vertex -.0219619 .2092 -.313694
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0219619 .211524 -.314469
vertex -.0219619 .2092 -.313694
vertex -.0202298 .210749 -.31292
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00290591 .222316 -.33367
vertex .00117386 .221541 -.332121
vertex .00117386 .223865 -.332895
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00290591 .222316 -.33367
vertex .00117386 .223865 -.332895
vertex .00290591 .223091 -.331346
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00290591 .222316 -.33367
vertex .00290591 .223091 -.331346
vertex .00117386 .221541 -.332121
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00117386 .223865 -.332895
vertex .00117386 .221541 -.332121
vertex .00290591 .223091 -.331346
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0650049 .264949 -.397324
vertex .0632728 .264175 -.395774
vertex .0632728 .266499 -.396549
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0650049 .264949 -.397324
vertex .0632728 .266499 -.396549
vertex .0650049 .265724 -.395
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0650049 .264949 -.397324
vertex .0650049 .265724 -.395
vertex .0632728 .264175 -.395774
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0632728 .266499 -.396549
vertex .0632728 .264175 -.395774
vertex .0650049 .265724 -.395
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0647756 .221807 -.338396
vertex -.0665076 .221033 -.336847
vertex -.0665076 .223357 -.337622
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0647756 .221807 -.338396
vertex -.0665076 .223357 -.337622
vertex -.0647756 .222582 -.336072
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0647756 .221807 -.338396
vertex -.0647756 .222582 -.336072
vertex -.0665076 .221033 -.336847
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0665076 .223357 -.337622
vertex -.0665076 .221033 -.336847
vertex -.0647756 .222582 -.336072
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0336727 .20186 -.308121
vertex -.0354048 .201085 -.306572
vertex -.0354048 .203409 -.307346
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0336727 .20186 -.308121
vertex -.0354048 .203409 -.307346
vertex -.0336727 .202634 -.305797
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0336727 .20186 -.308121
vertex -.0336727 .202634 -.305797
vertex -.0354048 .201085 -.306572
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0354048 .203409 -.307346
vertex -.0354048 .201085 -.306572
vertex -.0336727 .202634 -.305797
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0119286 .212942 -.32494
vertex -.0136607 .212167 -.323391
vertex -.0136607 .214491 -.324166
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0119286 .212942 -.32494
vertex -.0136607 .214491 -.324166
vertex -.0119286 .213716 -.322617
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0119286 .212942 -.32494
vertex -.0119286 .213716 -.322617
vertex -.0136607 .212167 -.323391
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0136607 .214491 -.324166
vertex -.0136607 .212167 -.323391
vertex -.0119286 .213716 -.322617
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0120741 .220145 -.335873
vertex .010342 .219371 -.334324
vertex .010342 .221694 -.335099
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0120741 .220145 -.335873
vertex .010342 .221694 -.335099
vertex .0120741 .22092 -.333549
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0120741 .220145 -.335873
vertex .0120741 .22092 -.333549
vertex .010342 .219371 -.334324
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .010342 .221694 -.335099
vertex .010342 .219371 -.334324
vertex .0120741 .22092 -.333549
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0716609 .259486 -.395583
vertex .0699289 .258712 -.394034
vertex .0699289 .261036 -.394808
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0716609 .259486 -.395583
vertex .0699289 .261036 -.394808
vertex .0716609 .260261 -.393259
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0716609 .259486 -.395583
vertex .0716609 .260261 -.393259
vertex .0699289 .258712 -.394034
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0699289 .261036 -.394808
vertex .0699289 .258712 -.394034
vertex .0716609 .260261 -.393259
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0618542 .219602 -.340617
vertex -.0635863 .218828 -.339068
vertex -.0635863 .221151 -.339842
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0618542 .219602 -.340617
vertex -.0635863 .221151 -.339842
vertex -.0618542 .220377 -.338293
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0618542 .219602 -.340617
vertex -.0618542 .220377 -.338293
vertex -.0635863 .218828 -.339068
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0635863 .221151 -.339842
vertex -.0635863 .218828 -.339068
vertex -.0618542 .220377 -.338293
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0306977 .19826 -.307685
vertex -.0324298 .197485 -.306135
vertex -.0324298 .199809 -.30691
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0306977 .19826 -.307685
vertex -.0324298 .199809 -.30691
vertex -.0306977 .199035 -.305361
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0306977 .19826 -.307685
vertex -.0306977 .199035 -.305361
vertex -.0324298 .197485 -.306135
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0324298 .199809 -.30691
vertex -.0324298 .197485 -.306135
vertex -.0306977 .199035 -.305361
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00614468 .21413 -.332173
vertex -.00787673 .213355 -.330623
vertex -.00787673 .215679 -.331398
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00614468 .21413 -.332173
vertex -.00787673 .215679 -.331398
vertex -.00614468 .214904 -.329849
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00614468 .21413 -.332173
vertex -.00614468 .214904 -.329849
vertex -.00787673 .213355 -.330623
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00787673 .215679 -.331398
vertex -.00787673 .213355 -.330623
vertex -.00614468 .214904 -.329849
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0179652 .214677 -.333017
vertex .0162332 .213902 -.331468
vertex .0162332 .216226 -.332242
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0179652 .214677 -.333017
vertex .0162332 .216226 -.332242
vertex .0179652 .215452 -.330693
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0179652 .214677 -.333017
vertex .0179652 .215452 -.330693
vertex .0162332 .213902 -.331468
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0162332 .216226 -.332242
vertex .0162332 .213902 -.331468
vertex .0179652 .215452 -.330693
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0746256 .254078 -.393815
vertex .0728936 .253304 -.392266
vertex .0728936 .255627 -.39304
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0746256 .254078 -.393815
vertex .0728936 .255627 -.39304
vertex .0746256 .254853 -.391491
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0746256 .254078 -.393815
vertex .0746256 .254853 -.391491
vertex .0728936 .253304 -.392266
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0728936 .255627 -.39304
vertex .0728936 .253304 -.392266
vertex .0746256 .254853 -.391491
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.065103 .217383 -.342844
vertex -.066835 .216609 -.341295
vertex -.066835 .218932 -.34207
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.065103 .217383 -.342844
vertex -.066835 .218932 -.34207
vertex -.065103 .218158 -.340521
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.065103 .217383 -.342844
vertex -.065103 .218158 -.340521
vertex -.066835 .216609 -.341295
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.066835 .218932 -.34207
vertex -.066835 .216609 -.341295
vertex -.065103 .218158 -.340521
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0363715 .196309 -.309778
vertex -.0381036 .195534 -.308229
vertex -.0381036 .197858 -.309004
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0363715 .196309 -.309778
vertex -.0381036 .197858 -.309004
vertex -.0363715 .197083 -.307455
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0363715 .196309 -.309778
vertex -.0363715 .197083 -.307455
vertex -.0381036 .195534 -.308229
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0381036 .197858 -.309004
vertex -.0381036 .195534 -.308229
vertex -.0363715 .197083 -.307455
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0120283 .209278 -.330127
vertex -.0137604 .208503 -.328577
vertex -.0137604 .210827 -.329352
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0120283 .209278 -.330127
vertex -.0137604 .210827 -.329352
vertex -.0120283 .210052 -.327803
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0120283 .209278 -.330127
vertex -.0120283 .210052 -.327803
vertex -.0137604 .208503 -.328577
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0137604 .210827 -.329352
vertex -.0137604 .208503 -.328577
vertex -.0120283 .210052 -.327803
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0119614 .21252 -.335214
vertex .0102294 .211745 -.333665
vertex .0102294 .214069 -.334439
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0119614 .21252 -.335214
vertex .0102294 .214069 -.334439
vertex .0119614 .213295 -.33289
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0119614 .21252 -.335214
vertex .0119614 .213295 -.33289
vertex .0102294 .211745 -.333665
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0102294 .214069 -.334439
vertex .0102294 .211745 -.333665
vertex .0119614 .213295 -.33289
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0677751 .251967 -.397107
vertex .0660431 .251193 -.395557
vertex .0660431 .253516 -.396332
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0677751 .251967 -.397107
vertex .0660431 .253516 -.396332
vertex .0677751 .252742 -.394783
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0677751 .251967 -.397107
vertex .0677751 .252742 -.394783
vertex .0660431 .251193 -.395557
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0660431 .253516 -.396332
vertex .0660431 .251193 -.395557
vertex .0677751 .252742 -.394783
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0718255 .216218 -.346781
vertex -.0735576 .215443 -.345232
vertex -.0735576 .217767 -.346007
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0718255 .216218 -.346781
vertex -.0735576 .217767 -.346007
vertex -.0718255 .216992 -.344457
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0718255 .216218 -.346781
vertex -.0718255 .216992 -.344457
vertex -.0735576 .215443 -.345232
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0735576 .217767 -.346007
vertex -.0735576 .215443 -.345232
vertex -.0718255 .216992 -.344457
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0418407 .193277 -.310176
vertex -.0435728 .192502 -.308627
vertex -.0435728 .194826 -.309402
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0418407 .193277 -.310176
vertex -.0435728 .194826 -.309402
vertex -.0418407 .194051 -.307852
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0418407 .193277 -.310176
vertex -.0418407 .194051 -.307852
vertex -.0435728 .192502 -.308627
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0435728 .194826 -.309402
vertex -.0435728 .192502 -.308627
vertex -.0418407 .194051 -.307852
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0190485 .188475 -.302515
vertex -.0207805 .1877 -.300966
vertex -.0207805 .190024 -.30174
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0190485 .188475 -.302515
vertex -.0207805 .190024 -.30174
vertex -.0190485 .18925 -.300191
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0190485 .188475 -.302515
vertex -.0190485 .18925 -.300191
vertex -.0207805 .1877 -.300966
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0207805 .190024 -.30174
vertex -.0207805 .1877 -.300966
vertex -.0190485 .18925 -.300191
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00591414 .209815 -.336566
vertex .00418209 .209041 -.335017
vertex .00418209 .211365 -.335791
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00591414 .209815 -.336566
vertex .00418209 .211365 -.335791
vertex .00591414 .21059 -.334242
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00591414 .209815 -.336566
vertex .00591414 .21059 -.334242
vertex .00418209 .209041 -.335017
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00418209 .211365 -.335791
vertex .00418209 .209041 -.335017
vertex .00591414 .21059 -.334242
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0293612 .20448 -.328053
vertex .0276292 .203706 -.326504
vertex .0276292 .20603 -.327279
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0293612 .20448 -.328053
vertex .0276292 .20603 -.327279
vertex .0293612 .205255 -.325729
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0293612 .20448 -.328053
vertex .0293612 .205255 -.325729
vertex .0276292 .203706 -.326504
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0276292 .20603 -.327279
vertex .0276292 .203706 -.326504
vertex .0293612 .205255 -.325729
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.124997 .219224 -.357577
vertex -.12673 .218449 -.356027
vertex -.12673 .220773 -.356802
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.124997 .219224 -.357577
vertex -.12673 .220773 -.356802
vertex -.124997 .219998 -.355253
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.124997 .219224 -.357577
vertex -.124997 .219998 -.355253
vertex -.12673 .218449 -.356027
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.12673 .220773 -.356802
vertex -.12673 .218449 -.356027
vertex -.124997 .219998 -.355253
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0720046 .213957 -.349029
vertex -.0737366 .213183 -.34748
vertex -.0737366 .215507 -.348255
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0720046 .213957 -.349029
vertex -.0737366 .215507 -.348255
vertex -.0720046 .214732 -.346706
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0720046 .213957 -.349029
vertex -.0720046 .214732 -.346706
vertex -.0737366 .213183 -.34748
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0737366 .215507 -.348255
vertex -.0737366 .213183 -.34748
vertex -.0720046 .214732 -.346706
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0417231 .190258 -.310567
vertex -.0434551 .189484 -.309018
vertex -.0434551 .191808 -.309793
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0417231 .190258 -.310567
vertex -.0434551 .191808 -.309793
vertex -.0417231 .191033 -.308244
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0417231 .190258 -.310567
vertex -.0417231 .191033 -.308244
vertex -.0434551 .189484 -.309018
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0434551 .191808 -.309793
vertex -.0434551 .189484 -.309018
vertex -.0417231 .191033 -.308244
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0189922 .185519 -.302875
vertex -.0207242 .184744 -.301326
vertex -.0207242 .187068 -.3021
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0189922 .185519 -.302875
vertex -.0207242 .187068 -.3021
vertex -.0189922 .186293 -.300551
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0189922 .185519 -.302875
vertex -.0189922 .186293 -.300551
vertex -.0207242 .184744 -.301326
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0207242 .187068 -.3021
vertex -.0207242 .184744 -.301326
vertex -.0189922 .186293 -.300551
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00590133 .206584 -.337063
vertex .00416928 .20581 -.335514
vertex .00416928 .208134 -.336289
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00590133 .206584 -.337063
vertex .00416928 .208134 -.336289
vertex .00590133 .207359 -.33474
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00590133 .206584 -.337063
vertex .00590133 .207359 -.33474
vertex .00416928 .20581 -.335514
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00416928 .208134 -.336289
vertex .00416928 .20581 -.335514
vertex .00590133 .207359 -.33474
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0292128 .200791 -.327662
vertex .0274808 .200017 -.326112
vertex .0274808 .20234 -.326887
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0292128 .200791 -.327662
vertex .0274808 .20234 -.326887
vertex .0292128 .201566 -.325338
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0292128 .200791 -.327662
vertex .0292128 .201566 -.325338
vertex .0274808 .200017 -.326112
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0274808 .20234 -.326887
vertex .0274808 .200017 -.326112
vertex .0292128 .201566 -.325338
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.124696 .215842 -.358149
vertex -.126428 .215067 -.3566
vertex -.126428 .217391 -.357375
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.124696 .215842 -.358149
vertex -.126428 .217391 -.357375
vertex -.124696 .216616 -.355826
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.124696 .215842 -.358149
vertex -.124696 .216616 -.355826
vertex -.126428 .215067 -.3566
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.126428 .217391 -.357375
vertex -.126428 .215067 -.3566
vertex -.124696 .216616 -.355826
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.091276 .236633 -.392475
vertex -.0930081 .235858 -.390926
vertex -.0930081 .238182 -.391701
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.091276 .236633 -.392475
vertex -.0930081 .238182 -.391701
vertex -.091276 .237408 -.390151
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.091276 .236633 -.392475
vertex -.091276 .237408 -.390151
vertex -.0930081 .235858 -.390926
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0930081 .238182 -.391701
vertex -.0930081 .235858 -.390926
vertex -.091276 .237408 -.390151
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0500154 .210644 -.349568
vertex -.0517474 .209869 -.348019
vertex -.0517474 .212193 -.348793
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0500154 .210644 -.349568
vertex -.0517474 .212193 -.348793
vertex -.0500154 .211419 -.347244
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0500154 .210644 -.349568
vertex -.0500154 .211419 -.347244
vertex -.0517474 .209869 -.348019
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0517474 .212193 -.348793
vertex -.0517474 .209869 -.348019
vertex -.0500154 .211419 -.347244
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0216373 .182576 -.303228
vertex -.0233693 .181801 -.301679
vertex -.0233693 .184125 -.302454
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0216373 .182576 -.303228
vertex -.0233693 .184125 -.302454
vertex -.0216373 .18335 -.300905
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0216373 .182576 -.303228
vertex -.0216373 .18335 -.300905
vertex -.0233693 .181801 -.301679
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0233693 .184125 -.302454
vertex -.0233693 .181801 -.301679
vertex -.0216373 .18335 -.300905
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00288022 .203367 -.337554
vertex .00114817 .202592 -.336005
vertex .00114817 .204916 -.336779
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00288022 .203367 -.337554
vertex .00114817 .204916 -.336779
vertex .00288022 .204142 -.33523
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00288022 .203367 -.337554
vertex .00288022 .204142 -.33523
vertex .00114817 .202592 -.336005
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00114817 .204916 -.336779
vertex .00114817 .202592 -.336005
vertex .00288022 .204142 -.33523
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0262813 .198169 -.328973
vertex .0245492 .197395 -.327423
vertex .0245492 .199718 -.328198
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0262813 .198169 -.328973
vertex .0245492 .199718 -.328198
vertex .0262813 .198944 -.326649
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0262813 .198169 -.328973
vertex .0262813 .198944 -.326649
vertex .0245492 .197395 -.327423
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0245492 .199718 -.328198
vertex .0245492 .197395 -.327423
vertex .0262813 .198944 -.326649
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.133623 .211961 -.357854
vertex -.135355 .211186 -.356305
vertex -.135355 .21351 -.357079
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.133623 .211961 -.357854
vertex -.135355 .21351 -.357079
vertex -.133623 .212735 -.35553
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.133623 .211961 -.357854
vertex -.133623 .212735 -.35553
vertex -.135355 .211186 -.356305
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.135355 .21351 -.357079
vertex -.135355 .211186 -.356305
vertex -.133623 .212735 -.35553
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.109787 .215038 -.363024
vertex -.111519 .214263 -.361474
vertex -.111519 .216587 -.362249
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.109787 .215038 -.363024
vertex -.111519 .216587 -.362249
vertex -.109787 .215813 -.3607
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.109787 .215038 -.363024
vertex -.109787 .215813 -.3607
vertex -.111519 .214263 -.361474
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.111519 .216587 -.362249
vertex -.111519 .214263 -.361474
vertex -.109787 .215813 -.3607
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0626343 .20837 -.351823
vertex -.0643664 .207596 -.350274
vertex -.0643664 .209919 -.351048
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0626343 .20837 -.351823
vertex -.0643664 .209919 -.351048
vertex -.0626343 .209145 -.349499
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0626343 .20837 -.351823
vertex -.0626343 .209145 -.349499
vertex -.0643664 .207596 -.350274
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0643664 .209919 -.351048
vertex -.0643664 .207596 -.350274
vertex -.0626343 .209145 -.349499
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0271192 .180673 -.305298
vertex -.0288513 .179898 -.303749
vertex -.0288513 .182222 -.304523
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0271192 .180673 -.305298
vertex -.0288513 .182222 -.304523
vertex -.0271192 .181447 -.302974
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0271192 .180673 -.305298
vertex -.0271192 .181447 -.302974
vertex -.0288513 .179898 -.303749
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0288513 .182222 -.304523
vertex -.0288513 .179898 -.303749
vertex -.0271192 .181447 -.302974
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00525045 .17503 -.295821
vertex -.0069825 .174256 -.294271
vertex -.0069825 .17658 -.295046
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00525045 .17503 -.295821
vertex -.0069825 .17658 -.295046
vertex -.00525045 .175805 -.293497
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00525045 .17503 -.295821
vertex -.00525045 .175805 -.293497
vertex -.0069825 .174256 -.294271
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0069825 .17658 -.295046
vertex -.0069825 .174256 -.294271
vertex -.00525045 .175805 -.293497
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0205205 .196573 -.332007
vertex .0187885 .195798 -.330458
vertex .0187885 .198122 -.331232
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0205205 .196573 -.332007
vertex .0187885 .198122 -.331232
vertex .0205205 .197348 -.329683
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0205205 .196573 -.332007
vertex .0205205 .197348 -.329683
vertex .0187885 .195798 -.330458
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0187885 .198122 -.331232
vertex .0187885 .195798 -.330458
vertex .0205205 .197348 -.329683
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0484545 .190418 -.321668
vertex .0467224 .189643 -.320119
vertex .0467224 .191967 -.320893
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0484545 .190418 -.321668
vertex .0467224 .191967 -.320893
vertex .0484545 .191193 -.319344
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0484545 .190418 -.321668
vertex .0484545 .191193 -.319344
vertex .0467224 .189643 -.320119
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0467224 .191967 -.320893
vertex .0467224 .189643 -.320119
vertex .0484545 .191193 -.319344
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.118884 .211143 -.362735
vertex -.120616 .210368 -.361186
vertex -.120616 .212692 -.36196
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.118884 .211143 -.362735
vertex -.120616 .212692 -.36196
vertex -.118884 .211918 -.360411
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.118884 .211143 -.362735
vertex -.118884 .211918 -.360411
vertex -.120616 .210368 -.361186
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.120616 .212692 -.36196
vertex -.120616 .210368 -.361186
vertex -.118884 .211918 -.360411
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0916802 .231385 -.397335
vertex -.0934122 .230611 -.395786
vertex -.0934122 .232934 -.396561
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0916802 .231385 -.397335
vertex -.0934122 .232934 -.396561
vertex -.0916802 .23216 -.395011
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0916802 .231385 -.397335
vertex -.0916802 .23216 -.395011
vertex -.0934122 .230611 -.395786
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0934122 .232934 -.396561
vertex -.0934122 .230611 -.395786
vertex -.0916802 .23216 -.395011
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0533971 .206082 -.354085
vertex -.0551292 .205308 -.352536
vertex -.0551292 .207632 -.35331
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0533971 .206082 -.354085
vertex -.0551292 .207632 -.35331
vertex -.0533971 .206857 -.351761
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0533971 .206082 -.354085
vertex -.0533971 .206857 -.351761
vertex -.0551292 .205308 -.352536
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0551292 .207632 -.35331
vertex -.0551292 .205308 -.352536
vertex -.0533971 .206857 -.351761
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0159431 .174707 -.300454
vertex -.0176752 .173933 -.298905
vertex -.0176752 .176256 -.29968
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0159431 .174707 -.300454
vertex -.0176752 .176256 -.29968
vertex -.0159431 .175482 -.298131
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0159431 .174707 -.300454
vertex -.0159431 .175482 -.298131
vertex -.0176752 .173933 -.298905
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0176752 .176256 -.29968
vertex -.0176752 .173933 -.298905
vertex -.0159431 .175482 -.298131
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00879438 .195455 -.33592
vertex .00706233 .194681 -.334371
vertex .00706233 .197005 -.335145
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00879438 .195455 -.33592
vertex .00706233 .197005 -.335145
vertex .00879438 .19623 -.333596
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00879438 .195455 -.33592
vertex .00879438 .19623 -.333596
vertex .00706233 .194681 -.334371
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00706233 .197005 -.335145
vertex .00706233 .194681 -.334371
vertex .00879438 .19623 -.333596
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0391557 .184322 -.31689
vertex .0374236 .183548 -.31534
vertex .0374236 .185871 -.316115
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0391557 .184322 -.31689
vertex .0374236 .185871 -.316115
vertex .0391557 .185097 -.314566
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0391557 .184322 -.31689
vertex .0391557 .185097 -.314566
vertex .0374236 .183548 -.31534
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0374236 .185871 -.316115
vertex .0374236 .183548 -.31534
vertex .0391557 .185097 -.314566
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.133299 .205778 -.359827
vertex -.135031 .205003 -.358278
vertex -.135031 .207327 -.359053
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.133299 .205778 -.359827
vertex -.135031 .207327 -.359053
vertex -.133299 .206553 -.357504
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.133299 .205778 -.359827
vertex -.133299 .206553 -.357504
vertex -.135031 .205003 -.358278
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.135031 .207327 -.359053
vertex -.135031 .205003 -.358278
vertex -.133299 .206553 -.357504
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.110051 .209771 -.366775
vertex -.111783 .208997 -.365226
vertex -.111783 .211321 -.366
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.110051 .209771 -.366775
vertex -.111783 .211321 -.366
vertex -.110051 .210546 -.364451
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.110051 .209771 -.366775
vertex -.110051 .210546 -.364451
vertex -.111783 .208997 -.365226
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.111783 .211321 -.366
vertex -.111783 .208997 -.365226
vertex -.110051 .210546 -.364451
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0664125 .20478 -.35809
vertex -.0681446 .204005 -.356541
vertex -.0681446 .206329 -.357316
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0664125 .20478 -.35809
vertex -.0681446 .206329 -.357316
vertex -.0664125 .205554 -.355767
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0664125 .20478 -.35809
vertex -.0664125 .205554 -.355767
vertex -.0681446 .204005 -.356541
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0681446 .206329 -.357316
vertex -.0681446 .204005 -.356541
vertex -.0664125 .205554 -.355767
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0268814 .174329 -.305116
vertex -.0286134 .173554 -.303566
vertex -.0286134 .175878 -.304341
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0268814 .174329 -.305116
vertex -.0286134 .175878 -.304341
vertex -.0268814 .175104 -.302792
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0268814 .174329 -.305116
vertex -.0268814 .175104 -.302792
vertex -.0286134 .173554 -.303566
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0286134 .175878 -.304341
vertex -.0286134 .173554 -.303566
vertex -.0268814 .175104 -.302792
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00259524 .168838 -.295563
vertex -.00432729 .168063 -.294014
vertex -.00432729 .170387 -.294788
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00259524 .168838 -.295563
vertex -.00432729 .170387 -.294788
vertex -.00259524 .169612 -.293239
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00259524 .168838 -.295563
vertex -.00259524 .169612 -.293239
vertex -.00432729 .168063 -.294014
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00432729 .170387 -.294788
vertex -.00432729 .168063 -.294014
vertex -.00259524 .169612 -.293239
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0185244 .171833 -.300773
vertex .0167923 .171058 -.299224
vertex .0167923 .173382 -.299999
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0185244 .171833 -.300773
vertex .0167923 .173382 -.299999
vertex .0185244 .172608 -.29845
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0185244 .171833 -.300773
vertex .0185244 .172608 -.29845
vertex .0167923 .171058 -.299224
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0167923 .173382 -.299999
vertex .0167923 .171058 -.299224
vertex .0185244 .172608 -.29845
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0425197 .184313 -.322484
vertex .0407876 .183538 -.320935
vertex .0407876 .185862 -.32171
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0425197 .184313 -.322484
vertex .0407876 .185862 -.32171
vertex .0425197 .185087 -.320161
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0425197 .184313 -.322484
vertex .0425197 .185087 -.320161
vertex .0407876 .183538 -.320935
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0407876 .185862 -.32171
vertex .0407876 .183538 -.320935
vertex .0425197 .185087 -.320161
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.133299 .202943 -.361245
vertex -.135031 .202168 -.359696
vertex -.135031 .204492 -.36047
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.133299 .202943 -.361245
vertex -.135031 .204492 -.36047
vertex -.133299 .203718 -.358921
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.133299 .202943 -.361245
vertex -.133299 .203718 -.358921
vertex -.135031 .202168 -.359696
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.135031 .204492 -.36047
vertex -.135031 .202168 -.359696
vertex -.133299 .203718 -.358921
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.110051 .206882 -.36822
vertex -.111783 .206107 -.366671
vertex -.111783 .208431 -.367445
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.110051 .206882 -.36822
vertex -.111783 .208431 -.367445
vertex -.110051 .207656 -.365896
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.110051 .206882 -.36822
vertex -.110051 .207656 -.365896
vertex -.111783 .206107 -.366671
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.111783 .208431 -.367445
vertex -.111783 .206107 -.366671
vertex -.110051 .207656 -.365896
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0699094 .202943 -.361245
vertex -.0716414 .202168 -.359696
vertex -.0716414 .204492 -.36047
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0699094 .202943 -.361245
vertex -.0716414 .204492 -.36047
vertex -.0699094 .203718 -.358921
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0699094 .202943 -.361245
vertex -.0699094 .203718 -.358921
vertex -.0716414 .202168 -.359696
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0716414 .204492 -.36047
vertex -.0716414 .202168 -.359696
vertex -.0699094 .203718 -.358921
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0211898 .169465 -.301958
vertex -.0229218 .16869 -.300408
vertex -.0229218 .171014 -.301183
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0211898 .169465 -.301958
vertex -.0229218 .171014 -.301183
vertex -.0211898 .17024 -.299634
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0211898 .169465 -.301958
vertex -.0211898 .17024 -.299634
vertex -.0229218 .16869 -.300408
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0229218 .171014 -.301183
vertex -.0229218 .16869 -.300408
vertex -.0211898 .17024 -.299634
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00258734 .164542 -.293239
vertex .000855287 .163767 -.29169
vertex .000855287 .166091 -.292464
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00258734 .164542 -.293239
vertex .000855287 .166091 -.292464
vertex .00258734 .165316 -.290915
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00258734 .164542 -.293239
vertex .00258734 .165316 -.290915
vertex .000855287 .163767 -.29169
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .000855287 .166091 -.292464
vertex .000855287 .163767 -.29169
vertex .00258734 .165316 -.290915
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0243518 .173404 -.308932
vertex .0226198 .172629 -.307383
vertex .0226198 .174953 -.308158
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0243518 .173404 -.308932
vertex .0226198 .174953 -.308158
vertex .0243518 .174178 -.306609
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0243518 .173404 -.308932
vertex .0243518 .174178 -.306609
vertex .0226198 .172629 -.307383
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0226198 .174953 -.308158
vertex .0226198 .172629 -.307383
vertex .0243518 .174178 -.306609
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0494779 .186696 -.332473
vertex .0477458 .185922 -.330924
vertex .0477458 .188246 -.331698
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0494779 .186696 -.332473
vertex .0477458 .188246 -.331698
vertex .0494779 .187471 -.330149
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0494779 .186696 -.332473
vertex .0494779 .187471 -.330149
vertex .0477458 .185922 -.330924
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0477458 .188246 -.331698
vertex .0477458 .185922 -.330924
vertex .0494779 .187471 -.330149
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.127888 .201565 -.365288
vertex -.12962 .20079 -.363739
vertex -.12962 .203114 -.364513
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.127888 .201565 -.365288
vertex -.12962 .203114 -.364513
vertex -.127888 .202339 -.362964
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.127888 .201565 -.365288
vertex -.127888 .202339 -.362964
vertex -.12962 .20079 -.363739
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.12962 .203114 -.364513
vertex -.12962 .20079 -.363739
vertex -.127888 .202339 -.362964
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.104085 .204963 -.371415
vertex -.105817 .204188 -.369866
vertex -.105817 .206512 -.370641
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.104085 .204963 -.371415
vertex -.105817 .206512 -.370641
vertex -.104085 .205737 -.369091
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.104085 .204963 -.371415
vertex -.104085 .205737 -.369091
vertex -.105817 .204188 -.369866
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.105817 .206512 -.370641
vertex -.105817 .204188 -.369866
vertex -.104085 .205737 -.369091
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0632584 .199137 -.360912
vertex -.0649905 .198363 -.359362
vertex -.0649905 .200686 -.360137
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0632584 .199137 -.360912
vertex -.0649905 .200686 -.360137
vertex -.0632584 .199912 -.358588
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0632584 .199137 -.360912
vertex -.0632584 .199912 -.358588
vertex -.0649905 .198363 -.359362
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0649905 .200686 -.360137
vertex -.0649905 .198363 -.359362
vertex -.0632584 .199912 -.358588
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0130833 .165155 -.29964
vertex -.0148154 .16438 -.298091
vertex -.0148154 .166704 -.298866
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0130833 .165155 -.29964
vertex -.0148154 .166704 -.298866
vertex -.0130833 .16593 -.297317
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0130833 .165155 -.29964
vertex -.0130833 .16593 -.297317
vertex -.0148154 .16438 -.298091
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0148154 .166704 -.298866
vertex -.0148154 .16438 -.298091
vertex -.0130833 .16593 -.297317
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0104407 .16467 -.298765
vertex .00870869 .163895 -.297216
vertex .00870869 .166219 -.29799
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0104407 .16467 -.298765
vertex .00870869 .166219 -.29799
vertex .0104407 .165444 -.296441
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0104407 .16467 -.298765
vertex .0104407 .165444 -.296441
vertex .00870869 .163895 -.297216
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00870869 .166219 -.29799
vertex .00870869 .163895 -.297216
vertex .0104407 .165444 -.296441
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0335535 .176806 -.320648
vertex .0318215 .176031 -.319098
vertex .0318215 .178355 -.319873
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0335535 .176806 -.320648
vertex .0318215 .178355 -.319873
vertex .0335535 .177581 -.318324
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0335535 .176806 -.320648
vertex .0335535 .177581 -.318324
vertex .0318215 .176031 -.319098
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0318215 .178355 -.319873
vertex .0318215 .176031 -.319098
vertex .0335535 .177581 -.318324
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.145265 .196316 -.362322
vertex -.146997 .195542 -.360773
vertex -.146997 .197865 -.361548
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.145265 .196316 -.362322
vertex -.146997 .197865 -.361548
vertex -.145265 .197091 -.359998
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.145265 .196316 -.362322
vertex -.145265 .197091 -.359998
vertex -.146997 .195542 -.360773
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.146997 .197865 -.361548
vertex -.146997 .195542 -.360773
vertex -.145265 .197091 -.359998
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.121797 .199188 -.367595
vertex -.123529 .198413 -.366045
vertex -.123529 .200737 -.36682
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.121797 .199188 -.367595
vertex -.123529 .200737 -.36682
vertex -.121797 .199962 -.365271
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.121797 .199188 -.367595
vertex -.121797 .199962 -.365271
vertex -.123529 .198413 -.366045
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.123529 .200737 -.36682
vertex -.123529 .198413 -.366045
vertex -.121797 .199962 -.365271
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0978254 .202538 -.373746
vertex -.0995575 .201763 -.372197
vertex -.0995575 .204087 -.372971
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0978254 .202538 -.373746
vertex -.0995575 .204087 -.372971
vertex -.0978254 .203313 -.371422
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0978254 .202538 -.373746
vertex -.0978254 .203313 -.371422
vertex -.0995575 .201763 -.372197
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0995575 .204087 -.372971
vertex -.0995575 .201763 -.372197
vertex -.0978254 .203313 -.371422
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0515351 .18818 -.347384
vertex -.0532672 .187405 -.345834
vertex -.0532672 .189729 -.346609
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0515351 .18818 -.347384
vertex -.0532672 .189729 -.346609
vertex -.0515351 .188955 -.34506
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0515351 .18818 -.347384
vertex -.0515351 .188955 -.34506
vertex -.0532672 .187405 -.345834
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0532672 .189729 -.346609
vertex -.0532672 .187405 -.345834
vertex -.0515351 .188955 -.34506
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00512526 .159943 -.295538
vertex -.00685731 .159168 -.293989
vertex -.00685731 .161492 -.294764
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00512526 .159943 -.295538
vertex -.00685731 .161492 -.294764
vertex -.00512526 .160717 -.293214
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00512526 .159943 -.295538
vertex -.00512526 .160717 -.293214
vertex -.00685731 .159168 -.293989
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00685731 .161492 -.294764
vertex -.00685731 .159168 -.293989
vertex -.00512526 .160717 -.293214
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0188315 .1676 -.309598
vertex .0170994 .166826 -.308049
vertex .0170994 .16915 -.308823
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0188315 .1676 -.309598
vertex .0170994 .16915 -.308823
vertex .0188315 .168375 -.307274
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0188315 .1676 -.309598
vertex .0188315 .168375 -.307274
vertex .0170994 .166826 -.308049
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0170994 .16915 -.308823
vertex .0170994 .166826 -.308049
vertex .0188315 .168375 -.307274
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0433077 .180044 -.332445
vertex .0415757 .179269 -.330896
vertex .0415757 .181593 -.331671
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0433077 .180044 -.332445
vertex .0415757 .181593 -.331671
vertex .0433077 .180818 -.330121
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0433077 .180044 -.332445
vertex .0433077 .180818 -.330121
vertex .0415757 .179269 -.330896
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0415757 .181593 -.331671
vertex .0415757 .179269 -.330896
vertex .0433077 .180818 -.330121
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.136135 .193967 -.364615
vertex -.137867 .193192 -.363066
vertex -.137867 .195516 -.36384
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.136135 .193967 -.364615
vertex -.137867 .195516 -.36384
vertex -.136135 .194741 -.362291
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.136135 .193967 -.364615
vertex -.136135 .194741 -.362291
vertex -.137867 .193192 -.363066
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.137867 .195516 -.36384
vertex -.137867 .193192 -.363066
vertex -.136135 .194741 -.362291
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.112739 .197269 -.37079
vertex -.114471 .196494 -.369241
vertex -.114471 .198818 -.370015
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.112739 .197269 -.37079
vertex -.114471 .198818 -.370015
vertex -.112739 .198044 -.368466
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.112739 .197269 -.37079
vertex -.112739 .198044 -.368466
vertex -.114471 .196494 -.369241
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.114471 .198818 -.370015
vertex -.114471 .196494 -.369241
vertex -.112739 .198044 -.368466
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0666534 .185476 -.348736
vertex -.0683854 .184701 -.347187
vertex -.0683854 .187025 -.347961
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0666534 .185476 -.348736
vertex -.0683854 .187025 -.347961
vertex -.0666534 .18625 -.346412
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0666534 .185476 -.348736
vertex -.0666534 .18625 -.346412
vertex -.0683854 .184701 -.347187
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0683854 .187025 -.347961
vertex -.0683854 .184701 -.347187
vertex -.0666534 .18625 -.346412
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0103002 .158115 -.29757
vertex -.0120323 .15734 -.296021
vertex -.0120323 .159664 -.296796
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0103002 .158115 -.29757
vertex -.0120323 .159664 -.296796
vertex -.0103002 .15889 -.295246
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0103002 .158115 -.29757
vertex -.0103002 .15889 -.295246
vertex -.0120323 .15734 -.296021
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0120323 .159664 -.296796
vertex -.0120323 .15734 -.296021
vertex -.0103002 .15889 -.295246
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0105534 .161889 -.304627
vertex .00882133 .161114 -.303078
vertex .00882133 .163438 -.303853
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0105534 .161889 -.304627
vertex .00882133 .163438 -.303853
vertex .0105534 .162664 -.302304
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0105534 .161889 -.304627
vertex .0105534 .162664 -.302304
vertex .00882133 .161114 -.303078
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00882133 .163438 -.303853
vertex .00882133 .161114 -.303078
vertex .0105534 .162664 -.302304
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0371324 .175569 -.33021
vertex .0354004 .174795 -.328661
vertex .0354004 .177118 -.329436
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0371324 .175569 -.33021
vertex .0354004 .177118 -.329436
vertex .0371324 .176344 -.327887
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0371324 .175569 -.33021
vertex .0371324 .176344 -.327887
vertex .0354004 .174795 -.328661
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0354004 .177118 -.329436
vertex .0354004 .174795 -.328661
vertex .0371324 .176344 -.327887
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.144554 .189744 -.363372
vertex -.146286 .18897 -.361823
vertex -.146286 .191293 -.362597
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.144554 .189744 -.363372
vertex -.146286 .191293 -.362597
vertex -.144554 .190519 -.361048
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.144554 .189744 -.363372
vertex -.144554 .190519 -.361048
vertex -.146286 .18897 -.361823
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.146286 .191293 -.362597
vertex -.146286 .18897 -.361823
vertex -.144554 .190519 -.361048
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.10569 .182306 -.349202
vertex -.107422 .181532 -.347653
vertex -.107422 .183855 -.348428
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.10569 .182306 -.349202
vertex -.107422 .183855 -.348428
vertex -.10569 .183081 -.346879
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.10569 .182306 -.349202
vertex -.10569 .183081 -.346879
vertex -.107422 .181532 -.347653
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.107422 .183855 -.348428
vertex -.107422 .181532 -.347653
vertex -.10569 .183081 -.346879
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0807229 .180447 -.34566
vertex -.0824549 .179672 -.344111
vertex -.0824549 .181996 -.344885
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0807229 .180447 -.34566
vertex -.0824549 .181996 -.344885
vertex -.0807229 .181221 -.343336
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0807229 .180447 -.34566
vertex -.0807229 .181221 -.343336
vertex -.0824549 .179672 -.344111
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0824549 .181996 -.344885
vertex -.0824549 .179672 -.344111
vertex -.0807229 .181221 -.343336
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0468823 .176728 -.338575
vertex -.0486143 .175953 -.337026
vertex -.0486143 .178277 -.337801
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0468823 .176728 -.338575
vertex -.0486143 .178277 -.337801
vertex -.0468823 .177502 -.336251
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0468823 .176728 -.338575
vertex -.0468823 .177502 -.336251
vertex -.0486143 .175953 -.337026
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0486143 .178277 -.337801
vertex -.0486143 .175953 -.337026
vertex -.0468823 .177502 -.336251
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00259248 .155809 -.298723
vertex .000860425 .155034 -.297174
vertex .000860425 .157358 -.297949
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00259248 .155809 -.298723
vertex .000860425 .157358 -.297949
vertex .00259248 .156583 -.2964
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00259248 .155809 -.298723
vertex .00259248 .156583 -.2964
vertex .000860425 .155034 -.297174
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .000860425 .157358 -.297949
vertex .000860425 .155034 -.297174
vertex .00259248 .156583 -.2964
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0248841 .16743 -.320863
vertex .0231521 .166656 -.319314
vertex .0231521 .16898 -.320089
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0248841 .16743 -.320863
vertex .0231521 .16898 -.320089
vertex .0248841 .168205 -.31854
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0248841 .16743 -.320863
vertex .0248841 .168205 -.31854
vertex .0231521 .166656 -.319314
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0231521 .16898 -.320089
vertex .0231521 .166656 -.319314
vertex .0248841 .168205 -.31854
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.156337 .186021 -.362998
vertex -.158069 .185246 -.361448
vertex -.158069 .18757 -.362223
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.156337 .186021 -.362998
vertex -.158069 .18757 -.362223
vertex -.156337 .186795 -.360674
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.156337 .186021 -.362998
vertex -.156337 .186795 -.360674
vertex -.158069 .185246 -.361448
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.158069 .18757 -.362223
vertex -.158069 .185246 -.361448
vertex -.156337 .186795 -.360674
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.113561 .177777 -.346995
vertex -.115293 .177002 -.345446
vertex -.115293 .179326 -.346221
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.113561 .177777 -.346995
vertex -.115293 .179326 -.346221
vertex -.113561 .178551 -.344671
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.113561 .177777 -.346995
vertex -.113561 .178551 -.344671
vertex -.115293 .177002 -.345446
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.115293 .179326 -.346221
vertex -.115293 .177002 -.345446
vertex -.113561 .178551 -.344671
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0698999 .173197 -.338105
vertex -.0716319 .172422 -.336556
vertex -.0716319 .174746 -.33733
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0698999 .173197 -.338105
vertex -.0716319 .174746 -.33733
vertex -.0698999 .173971 -.335781
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0698999 .173197 -.338105
vertex -.0698999 .173971 -.335781
vertex -.0716319 .172422 -.336556
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0716319 .174746 -.33733
vertex -.0716319 .172422 -.336556
vertex -.0698999 .173971 -.335781
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0339566 .168159 -.328325
vertex -.0356887 .167384 -.326776
vertex -.0356887 .169708 -.327551
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0339566 .168159 -.328325
vertex -.0356887 .169708 -.327551
vertex -.0339566 .168933 -.326002
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0339566 .168159 -.328325
vertex -.0339566 .168933 -.326002
vertex -.0356887 .167384 -.326776
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0356887 .169708 -.327551
vertex -.0356887 .167384 -.326776
vertex -.0339566 .168933 -.326002
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0106379 .15854 -.309656
vertex .00890582 .157766 -.308107
vertex .00890582 .16009 -.308881
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0106379 .15854 -.309656
vertex .00890582 .16009 -.308881
vertex .0106379 .159315 -.307332
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0106379 .15854 -.309656
vertex .0106379 .159315 -.307332
vertex .00890582 .157766 -.308107
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00890582 .16009 -.308881
vertex .00890582 .157766 -.308107
vertex .0106379 .159315 -.307332
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0344491 .171365 -.334549
vertex .032717 .17059 -.333
vertex .032717 .172914 -.333774
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0344491 .171365 -.334549
vertex .032717 .172914 -.333774
vertex .0344491 .172139 -.332225
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0344491 .171365 -.334549
vertex .0344491 .172139 -.332225
vertex .032717 .17059 -.333
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .032717 .172914 -.333774
vertex .032717 .17059 -.333
vertex .0344491 .172139 -.332225
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.150461 .183678 -.365287
vertex -.152193 .182904 -.363738
vertex -.152193 .185227 -.364512
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.150461 .183678 -.365287
vertex -.152193 .185227 -.364512
vertex -.150461 .184453 -.362963
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.150461 .183678 -.365287
vertex -.150461 .184453 -.362963
vertex -.152193 .182904 -.363738
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.152193 .185227 -.364512
vertex -.152193 .182904 -.363738
vertex -.150461 .184453 -.362963
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.105918 .1724 -.342975
vertex -.10765 .171625 -.341426
vertex -.10765 .173949 -.342201
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.105918 .1724 -.342975
vertex -.10765 .173949 -.342201
vertex -.105918 .173174 -.340652
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.105918 .1724 -.342975
vertex -.105918 .173174 -.340652
vertex -.10765 .171625 -.341426
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.10765 .173949 -.342201
vertex -.10765 .171625 -.341426
vertex -.105918 .173174 -.340652
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0611696 .162926 -.324234
vertex -.0629016 .162151 -.322684
vertex -.0629016 .164475 -.323459
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0611696 .162926 -.324234
vertex -.0629016 .164475 -.323459
vertex -.0611696 .1637 -.32191
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0611696 .162926 -.324234
vertex -.0611696 .1637 -.32191
vertex -.0629016 .162151 -.322684
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0629016 .164475 -.323459
vertex -.0629016 .162151 -.322684
vertex -.0611696 .1637 -.32191
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0328211 .160219 -.318879
vertex -.0345531 .159444 -.31733
vertex -.0345531 .161768 -.318104
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0328211 .160219 -.318879
vertex -.0345531 .161768 -.318104
vertex -.0328211 .160994 -.316555
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0328211 .160219 -.318879
vertex -.0328211 .160994 -.316555
vertex -.0345531 .159444 -.31733
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0345531 .161768 -.318104
vertex -.0345531 .159444 -.31733
vertex -.0328211 .160994 -.316555
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0107505 .157963 -.314417
vertex .00901846 .157189 -.312867
vertex .00901846 .159512 -.313642
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0107505 .157963 -.314417
vertex .00901846 .159512 -.313642
vertex .0107505 .158738 -.312093
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0107505 .157963 -.314417
vertex .0107505 .158738 -.312093
vertex .00901846 .157189 -.312867
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00901846 .159512 -.313642
vertex .00901846 .157189 -.312867
vertex .0107505 .158738 -.312093
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0348073 .170595 -.339406
vertex .0330752 .169821 -.337856
vertex .0330752 .172144 -.338631
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0348073 .170595 -.339406
vertex .0330752 .172144 -.338631
vertex .0348073 .17137 -.337082
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0348073 .170595 -.339406
vertex .0348073 .17137 -.337082
vertex .0330752 .169821 -.337856
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0330752 .172144 -.338631
vertex .0330752 .169821 -.337856
vertex .0348073 .17137 -.337082
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.150461 .180878 -.366687
vertex -.152193 .180103 -.365138
vertex -.152193 .182427 -.365913
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.150461 .180878 -.366687
vertex -.152193 .182427 -.365913
vertex -.150461 .181652 -.364363
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.150461 .180878 -.366687
vertex -.150461 .181652 -.364363
vertex -.152193 .180103 -.365138
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.152193 .182427 -.365913
vertex -.152193 .180103 -.365138
vertex -.150461 .181652 -.364363
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0999958 .164884 -.334435
vertex -.101728 .164109 -.332886
vertex -.101728 .166433 -.33366
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0999958 .164884 -.334435
vertex -.101728 .166433 -.33366
vertex -.0999958 .165658 -.332111
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0999958 .164884 -.334435
vertex -.0999958 .165658 -.332111
vertex -.101728 .164109 -.332886
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.101728 .166433 -.33366
vertex -.101728 .164109 -.332886
vertex -.0999958 .165658 -.332111
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0659788 .158664 -.321892
vertex -.0677109 .157889 -.320343
vertex -.0677109 .160213 -.321118
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0659788 .158664 -.321892
vertex -.0677109 .160213 -.321118
vertex -.0659788 .159438 -.319569
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0659788 .158664 -.321892
vertex -.0659788 .159438 -.319569
vertex -.0677109 .157889 -.320343
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0677109 .160213 -.321118
vertex -.0677109 .157889 -.320343
vertex -.0659788 .159438 -.319569
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0436239 .157331 -.319205
vertex -.0453559 .156556 -.317656
vertex -.0453559 .15888 -.31843
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0436239 .157331 -.319205
vertex -.0453559 .15888 -.31843
vertex -.0436239 .158106 -.316881
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0436239 .157331 -.319205
vertex -.0436239 .158106 -.316881
vertex -.0453559 .156556 -.317656
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0453559 .15888 -.31843
vertex -.0453559 .156556 -.317656
vertex -.0436239 .158106 -.316881
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0116628 .167105 -.338914
vertex -.0133948 .166331 -.337365
vertex -.0133948 .168654 -.33814
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0116628 .167105 -.338914
vertex -.0133948 .168654 -.33814
vertex -.0116628 .16788 -.336591
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0116628 .167105 -.338914
vertex -.0116628 .16788 -.336591
vertex -.0133948 .166331 -.337365
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0133948 .168654 -.33814
vertex -.0133948 .166331 -.337365
vertex -.0116628 .16788 -.336591
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0194457 .160885 -.326372
vertex .0177136 .160111 -.324823
vertex .0177136 .162434 -.325597
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0194457 .160885 -.326372
vertex .0177136 .162434 -.325597
vertex .0194457 .16166 -.324048
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0194457 .160885 -.326372
vertex .0194457 .16166 -.324048
vertex .0177136 .160111 -.324823
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0177136 .162434 -.325597
vertex .0177136 .160111 -.324823
vertex .0194457 .16166 -.324048
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0448838 .173325 -.351457
vertex .0431518 .17255 -.349908
vertex .0431518 .174874 -.350682
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0448838 .173325 -.351457
vertex .0431518 .174874 -.350682
vertex .0448838 .1741 -.349133
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0448838 .173325 -.351457
vertex .0448838 .1741 -.349133
vertex .0431518 .17255 -.349908
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0431518 .174874 -.350682
vertex .0431518 .17255 -.349908
vertex .0448838 .1741 -.349133
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.114587 .158831 -.328517
vertex -.116319 .158056 -.326968
vertex -.116319 .16038 -.327743
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.114587 .158831 -.328517
vertex -.116319 .16038 -.327743
vertex -.114587 .159606 -.326193
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.114587 .158831 -.328517
vertex -.114587 .159606 -.326193
vertex -.116319 .158056 -.326968
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.116319 .16038 -.327743
vertex -.116319 .158056 -.326968
vertex -.114587 .159606 -.326193
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0856913 .157081 -.32492
vertex -.0874234 .156307 -.323371
vertex -.0874234 .158631 -.324145
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0856913 .157081 -.32492
vertex -.0874234 .158631 -.324145
vertex -.0856913 .157856 -.322596
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0856913 .157081 -.32492
vertex -.0856913 .157856 -.322596
vertex -.0874234 .156307 -.323371
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0874234 .158631 -.324145
vertex -.0874234 .156307 -.323371
vertex -.0856913 .157856 -.322596
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.05997 .154894 -.320423
vertex -.061702 .15412 -.318874
vertex -.061702 .156444 -.319649
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.05997 .154894 -.320423
vertex -.061702 .156444 -.319649
vertex -.05997 .155669 -.318099
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.05997 .154894 -.320423
vertex -.05997 .155669 -.318099
vertex -.061702 .15412 -.318874
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.061702 .156444 -.319649
vertex -.061702 .15412 -.318874
vertex -.05997 .155669 -.318099
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0379552 .15402 -.318625
vertex -.0396873 .153245 -.317075
vertex -.0396873 .155569 -.31785
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0379552 .15402 -.318625
vertex -.0396873 .155569 -.31785
vertex -.0379552 .154794 -.316301
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0379552 .15402 -.318625
vertex -.0379552 .154794 -.316301
vertex -.0396873 .153245 -.317075
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0396873 .155569 -.31785
vertex -.0396873 .153245 -.317075
vertex -.0379552 .154794 -.316301
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -8.88888e-6 .15052 -.31143
vertex -.00174094 .149746 -.309881
vertex -.00174094 .152069 -.310655
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -8.88888e-6 .15052 -.31143
vertex -.00174094 .152069 -.310655
vertex -8.88888e-6 .151295 -.309106
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -8.88888e-6 .15052 -.31143
vertex -8.88888e-6 .151295 -.309106
vertex -.00174094 .149746 -.309881
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00174094 .152069 -.310655
vertex -.00174094 .149746 -.309881
vertex -8.88888e-6 .151295 -.309106
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0225257 .160581 -.332114
vertex .0207937 .159806 -.330565
vertex .0207937 .16213 -.33134
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0225257 .160581 -.332114
vertex .0207937 .16213 -.33134
vertex .0225257 .161355 -.329791
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0225257 .160581 -.332114
vertex .0225257 .161355 -.329791
vertex .0207937 .159806 -.330565
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0207937 .16213 -.33134
vertex .0207937 .159806 -.330565
vertex .0225257 .161355 -.329791
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0483654 .172391 -.356396
vertex .0466334 .171616 -.354847
vertex .0466334 .17394 -.355622
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0483654 .172391 -.356396
vertex .0466334 .17394 -.355622
vertex .0483654 .173165 -.354072
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0483654 .172391 -.356396
vertex .0483654 .173165 -.354072
vertex .0466334 .171616 -.354847
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0466334 .17394 -.355622
vertex .0466334 .171616 -.354847
vertex .0483654 .173165 -.354072
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.112367 .153319 -.323447
vertex -.114099 .152544 -.321898
vertex -.114099 .154868 -.322672
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.112367 .153319 -.323447
vertex -.114099 .154868 -.322672
vertex -.112367 .154093 -.321123
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.112367 .153319 -.323447
vertex -.112367 .154093 -.321123
vertex -.114099 .152544 -.321898
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.114099 .154868 -.322672
vertex -.114099 .152544 -.321898
vertex -.112367 .154093 -.321123
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0899378 .152458 -.321642
vertex -.0916699 .151683 -.320092
vertex -.0916699 .154007 -.320867
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0899378 .152458 -.321642
vertex -.0916699 .154007 -.320867
vertex -.0899378 .153232 -.319318
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0899378 .152458 -.321642
vertex -.0899378 .153232 -.319318
vertex -.0916699 .151683 -.320092
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0916699 .154007 -.320867
vertex -.0916699 .151683 -.320092
vertex -.0899378 .153232 -.319318
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0650452 .151596 -.319836
vertex -.0667773 .150822 -.318287
vertex -.0667773 .153146 -.319061
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0650452 .151596 -.319836
vertex -.0667773 .153146 -.319061
vertex -.0650452 .152371 -.317512
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0650452 .151596 -.319836
vertex -.0650452 .152371 -.317512
vertex -.0667773 .150822 -.318287
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0667773 .153146 -.319061
vertex -.0667773 .150822 -.318287
vertex -.0650452 .152371 -.317512
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0434985 .152027 -.320739
vertex -.0452306 .151252 -.31919
vertex -.0452306 .153576 -.319964
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0434985 .152027 -.320739
vertex -.0452306 .153576 -.319964
vertex -.0434985 .152802 -.318415
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0434985 .152027 -.320739
vertex -.0434985 .152802 -.318415
vertex -.0452306 .151252 -.31919
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0452306 .153576 -.319964
vertex -.0452306 .151252 -.31919
vertex -.0434985 .152802 -.318415
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0174491 .16193 -.341502
vertex -.0191812 .161155 -.339953
vertex -.0191812 .163479 -.340728
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0174491 .16193 -.341502
vertex -.0191812 .163479 -.340728
vertex -.0174491 .162704 -.339178
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0174491 .16193 -.341502
vertex -.0174491 .162704 -.339178
vertex -.0191812 .161155 -.339953
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0191812 .163479 -.340728
vertex -.0191812 .161155 -.339953
vertex -.0174491 .162704 -.339178
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0081388 .152458 -.321642
vertex .00640675 .151683 -.320092
vertex .00640675 .154007 -.320867
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0081388 .152458 -.321642
vertex .00640675 .154007 -.320867
vertex .0081388 .153232 -.319318
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0081388 .152458 -.321642
vertex .0081388 .153232 -.319318
vertex .00640675 .151683 -.320092
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00640675 .154007 -.320867
vertex .00640675 .151683 -.320092
vertex .0081388 .153232 -.319318
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0320625 .163652 -.345113
vertex .0303305 .162877 -.343564
vertex .0303305 .165201 -.344339
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0320625 .163652 -.345113
vertex .0303305 .165201 -.344339
vertex .0320625 .164426 -.342789
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0320625 .163652 -.345113
vertex .0320625 .164426 -.342789
vertex .0303305 .162877 -.343564
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0303305 .165201 -.344339
vertex .0303305 .162877 -.343564
vertex .0320625 .164426 -.342789
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.12263 .150021 -.32286
vertex -.124362 .149246 -.321311
vertex -.124362 .15157 -.322085
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.12263 .150021 -.32286
vertex -.124362 .15157 -.322085
vertex -.12263 .150795 -.320536
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.12263 .150021 -.32286
vertex -.12263 .150795 -.320536
vertex -.124362 .149246 -.321311
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.124362 .15157 -.322085
vertex -.124362 .149246 -.321311
vertex -.12263 .150795 -.320536
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0999758 .14875 -.320141
vertex -.101708 .147975 -.318592
vertex -.101708 .150299 -.319367
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0999758 .14875 -.320141
vertex -.101708 .150299 -.319367
vertex -.0999758 .149524 -.317818
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0999758 .14875 -.320141
vertex -.0999758 .149524 -.317818
vertex -.101708 .147975 -.318592
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.101708 .150299 -.319367
vertex -.101708 .147975 -.318592
vertex -.0999758 .149524 -.317818
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0758813 .149173 -.321048
vertex -.0776133 .148399 -.319498
vertex -.0776133 .150723 -.320273
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0758813 .149173 -.321048
vertex -.0776133 .150723 -.320273
vertex -.0758813 .149948 -.318724
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0758813 .149173 -.321048
vertex -.0758813 .149948 -.318724
vertex -.0776133 .148399 -.319498
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0776133 .150723 -.320273
vertex -.0776133 .148399 -.319498
vertex -.0758813 .149948 -.318724
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0545213 .150021 -.32286
vertex -.0562533 .149246 -.321311
vertex -.0562533 .15157 -.322085
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0545213 .150021 -.32286
vertex -.0562533 .15157 -.322085
vertex -.0545213 .150795 -.320536
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0545213 .150021 -.32286
vertex -.0545213 .150795 -.320536
vertex -.0562533 .149246 -.321311
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0562533 .15157 -.322085
vertex -.0562533 .149246 -.321311
vertex -.0545213 .150795 -.320536
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0327265 .150021 -.32286
vertex -.0344585 .149246 -.321311
vertex -.0344585 .15157 -.322085
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0327265 .150021 -.32286
vertex -.0344585 .15157 -.322085
vertex -.0327265 .150795 -.320536
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0327265 .150021 -.32286
vertex -.0327265 .150795 -.320536
vertex -.0344585 .149246 -.321311
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0344585 .15157 -.322085
vertex -.0344585 .149246 -.321311
vertex -.0327265 .150795 -.320536
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00866736 .157647 -.339171
vertex -.0103994 .156873 -.337622
vertex -.0103994 .159196 -.338397
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00866736 .157647 -.339171
vertex -.0103994 .159196 -.338397
vertex -.00866736 .158422 -.336848
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00866736 .157647 -.339171
vertex -.00866736 .158422 -.336848
vertex -.0103994 .156873 -.337622
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0103994 .159196 -.338397
vertex -.0103994 .156873 -.337622
vertex -.00866736 .158422 -.336848
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .016834 .155105 -.333734
vertex .0151019 .15433 -.332185
vertex .0151019 .156654 -.33296
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .016834 .155105 -.333734
vertex .0151019 .156654 -.33296
vertex .016834 .15588 -.33141
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .016834 .155105 -.333734
vertex .016834 .15588 -.33141
vertex .0151019 .15433 -.332185
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0151019 .156654 -.33296
vertex .0151019 .15433 -.332185
vertex .016834 .15588 -.33141
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0425126 .167816 -.36092
vertex .0407806 .167041 -.359371
vertex .0407806 .169365 -.360145
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0425126 .167816 -.36092
vertex .0407806 .169365 -.360145
vertex .0425126 .16859 -.358596
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0425126 .167816 -.36092
vertex .0425126 .16859 -.358596
vertex .0407806 .167041 -.359371
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0407806 .169365 -.360145
vertex .0407806 .167041 -.359371
vertex .0425126 .16859 -.358596
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.113482 .146334 -.321349
vertex -.115214 .145559 -.3198
vertex -.115214 .147883 -.320575
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.113482 .146334 -.321349
vertex -.115214 .147883 -.320575
vertex -.113482 .147108 -.319026
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.113482 .146334 -.321349
vertex -.113482 .147108 -.319026
vertex -.115214 .145559 -.3198
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.115214 .147883 -.320575
vertex -.115214 .145559 -.3198
vertex -.113482 .147108 -.319026
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.089682 .147167 -.323169
vertex -.0914141 .146393 -.321619
vertex -.0914141 .148716 -.322394
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.089682 .147167 -.323169
vertex -.0914141 .148716 -.322394
vertex -.089682 .147942 -.320845
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.089682 .147167 -.323169
vertex -.089682 .147942 -.320845
vertex -.0914141 .146393 -.321619
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0914141 .148716 -.322394
vertex -.0914141 .146393 -.321619
vertex -.089682 .147942 -.320845
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0679486 .147167 -.323169
vertex -.0696807 .146393 -.321619
vertex -.0696807 .148716 -.322394
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0679486 .147167 -.323169
vertex -.0696807 .148716 -.322394
vertex -.0679486 .147942 -.320845
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0679486 .147167 -.323169
vertex -.0679486 .147942 -.320845
vertex -.0696807 .146393 -.321619
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0696807 .148716 -.322394
vertex -.0696807 .146393 -.321619
vertex -.0679486 .147942 -.320845
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0464812 .148001 -.324988
vertex -.0482133 .147226 -.323439
vertex -.0482133 .14955 -.324213
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0464812 .148001 -.324988
vertex -.0482133 .14955 -.324213
vertex -.0464812 .148776 -.322664
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0464812 .148001 -.324988
vertex -.0464812 .148776 -.322664
vertex -.0482133 .147226 -.323439
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0482133 .14955 -.324213
vertex -.0482133 .147226 -.323439
vertex -.0464812 .148776 -.322664
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0262003 .157171 -.345
vertex -.0279324 .156396 -.34345
vertex -.0279324 .15872 -.344225
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0262003 .157171 -.345
vertex -.0279324 .15872 -.344225
vertex -.0262003 .157946 -.342676
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0262003 .157171 -.345
vertex -.0262003 .157946 -.342676
vertex -.0279324 .156396 -.34345
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0279324 .15872 -.344225
vertex -.0279324 .156396 -.34345
vertex -.0262003 .157946 -.342676
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00293217 .15467 -.339542
vertex -.00466422 .153895 -.337993
vertex -.00466422 .156219 -.338767
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00293217 .15467 -.339542
vertex -.00466422 .156219 -.338767
vertex -.00293217 .155445 -.337218
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00293217 .15467 -.339542
vertex -.00293217 .155445 -.337218
vertex -.00466422 .153895 -.337993
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00466422 .156219 -.338767
vertex -.00466422 .153895 -.337993
vertex -.00293217 .155445 -.337218
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0199063 .15467 -.339542
vertex .0181743 .153895 -.337993
vertex .0181743 .156219 -.338767
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0199063 .15467 -.339542
vertex .0181743 .156219 -.338767
vertex .0199063 .155445 -.337218
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0199063 .15467 -.339542
vertex .0199063 .155445 -.337218
vertex .0181743 .153895 -.337993
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0181743 .156219 -.338767
vertex .0181743 .153895 -.337993
vertex .0199063 .155445 -.337218
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0457844 .165924 -.364102
vertex .0440524 .16515 -.362553
vertex .0440524 .167473 -.363327
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0457844 .165924 -.364102
vertex .0440524 .167473 -.363327
vertex .0457844 .166699 -.361778
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0457844 .165924 -.364102
vertex .0457844 .166699 -.361778
vertex .0440524 .16515 -.362553
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0440524 .167473 -.363327
vertex .0440524 .16515 -.362553
vertex .0457844 .166699 -.361778
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.108389 .144327 -.323471
vertex -.110121 .143553 -.321921
vertex -.110121 .145877 -.322696
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.108389 .144327 -.323471
vertex -.110121 .145877 -.322696
vertex -.108389 .145102 -.321147
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.108389 .144327 -.323471
vertex -.108389 .145102 -.321147
vertex -.110121 .143553 -.321921
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.110121 .145877 -.322696
vertex -.110121 .143553 -.321921
vertex -.108389 .145102 -.321147
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0872135 .145147 -.325297
vertex -.0889455 .144373 -.323747
vertex -.0889455 .146697 -.324522
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0872135 .145147 -.325297
vertex -.0889455 .146697 -.324522
vertex -.0872135 .145922 -.322973
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0872135 .145147 -.325297
vertex -.0872135 .145922 -.322973
vertex -.0889455 .144373 -.323747
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0889455 .146697 -.324522
vertex -.0889455 .144373 -.323747
vertex -.0872135 .145922 -.322973
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0654187 .145147 -.325297
vertex -.0671507 .144373 -.323747
vertex -.0671507 .146697 -.324522
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0654187 .145147 -.325297
vertex -.0671507 .146697 -.324522
vertex -.0654187 .145922 -.322973
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0654187 .145147 -.325297
vertex -.0654187 .145922 -.322973
vertex -.0671507 .144373 -.323747
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0671507 .146697 -.324522
vertex -.0671507 .144373 -.323747
vertex -.0654187 .145922 -.322973
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0438745 .145967 -.327123
vertex -.0456066 .145193 -.325574
vertex -.0456066 .147517 -.326348
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0438745 .145967 -.327123
vertex -.0456066 .147517 -.326348
vertex -.0438745 .146742 -.324799
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0438745 .145967 -.327123
vertex -.0438745 .146742 -.324799
vertex -.0456066 .145193 -.325574
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0456066 .147517 -.326348
vertex -.0456066 .145193 -.325574
vertex -.0438745 .146742 -.324799
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0231077 .153347 -.343558
vertex -.0248397 .152572 -.342009
vertex -.0248397 .154896 -.342783
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0231077 .153347 -.343558
vertex -.0248397 .154896 -.342783
vertex -.0231077 .154121 -.341234
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0231077 .153347 -.343558
vertex -.0231077 .154121 -.341234
vertex -.0248397 .152572 -.342009
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0248397 .154896 -.342783
vertex -.0248397 .152572 -.342009
vertex -.0231077 .154121 -.341234
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -7.98964e-5 .152527 -.341732
vertex -.00181195 .151752 -.340182
vertex -.00181195 .154076 -.340957
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -7.98964e-5 .152527 -.341732
vertex -.00181195 .154076 -.340957
vertex -7.98964e-5 .153301 -.339408
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -7.98964e-5 .152527 -.341732
vertex -7.98964e-5 .153301 -.339408
vertex -.00181195 .151752 -.340182
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00181195 .154076 -.340957
vertex -.00181195 .151752 -.340182
vertex -7.98964e-5 .153301 -.339408
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0231731 .154986 -.34721
vertex .0214411 .154212 -.345661
vertex .0214411 .156536 -.346435
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0231731 .154986 -.34721
vertex .0214411 .156536 -.346435
vertex .0231731 .155761 -.344886
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0231731 .154986 -.34721
vertex .0231731 .155761 -.344886
vertex .0214411 .154212 -.345661
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0214411 .156536 -.346435
vertex .0214411 .154212 -.345661
vertex .0231731 .155761 -.344886
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.124643 .141904 -.324682
vertex -.126375 .14113 -.323133
vertex -.126375 .143454 -.323907
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.124643 .141904 -.324682
vertex -.126375 .143454 -.323907
vertex -.124643 .142679 -.322358
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.124643 .141904 -.324682
vertex -.124643 .142679 -.322358
vertex -.126375 .14113 -.323133
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.126375 .143454 -.323907
vertex -.126375 .14113 -.323133
vertex -.124643 .142679 -.322358
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.10356 .142711 -.326515
vertex -.105292 .141936 -.324966
vertex -.105292 .14426 -.32574
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.10356 .142711 -.326515
vertex -.105292 .14426 -.32574
vertex -.10356 .143485 -.324191
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.10356 .142711 -.326515
vertex -.10356 .143485 -.324191
vertex -.105292 .141936 -.324966
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.105292 .14426 -.32574
vertex -.105292 .141936 -.324966
vertex -.10356 .143485 -.324191
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0819975 .143114 -.327431
vertex -.0837296 .142339 -.325882
vertex -.0837296 .144663 -.326657
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0819975 .143114 -.327431
vertex -.0837296 .144663 -.326657
vertex -.0819975 .143888 -.325108
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0819975 .143114 -.327431
vertex -.0819975 .143888 -.325108
vertex -.0837296 .142339 -.325882
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0837296 .144663 -.326657
vertex -.0837296 .142339 -.325882
vertex -.0819975 .143888 -.325108
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0601413 .143114 -.327431
vertex -.0618734 .142339 -.325882
vertex -.0618734 .144663 -.326657
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0601413 .143114 -.327431
vertex -.0618734 .144663 -.326657
vertex -.0601413 .143888 -.325108
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0601413 .143114 -.327431
vertex -.0601413 .143888 -.325108
vertex -.0618734 .142339 -.325882
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0618734 .144663 -.326657
vertex -.0618734 .142339 -.325882
vertex -.0601413 .143888 -.325108
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0385051 .14392 -.329264
vertex -.0402371 .143145 -.327715
vertex -.0402371 .145469 -.32849
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0385051 .14392 -.329264
vertex -.0402371 .145469 -.32849
vertex -.0385051 .144695 -.326941
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0385051 .14392 -.329264
vertex -.0385051 .144695 -.326941
vertex -.0402371 .143145 -.327715
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0402371 .145469 -.32849
vertex -.0402371 .143145 -.327715
vertex -.0385051 .144695 -.326941
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0171577 .14916 -.341179
vertex -.0188897 .148386 -.33963
vertex -.0188897 .150709 -.340404
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0171577 .14916 -.341179
vertex -.0188897 .150709 -.340404
vertex -.0171577 .149935 -.338855
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0171577 .14916 -.341179
vertex -.0171577 .149935 -.338855
vertex -.0188897 .148386 -.33963
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0188897 .150709 -.340404
vertex -.0188897 .148386 -.33963
vertex -.0171577 .149935 -.338855
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00564508 .149966 -.343012
vertex .00391303 .149192 -.341463
vertex .00391303 .151516 -.342237
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00564508 .149966 -.343012
vertex .00391303 .151516 -.342237
vertex .00564508 .150741 -.340688
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00564508 .149966 -.343012
vertex .00564508 .150741 -.340688
vertex .00391303 .149192 -.341463
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00391303 .151516 -.342237
vertex .00391303 .149192 -.341463
vertex .00564508 .150741 -.340688
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .033127 .158431 -.362258
vertex .0313949 .157657 -.360709
vertex .0313949 .15998 -.361483
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .033127 .158431 -.362258
vertex .0313949 .15998 -.361483
vertex .033127 .159206 -.359934
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .033127 .158431 -.362258
vertex .033127 .159206 -.359934
vertex .0313949 .157657 -.360709
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0313949 .15998 -.361483
vertex .0313949 .157657 -.360709
vertex .033127 .159206 -.359934
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.122282 .139878 -.326813
vertex -.124014 .139103 -.325264
vertex -.124014 .141427 -.326039
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.122282 .139878 -.326813
vertex -.124014 .141427 -.326039
vertex -.122282 .140652 -.32449
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.122282 .139878 -.326813
vertex -.122282 .140652 -.32449
vertex -.124014 .139103 -.325264
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.124014 .141427 -.326039
vertex -.124014 .139103 -.325264
vertex -.122282 .140652 -.32449
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.101122 .14067 -.328653
vertex -.102854 .139896 -.327104
vertex -.102854 .142219 -.327879
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.101122 .14067 -.328653
vertex -.102854 .142219 -.327879
vertex -.101122 .141445 -.326329
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.101122 .14067 -.328653
vertex -.101122 .141445 -.326329
vertex -.102854 .139896 -.327104
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.102854 .142219 -.327879
vertex -.102854 .139896 -.327104
vertex -.101122 .141445 -.326329
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0794906 .141066 -.329573
vertex -.0812227 .140292 -.328024
vertex -.0812227 .142616 -.328799
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0794906 .141066 -.329573
vertex -.0812227 .142616 -.328799
vertex -.0794906 .141841 -.327249
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0794906 .141066 -.329573
vertex -.0794906 .141841 -.327249
vertex -.0812227 .140292 -.328024
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0812227 .142616 -.328799
vertex -.0812227 .140292 -.328024
vertex -.0794906 .141841 -.327249
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0577367 .141463 -.330493
vertex -.0594688 .140688 -.328944
vertex -.0594688 .143012 -.329719
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0577367 .141463 -.330493
vertex -.0594688 .143012 -.329719
vertex -.0577367 .142237 -.328169
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0577367 .141463 -.330493
vertex -.0577367 .142237 -.328169
vertex -.0594688 .140688 -.328944
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0594688 .143012 -.329719
vertex -.0594688 .140688 -.328944
vertex -.0577367 .142237 -.328169
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.03586 .141859 -.331413
vertex -.0375921 .141084 -.329864
vertex -.0375921 .143408 -.330638
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.03586 .141859 -.331413
vertex -.0375921 .143408 -.330638
vertex -.03586 .142633 -.329089
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.03586 .141859 -.331413
vertex -.03586 .142633 -.329089
vertex -.0375921 .141084 -.329864
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0375921 .143408 -.330638
vertex -.0375921 .141084 -.329864
vertex -.03586 .142633 -.329089
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0143105 .146614 -.342452
vertex -.0160426 .145839 -.340903
vertex -.0160426 .148163 -.341677
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0143105 .146614 -.342452
vertex -.0160426 .148163 -.341677
vertex -.0143105 .147388 -.340128
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0143105 .146614 -.342452
vertex -.0143105 .147388 -.340128
vertex -.0160426 .145839 -.340903
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0160426 .148163 -.341677
vertex -.0160426 .145839 -.340903
vertex -.0143105 .147388 -.340128
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0172691 .148991 -.347972
vertex .015537 .148216 -.346422
vertex .015537 .15054 -.347197
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0172691 .148991 -.347972
vertex .015537 .15054 -.347197
vertex .0172691 .149766 -.345648
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0172691 .148991 -.347972
vertex .0172691 .149766 -.345648
vertex .015537 .148216 -.346422
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .015537 .15054 -.347197
vertex .015537 .148216 -.346422
vertex .0172691 .149766 -.345648
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0434567 .160482 -.374649
vertex .0417247 .159707 -.3731
vertex .0417247 .162031 -.373875
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0434567 .160482 -.374649
vertex .0417247 .162031 -.373875
vertex .0434567 .161256 -.372325
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0434567 .160482 -.374649
vertex .0434567 .161256 -.372325
vertex .0417247 .159707 -.3731
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0417247 .162031 -.373875
vertex .0417247 .159707 -.3731
vertex .0434567 .161256 -.372325
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.109318 .138227 -.329875
vertex -.11105 .137452 -.328326
vertex -.11105 .139776 -.3291
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.109318 .138227 -.329875
vertex -.11105 .139776 -.3291
vertex -.109318 .139001 -.327551
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.109318 .138227 -.329875
vertex -.109318 .139001 -.327551
vertex -.11105 .137452 -.328326
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.11105 .139776 -.3291
vertex -.11105 .137452 -.328326
vertex -.109318 .139001 -.327551
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0877097 .138616 -.330798
vertex -.0894418 .137841 -.329249
vertex -.0894418 .140165 -.330024
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0877097 .138616 -.330798
vertex -.0894418 .140165 -.330024
vertex -.0877097 .139391 -.328475
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0877097 .138616 -.330798
vertex -.0877097 .139391 -.328475
vertex -.0894418 .137841 -.329249
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0894418 .140165 -.330024
vertex -.0894418 .137841 -.329249
vertex -.0877097 .139391 -.328475
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0659788 .139005 -.331722
vertex -.0677109 .138231 -.330173
vertex -.0677109 .140554 -.330947
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0659788 .139005 -.331722
vertex -.0677109 .140554 -.330947
vertex -.0659788 .13978 -.329398
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0659788 .139005 -.331722
vertex -.0659788 .13978 -.329398
vertex -.0677109 .138231 -.330173
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0677109 .140554 -.330947
vertex -.0677109 .138231 -.330173
vertex -.0659788 .13978 -.329398
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0442505 .139784 -.333568
vertex -.0459825 .139009 -.332019
vertex -.0459825 .141333 -.332794
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0442505 .139784 -.333568
vertex -.0459825 .141333 -.332794
vertex -.0442505 .140559 -.331245
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0442505 .139784 -.333568
vertex -.0442505 .140559 -.331245
vertex -.0459825 .139009 -.332019
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0459825 .141333 -.332794
vertex -.0459825 .139009 -.332019
vertex -.0442505 .140559 -.331245
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0228519 .144067 -.343725
vertex -.024584 .143292 -.342176
vertex -.024584 .145616 -.342951
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0228519 .144067 -.343725
vertex -.024584 .145616 -.342951
vertex -.0228519 .144842 -.341402
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0228519 .144067 -.343725
vertex -.0228519 .144842 -.341402
vertex -.024584 .143292 -.342176
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.024584 .145616 -.342951
vertex -.024584 .143292 -.342176
vertex -.0228519 .144842 -.341402
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00852805 .145235 -.346495
vertex .006796 .144461 -.344946
vertex .006796 .146784 -.345721
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00852805 .145235 -.346495
vertex .006796 .146784 -.345721
vertex .00852805 .14601 -.344172
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00852805 .145235 -.346495
vertex .00852805 .14601 -.344172
vertex .006796 .144461 -.344946
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .006796 .146784 -.345721
vertex .006796 .144461 -.344946
vertex .00852805 .14601 -.344172
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0334545 .15458 -.368656
vertex .0317224 .153805 -.367107
vertex .0317224 .156129 -.367881
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0334545 .15458 -.368656
vertex .0317224 .156129 -.367881
vertex .0334545 .155354 -.366332
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0334545 .15458 -.368656
vertex .0334545 .155354 -.366332
vertex .0317224 .153805 -.367107
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0317224 .156129 -.367881
vertex .0317224 .153805 -.367107
vertex .0334545 .155354 -.366332
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.117514 .135783 -.331097
vertex -.119246 .135008 -.329548
vertex -.119246 .137332 -.330322
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.117514 .135783 -.331097
vertex -.119246 .137332 -.330322
vertex -.117514 .136558 -.328773
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.117514 .135783 -.331097
vertex -.117514 .136558 -.328773
vertex -.119246 .135008 -.329548
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.119246 .137332 -.330322
vertex -.119246 .135008 -.329548
vertex -.117514 .136558 -.328773
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0959288 .136165 -.332024
vertex -.0976609 .135391 -.330474
vertex -.0976609 .137715 -.331249
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0959288 .136165 -.332024
vertex -.0976609 .137715 -.331249
vertex -.0959288 .13694 -.3297
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0959288 .136165 -.332024
vertex -.0959288 .13694 -.3297
vertex -.0976609 .135391 -.330474
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0976609 .137715 -.331249
vertex -.0976609 .135391 -.330474
vertex -.0959288 .13694 -.3297
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0744307 .13693 -.333877
vertex -.0761627 .136156 -.332328
vertex -.0761627 .13848 -.333103
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0744307 .13693 -.333877
vertex -.0761627 .13848 -.333103
vertex -.0744307 .137705 -.331553
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0744307 .13693 -.333877
vertex -.0744307 .137705 -.331553
vertex -.0761627 .136156 -.332328
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0761627 .13848 -.333103
vertex -.0761627 .136156 -.332328
vertex -.0744307 .137705 -.331553
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0523903 .13693 -.333877
vertex -.0541224 .136156 -.332328
vertex -.0541224 .13848 -.333103
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0523903 .13693 -.333877
vertex -.0541224 .13848 -.333103
vertex -.0523903 .137705 -.331553
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0523903 .13693 -.333877
vertex -.0523903 .137705 -.331553
vertex -.0541224 .136156 -.332328
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0541224 .13848 -.333103
vertex -.0541224 .136156 -.332328
vertex -.0523903 .137705 -.331553
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0286255 .141903 -.345925
vertex -.0303575 .141128 -.344376
vertex -.0303575 .143452 -.345151
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0286255 .141903 -.345925
vertex -.0303575 .143452 -.345151
vertex -.0286255 .142678 -.343602
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0286255 .141903 -.345925
vertex -.0286255 .142678 -.343602
vertex -.0303575 .141128 -.344376
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0303575 .143452 -.345151
vertex -.0303575 .141128 -.344376
vertex -.0286255 .142678 -.343602
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00309554 .148023 -.360754
vertex -.00482759 .147248 -.359205
vertex -.00482759 .149572 -.359979
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00309554 .148023 -.360754
vertex -.00482759 .149572 -.359979
vertex -.00309554 .148798 -.35843
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00309554 .148023 -.360754
vertex -.00309554 .148798 -.35843
vertex -.00482759 .147248 -.359205
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00482759 .149572 -.359979
vertex -.00482759 .147248 -.359205
vertex -.00309554 .148798 -.35843
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0238794 .14917 -.363534
vertex .0221474 .148396 -.361985
vertex .0221474 .15072 -.36276
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0238794 .14917 -.363534
vertex .0221474 .15072 -.36276
vertex .0238794 .149945 -.361211
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0238794 .14917 -.363534
vertex .0238794 .149945 -.361211
vertex .0221474 .148396 -.361985
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0221474 .15072 -.36276
vertex .0221474 .148396 -.361985
vertex .0238794 .149945 -.361211
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.126421 .134091 -.334179
vertex -.128153 .133316 -.33263
vertex -.128153 .13564 -.333405
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.126421 .134091 -.334179
vertex -.128153 .13564 -.333405
vertex -.126421 .134865 -.331855
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.126421 .134091 -.334179
vertex -.126421 .134865 -.331855
vertex -.128153 .133316 -.33263
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.128153 .13564 -.333405
vertex -.128153 .133316 -.33263
vertex -.126421 .134865 -.331855
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.104442 .134091 -.334179
vertex -.106174 .133316 -.33263
vertex -.106174 .13564 -.333405
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.104442 .134091 -.334179
vertex -.106174 .13564 -.333405
vertex -.104442 .134865 -.331855
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.104442 .134091 -.334179
vertex -.104442 .134865 -.331855
vertex -.106174 .133316 -.33263
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.106174 .13564 -.333405
vertex -.106174 .133316 -.33263
vertex -.104442 .134865 -.331855
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0826958 .134466 -.335109
vertex -.0844279 .133692 -.33356
vertex -.0844279 .136015 -.334335
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0826958 .134466 -.335109
vertex -.0844279 .136015 -.334335
vertex -.0826958 .135241 -.332786
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0826958 .134466 -.335109
vertex -.0826958 .135241 -.332786
vertex -.0844279 .133692 -.33356
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0844279 .136015 -.334335
vertex -.0844279 .133692 -.33356
vertex -.0826958 .135241 -.332786
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0608268 .134842 -.33604
vertex -.0625589 .134067 -.33449
vertex -.0625589 .136391 -.335265
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0608268 .134842 -.33604
vertex -.0625589 .136391 -.335265
vertex -.0608268 .135617 -.333716
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0608268 .134842 -.33604
vertex -.0608268 .135617 -.333716
vertex -.0625589 .134067 -.33449
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0625589 .136391 -.335265
vertex -.0625589 .134067 -.33449
vertex -.0608268 .135617 -.333716
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.038835 .135218 -.33697
vertex -.0405671 .134443 -.335421
vertex -.0405671 .136767 -.336195
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.038835 .135218 -.33697
vertex -.0405671 .136767 -.336195
vertex -.038835 .135992 -.334646
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.038835 .135218 -.33697
vertex -.038835 .135992 -.334646
vertex -.0405671 .134443 -.335421
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0405671 .136767 -.336195
vertex -.0405671 .134443 -.335421
vertex -.038835 .135992 -.334646
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0117625 .142355 -.354644
vertex -.0134945 .14158 -.353095
vertex -.0134945 .143904 -.353869
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0117625 .142355 -.354644
vertex -.0134945 .143904 -.353869
vertex -.0117625 .143129 -.35232
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0117625 .142355 -.354644
vertex -.0117625 .143129 -.35232
vertex -.0134945 .14158 -.353095
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0134945 .143904 -.353869
vertex -.0134945 .14158 -.353095
vertex -.0117625 .143129 -.35232
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0143042 .140476 -.349993
vertex .0125722 .139702 -.348444
vertex .0125722 .142026 -.349218
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0143042 .140476 -.349993
vertex .0125722 .142026 -.349218
vertex .0143042 .141251 -.347669
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0143042 .140476 -.349993
vertex .0143042 .141251 -.347669
vertex .0125722 .139702 -.348444
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0125722 .142026 -.349218
vertex .0125722 .139702 -.348444
vertex .0143042 .141251 -.347669
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0406327 .153248 -.38162
vertex .0389006 .152474 -.380071
vertex .0389006 .154797 -.380846
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0406327 .153248 -.38162
vertex .0389006 .154797 -.380846
vertex .0406327 .154023 -.379296
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0406327 .153248 -.38162
vertex .0406327 .154023 -.379296
vertex .0389006 .152474 -.380071
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0389006 .154797 -.380846
vertex .0389006 .152474 -.380071
vertex .0406327 .154023 -.379296
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.118179 .131633 -.335408
vertex -.119911 .130859 -.333859
vertex -.119911 .133182 -.334633
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.118179 .131633 -.335408
vertex -.119911 .133182 -.334633
vertex -.118179 .132408 -.333084
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.118179 .131633 -.335408
vertex -.118179 .132408 -.333084
vertex -.119911 .130859 -.333859
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.119911 .133182 -.334633
vertex -.119911 .130859 -.333859
vertex -.118179 .132408 -.333084
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0964711 .132002 -.336341
vertex -.0982031 .131227 -.334792
vertex -.0982031 .133551 -.335567
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0964711 .132002 -.336341
vertex -.0982031 .133551 -.335567
vertex -.0964711 .132777 -.334018
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0964711 .132002 -.336341
vertex -.0964711 .132777 -.334018
vertex -.0982031 .131227 -.334792
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0982031 .133551 -.335567
vertex -.0982031 .131227 -.334792
vertex -.0964711 .132777 -.334018
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0746404 .132371 -.337275
vertex -.0763725 .131596 -.335726
vertex -.0763725 .13392 -.3365
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0746404 .132371 -.337275
vertex -.0763725 .13392 -.3365
vertex -.0746404 .133145 -.334951
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0746404 .132371 -.337275
vertex -.0746404 .133145 -.334951
vertex -.0763725 .131596 -.335726
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0763725 .13392 -.3365
vertex -.0763725 .131596 -.335726
vertex -.0746404 .133145 -.334951
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.052687 .13274 -.338209
vertex -.0544191 .131965 -.33666
vertex -.0544191 .134289 -.337434
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.052687 .13274 -.338209
vertex -.0544191 .134289 -.337434
vertex -.052687 .133514 -.335885
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.052687 .13274 -.338209
vertex -.052687 .133514 -.335885
vertex -.0544191 .131965 -.33666
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0544191 .134289 -.337434
vertex -.0544191 .131965 -.33666
vertex -.052687 .133514 -.335885
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0283084 .135321 -.344744
vertex -.0300404 .134546 -.343195
vertex -.0300404 .13687 -.34397
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0283084 .135321 -.344744
vertex -.0300404 .13687 -.34397
vertex -.0283084 .136096 -.34242
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0283084 .135321 -.344744
vertex -.0283084 .136096 -.34242
vertex -.0300404 .134546 -.343195
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0300404 .13687 -.34397
vertex -.0300404 .134546 -.343195
vertex -.0283084 .136096 -.34242
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00609102 .143065 -.364351
vertex -.00782307 .142291 -.362802
vertex -.00782307 .144614 -.363576
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00609102 .143065 -.364351
vertex -.00782307 .144614 -.363576
vertex -.00609102 .14384 -.362027
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00609102 .143065 -.364351
vertex -.00609102 .14384 -.362027
vertex -.00782307 .142291 -.362802
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00782307 .144614 -.363576
vertex -.00782307 .142291 -.362802
vertex -.00609102 .14384 -.362027
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0201111 .138271 -.352213
vertex .018379 .137497 -.350664
vertex .018379 .13982 -.351439
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0201111 .138271 -.352213
vertex .018379 .13982 -.351439
vertex .0201111 .139046 -.34989
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0201111 .138271 -.352213
vertex .0201111 .139046 -.34989
vertex .018379 .137497 -.350664
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .018379 .13982 -.351439
vertex .018379 .137497 -.350664
vertex .0201111 .139046 -.34989
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.127488 .130262 -.339448
vertex -.12922 .129487 -.337899
vertex -.12922 .131811 -.338673
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.127488 .130262 -.339448
vertex -.12922 .131811 -.338673
vertex -.127488 .131036 -.337124
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.127488 .130262 -.339448
vertex -.127488 .131036 -.337124
vertex -.12922 .129487 -.337899
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.12922 .131811 -.338673
vertex -.12922 .129487 -.337899
vertex -.127488 .131036 -.337124
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.104736 .129538 -.337574
vertex -.106468 .128763 -.336024
vertex -.106468 .131087 -.336799
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.104736 .129538 -.337574
vertex -.106468 .131087 -.336799
vertex -.104736 .130312 -.33525
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.104736 .129538 -.337574
vertex -.104736 .130312 -.33525
vertex -.106468 .128763 -.336024
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.106468 .131087 -.336799
vertex -.106468 .128763 -.336024
vertex -.104736 .130312 -.33525
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0831614 .130262 -.339448
vertex -.0848934 .129487 -.337899
vertex -.0848934 .131811 -.338673
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0831614 .130262 -.339448
vertex -.0848934 .131811 -.338673
vertex -.0831614 .131036 -.337124
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0831614 .130262 -.339448
vertex -.0831614 .131036 -.337124
vertex -.0848934 .129487 -.337899
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0848934 .131811 -.338673
vertex -.0848934 .129487 -.337899
vertex -.0831614 .131036 -.337124
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0611696 .130624 -.340385
vertex -.0629016 .129849 -.338836
vertex -.0629016 .132173 -.33961
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0611696 .130624 -.340385
vertex -.0629016 .132173 -.33961
vertex -.0611696 .131398 -.338061
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0611696 .130624 -.340385
vertex -.0611696 .131398 -.338061
vertex -.0629016 .129849 -.338836
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0629016 .132173 -.33961
vertex -.0629016 .129849 -.338836
vertex -.0611696 .131398 -.338061
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0369853 .133519 -.347881
vertex -.0387174 .132744 -.346332
vertex -.0387174 .135068 -.347107
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0369853 .133519 -.347881
vertex -.0387174 .135068 -.347107
vertex -.0369853 .134293 -.345558
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0369853 .133519 -.347881
vertex -.0369853 .134293 -.345558
vertex -.0387174 .132744 -.346332
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0387174 .135068 -.347107
vertex -.0387174 .132744 -.346332
vertex -.0369853 .134293 -.345558
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0147196 .1375 -.358189
vertex -.0164516 .136725 -.35664
vertex -.0164516 .139049 -.357415
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0147196 .1375 -.358189
vertex -.0164516 .139049 -.357415
vertex -.0147196 .138274 -.355865
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0147196 .1375 -.358189
vertex -.0147196 .138274 -.355865
vertex -.0164516 .136725 -.35664
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0164516 .139049 -.357415
vertex -.0164516 .136725 -.35664
vertex -.0147196 .138274 -.355865
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00899925 .14329 -.373183
vertex .0072672 .142516 -.371633
vertex .0072672 .14484 -.372408
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00899925 .14329 -.373183
vertex .0072672 .14484 -.372408
vertex .00899925 .144065 -.370859
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00899925 .14329 -.373183
vertex .00899925 .144065 -.370859
vertex .0072672 .142516 -.371633
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0072672 .14484 -.372408
vertex .0072672 .142516 -.371633
vertex .00899925 .144065 -.370859
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0507705 .149805 -.39005
vertex .0490384 .14903 -.388501
vertex .0490384 .151354 -.389275
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0507705 .149805 -.39005
vertex .0490384 .151354 -.389275
vertex .0507705 .150579 -.387726
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0507705 .149805 -.39005
vertex .0507705 .150579 -.387726
vertex .0490384 .14903 -.388501
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0490384 .151354 -.389275
vertex .0490384 .14903 -.388501
vertex .0507705 .150579 -.387726
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.110556 .127429 -.339746
vertex -.112288 .126654 -.338197
vertex -.112288 .128978 -.338972
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.110556 .127429 -.339746
vertex -.112288 .128978 -.338972
vertex -.110556 .128203 -.337422
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.110556 .127429 -.339746
vertex -.110556 .128203 -.337422
vertex -.112288 .126654 -.338197
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.112288 .128978 -.338972
vertex -.112288 .126654 -.338197
vertex -.110556 .128203 -.337422
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0889503 .128139 -.341627
vertex -.0906823 .127364 -.340078
vertex -.0906823 .129688 -.340853
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0889503 .128139 -.341627
vertex -.0906823 .129688 -.340853
vertex -.0889503 .128913 -.339303
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0889503 .128139 -.341627
vertex -.0889503 .128913 -.339303
vertex -.0906823 .127364 -.340078
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0906823 .129688 -.340853
vertex -.0906823 .127364 -.340078
vertex -.0889503 .128913 -.339303
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0669124 .128494 -.342568
vertex -.0686445 .127719 -.341018
vertex -.0686445 .130043 -.341793
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0669124 .128494 -.342568
vertex -.0686445 .130043 -.341793
vertex -.0669124 .129268 -.340244
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0669124 .128494 -.342568
vertex -.0669124 .129268 -.340244
vertex -.0686445 .127719 -.341018
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0686445 .130043 -.341793
vertex -.0686445 .127719 -.341018
vertex -.0669124 .129268 -.340244
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0425466 .130624 -.348211
vertex -.0442787 .12985 -.346662
vertex -.0442787 .132173 -.347436
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0425466 .130624 -.348211
vertex -.0442787 .132173 -.347436
vertex -.0425466 .131399 -.345887
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0425466 .130624 -.348211
vertex -.0425466 .131399 -.345887
vertex -.0442787 .12985 -.346662
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0442787 .132173 -.347436
vertex -.0442787 .12985 -.346662
vertex -.0425466 .131399 -.345887
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0203423 .133465 -.355735
vertex -.0220744 .13269 -.354186
vertex -.0220744 .135014 -.35496
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0203423 .133465 -.355735
vertex -.0220744 .135014 -.35496
vertex -.0203423 .134239 -.353411
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0203423 .133465 -.355735
vertex -.0203423 .134239 -.353411
vertex -.0220744 .13269 -.354186
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0220744 .135014 -.35496
vertex -.0220744 .13269 -.354186
vertex -.0203423 .134239 -.353411
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00291105 .14092 -.375486
vertex .001179 .140146 -.373936
vertex .001179 .14247 -.374711
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00291105 .14092 -.375486
vertex .001179 .14247 -.374711
vertex .00291105 .141695 -.373162
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00291105 .14092 -.375486
vertex .00291105 .141695 -.373162
vertex .001179 .140146 -.373936
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .001179 .14247 -.374711
vertex .001179 .140146 -.373936
vertex .00291105 .141695 -.373162
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0260151 .13382 -.356675
vertex .0242831 .133045 -.355126
vertex .0242831 .135369 -.355901
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0260151 .13382 -.356675
vertex .0242831 .135369 -.355901
vertex .0260151 .134594 -.354352
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0260151 .13382 -.356675
vertex .0260151 .134594 -.354352
vertex .0242831 .133045 -.355126
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0242831 .135369 -.355901
vertex .0242831 .133045 -.355126
vertex .0260151 .134594 -.354352
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.121504 .127743 -.348533
vertex -.123236 .126969 -.346984
vertex -.123236 .129292 -.347759
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.121504 .127743 -.348533
vertex -.123236 .129292 -.347759
vertex -.121504 .128518 -.346209
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.121504 .127743 -.348533
vertex -.121504 .128518 -.346209
vertex -.123236 .126969 -.346984
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.123236 .129292 -.347759
vertex -.123236 .126969 -.346984
vertex -.121504 .128518 -.346209
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0975556 .126002 -.343813
vertex -.0992877 .125228 -.342264
vertex -.0992877 .127551 -.343039
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0975556 .126002 -.343813
vertex -.0992877 .127551 -.343039
vertex -.0975556 .126777 -.34149
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0975556 .126002 -.343813
vertex -.0975556 .126777 -.34149
vertex -.0992877 .125228 -.342264
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0992877 .127551 -.343039
vertex -.0992877 .125228 -.342264
vertex -.0975556 .126777 -.34149
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0754794 .12635 -.344757
vertex -.0772114 .125576 -.343208
vertex -.0772114 .1279 -.343983
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0754794 .12635 -.344757
vertex -.0772114 .1279 -.343983
vertex -.0754794 .127125 -.342434
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0754794 .12635 -.344757
vertex -.0754794 .127125 -.342434
vertex -.0772114 .125576 -.343208
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0772114 .1279 -.343983
vertex -.0772114 .125576 -.343208
vertex -.0754794 .127125 -.342434
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.053132 .12635 -.344757
vertex -.0548641 .125576 -.343208
vertex -.0548641 .1279 -.343983
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.053132 .12635 -.344757
vertex -.0548641 .1279 -.343983
vertex -.053132 .127125 -.342434
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.053132 .12635 -.344757
vertex -.053132 .127125 -.342434
vertex -.0548641 .125576 -.343208
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0548641 .1279 -.343983
vertex -.0548641 .125576 -.343208
vertex -.053132 .127125 -.342434
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0284669 .128439 -.350421
vertex -.030199 .127665 -.348872
vertex -.030199 .129989 -.349646
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0284669 .128439 -.350421
vertex -.030199 .129989 -.349646
vertex -.0284669 .129214 -.348097
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0284669 .128439 -.350421
vertex -.0284669 .129214 -.348097
vertex -.030199 .127665 -.348872
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.030199 .129989 -.349646
vertex -.030199 .127665 -.348872
vertex -.0284669 .129214 -.348097
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00621621 .137492 -.374964
vertex -.00794826 .136718 -.373414
vertex -.00794826 .139041 -.374189
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00621621 .137492 -.374964
vertex -.00794826 .139041 -.374189
vertex -.00621621 .138267 -.37264
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00621621 .137492 -.374964
vertex -.00621621 .138267 -.37264
vertex -.00794826 .136718 -.373414
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00794826 .139041 -.374189
vertex -.00794826 .136718 -.373414
vertex -.00621621 .138267 -.37264
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0178347 .135403 -.3693
vertex .0161026 .134629 -.367751
vertex .0161026 .136952 -.368525
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0178347 .135403 -.3693
vertex .0161026 .136952 -.368525
vertex .0178347 .136178 -.366976
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0178347 .135403 -.3693
vertex .0178347 .136178 -.366976
vertex .0161026 .134629 -.367751
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0161026 .136952 -.368525
vertex .0161026 .134629 -.367751
vertex .0178347 .136178 -.366976
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.130803 .12078 -.33748
vertex -.132535 .120005 -.335931
vertex -.132535 .122329 -.336706
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.130803 .12078 -.33748
vertex -.132535 .122329 -.336706
vertex -.130803 .121555 -.335157
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.130803 .12078 -.33748
vertex -.130803 .121555 -.335157
vertex -.132535 .120005 -.335931
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.132535 .122329 -.336706
vertex -.132535 .120005 -.335931
vertex -.130803 .121555 -.335157
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.111175 .123169 -.344112
vertex -.112907 .122395 -.342563
vertex -.112907 .124718 -.343337
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.111175 .123169 -.344112
vertex -.112907 .124718 -.343337
vertex -.111175 .123944 -.341788
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.111175 .123169 -.344112
vertex -.111175 .123944 -.341788
vertex -.112907 .122395 -.342563
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.112907 .124718 -.343337
vertex -.112907 .122395 -.342563
vertex -.111175 .123944 -.341788
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0894465 .123852 -.346007
vertex -.0911785 .123077 -.344457
vertex -.0911785 .125401 -.345232
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0894465 .123852 -.346007
vertex -.0911785 .125401 -.345232
vertex -.0894465 .124627 -.343683
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0894465 .123852 -.346007
vertex -.0894465 .124627 -.343683
vertex -.0911785 .123077 -.344457
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0911785 .125401 -.345232
vertex -.0911785 .123077 -.344457
vertex -.0894465 .124627 -.343683
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0670991 .123852 -.346007
vertex -.0688312 .123077 -.344457
vertex -.0688312 .125401 -.345232
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0670991 .123852 -.346007
vertex -.0688312 .125401 -.345232
vertex -.0670991 .124627 -.343683
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0670991 .123852 -.346007
vertex -.0670991 .124627 -.343683
vertex -.0688312 .123077 -.344457
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0688312 .125401 -.345232
vertex -.0688312 .123077 -.344457
vertex -.0670991 .124627 -.343683
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.042429 .125217 -.349796
vertex -.044161 .124443 -.348247
vertex -.044161 .126766 -.349022
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.042429 .125217 -.349796
vertex -.044161 .126766 -.349022
vertex -.042429 .125992 -.347472
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.042429 .125217 -.349796
vertex -.042429 .125992 -.347472
vertex -.044161 .124443 -.348247
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.044161 .126766 -.349022
vertex -.044161 .124443 -.348247
vertex -.042429 .125992 -.347472
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0205674 .129654 -.362112
vertex -.0222994 .12888 -.360563
vertex -.0222994 .131203 -.361337
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0205674 .129654 -.362112
vertex -.0222994 .131203 -.361337
vertex -.0205674 .130429 -.359788
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0205674 .129654 -.362112
vertex -.0205674 .130429 -.359788
vertex -.0222994 .12888 -.360563
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0222994 .131203 -.361337
vertex -.0222994 .12888 -.360563
vertex -.0205674 .130429 -.359788
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00289563 .134433 -.375375
vertex .00116358 .133658 -.373826
vertex .00116358 .135982 -.374601
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00289563 .134433 -.375375
vertex .00116358 .135982 -.374601
vertex .00289563 .135207 -.373052
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00289563 .134433 -.375375
vertex .00289563 .135207 -.373052
vertex .00116358 .133658 -.373826
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00116358 .135982 -.374601
vertex .00116358 .133658 -.373826
vertex .00289563 .135207 -.373052
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0264143 .130678 -.364954
vertex .0246823 .129904 -.363405
vertex .0246823 .132227 -.36418
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0264143 .130678 -.364954
vertex .0246823 .132227 -.36418
vertex .0264143 .131453 -.36263
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0264143 .130678 -.364954
vertex .0264143 .131453 -.36263
vertex .0246823 .129904 -.363405
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0246823 .132227 -.36418
vertex .0246823 .129904 -.363405
vertex .0264143 .131453 -.36263
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.128079 .118343 -.338699
vertex -.129811 .117569 -.337149
vertex -.129811 .119893 -.337924
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.128079 .118343 -.338699
vertex -.129811 .119893 -.337924
vertex -.128079 .119118 -.336375
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.128079 .118343 -.338699
vertex -.128079 .119118 -.336375
vertex -.129811 .117569 -.337149
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.129811 .119893 -.337924
vertex -.129811 .117569 -.337149
vertex -.128079 .119118 -.336375
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.109 .121353 -.347256
vertex -.110732 .120579 -.345707
vertex -.110732 .122903 -.346481
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.109 .121353 -.347256
vertex -.110732 .122903 -.346481
vertex -.109 .122128 -.344932
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.109 .121353 -.347256
vertex -.109 .122128 -.344932
vertex -.110732 .120579 -.345707
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.110732 .122903 -.346481
vertex -.110732 .120579 -.345707
vertex -.109 .122128 -.344932
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0866531 .121353 -.347256
vertex -.0883851 .120579 -.345707
vertex -.0883851 .122903 -.346481
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0866531 .121353 -.347256
vertex -.0883851 .122903 -.346481
vertex -.0866531 .122128 -.344932
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0866531 .121353 -.347256
vertex -.0866531 .122128 -.344932
vertex -.0883851 .120579 -.345707
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0883851 .122903 -.346481
vertex -.0883851 .120579 -.345707
vertex -.0866531 .122128 -.344932
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.061855 .122022 -.349158
vertex -.0635871 .121248 -.347608
vertex -.0635871 .123571 -.348383
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.061855 .122022 -.349158
vertex -.0635871 .123571 -.348383
vertex -.061855 .122797 -.346834
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.061855 .122022 -.349158
vertex -.061855 .122797 -.346834
vertex -.0635871 .121248 -.347608
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0635871 .123571 -.348383
vertex -.0635871 .121248 -.347608
vertex -.061855 .122797 -.346834
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0393849 .122022 -.349158
vertex -.0411169 .121248 -.347608
vertex -.0411169 .123571 -.348383
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0393849 .122022 -.349158
vertex -.0411169 .123571 -.348383
vertex -.0393849 .122797 -.346834
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0393849 .122022 -.349158
vertex -.0393849 .122797 -.346834
vertex -.0411169 .121248 -.347608
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0411169 .123571 -.348383
vertex -.0411169 .121248 -.347608
vertex -.0393849 .122797 -.346834
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0178864 .128711 -.368174
vertex -.0196184 .127937 -.366625
vertex -.0196184 .130261 -.367399
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0178864 .128711 -.368174
vertex -.0196184 .130261 -.367399
vertex -.0178864 .129486 -.36585
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0178864 .128711 -.368174
vertex -.0178864 .129486 -.36585
vertex -.0196184 .127937 -.366625
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0196184 .130261 -.367399
vertex -.0196184 .127937 -.366625
vertex -.0178864 .129486 -.36585
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00588851 .130718 -.373879
vertex .00415646 .129943 -.372329
vertex .00415646 .132267 -.373104
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00588851 .130718 -.373879
vertex .00415646 .132267 -.373104
vertex .00588851 .131493 -.371555
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00588851 .130718 -.373879
vertex .00588851 .131493 -.371555
vertex .00415646 .129943 -.372329
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00415646 .132267 -.373104
vertex .00415646 .129943 -.372329
vertex .00588851 .131493 -.371555
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0291386 .127039 -.36342
vertex .0274066 .126264 -.36187
vertex .0274066 .128588 -.362645
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0291386 .127039 -.36342
vertex .0274066 .128588 -.362645
vertex .0291386 .127814 -.361096
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0291386 .127039 -.36342
vertex .0291386 .127814 -.361096
vertex .0274066 .126264 -.36187
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0274066 .128588 -.362645
vertex .0274066 .126264 -.36187
vertex .0291386 .127814 -.361096
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.130432 .115579 -.338963
vertex -.132164 .114804 -.337414
vertex -.132164 .117128 -.338188
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.130432 .115579 -.338963
vertex -.132164 .117128 -.338188
vertex -.130432 .116354 -.336639
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.130432 .115579 -.338963
vertex -.130432 .116354 -.336639
vertex -.132164 .114804 -.337414
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.132164 .117128 -.338188
vertex -.132164 .114804 -.337414
vertex -.130432 .116354 -.336639
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.109302 .119182 -.349459
vertex -.111034 .118408 -.34791
vertex -.111034 .120732 -.348685
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.109302 .119182 -.349459
vertex -.111034 .120732 -.348685
vertex -.109302 .119957 -.347136
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.109302 .119182 -.349459
vertex -.109302 .119957 -.347136
vertex -.111034 .118408 -.34791
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.111034 .120732 -.348685
vertex -.111034 .118408 -.34791
vertex -.109302 .119957 -.347136
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0871339 .11951 -.350414
vertex -.088866 .118735 -.348864
vertex -.088866 .121059 -.349639
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0871339 .11951 -.350414
vertex -.088866 .121059 -.349639
vertex -.0871339 .120285 -.34809
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0871339 .11951 -.350414
vertex -.0871339 .120285 -.34809
vertex -.088866 .118735 -.348864
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.088866 .121059 -.349639
vertex -.088866 .118735 -.348864
vertex -.0871339 .120285 -.34809
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0646638 .11951 -.350414
vertex -.0663959 .118735 -.348864
vertex -.0663959 .121059 -.349639
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0646638 .11951 -.350414
vertex -.0663959 .121059 -.349639
vertex -.0646638 .120285 -.34809
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0646638 .11951 -.350414
vertex -.0646638 .120285 -.34809
vertex -.0663959 .118735 -.348864
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0663959 .121059 -.349639
vertex -.0663959 .118735 -.348864
vertex -.0646638 .120285 -.34809
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.042076 .119182 -.349459
vertex -.0438081 .118408 -.34791
vertex -.0438081 .120732 -.348685
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.042076 .119182 -.349459
vertex -.0438081 .120732 -.348685
vertex -.042076 .119957 -.347136
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.042076 .119182 -.349459
vertex -.042076 .119957 -.347136
vertex -.0438081 .118408 -.34791
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0438081 .120732 -.348685
vertex -.0438081 .118408 -.34791
vertex -.042076 .119957 -.347136
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0207924 .125734 -.368544
vertex -.0225244 .12496 -.366995
vertex -.0225244 .127283 -.36777
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0207924 .125734 -.368544
vertex -.0225244 .127283 -.36777
vertex -.0207924 .126509 -.36622
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0207924 .125734 -.368544
vertex -.0207924 .126509 -.36622
vertex -.0225244 .12496 -.366995
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0225244 .127283 -.36777
vertex -.0225244 .12496 -.366995
vertex -.0207924 .126509 -.36622
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00288022 .128027 -.375224
vertex .00114817 .127253 -.373675
vertex .00114817 .129577 -.374449
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00288022 .128027 -.375224
vertex .00114817 .129577 -.374449
vertex .00288022 .128802 -.3729
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00288022 .128027 -.375224
vertex .00288022 .128802 -.3729
vertex .00114817 .127253 -.373675
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00114817 .129577 -.374449
vertex .00114817 .127253 -.373675
vertex .00288022 .128802 -.3729
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0262147 .124424 -.364727
vertex .0244827 .123649 -.363178
vertex .0244827 .125973 -.363953
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0262147 .124424 -.364727
vertex .0244827 .125973 -.363953
vertex .0262147 .125198 -.362403
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0262147 .124424 -.364727
vertex .0262147 .125198 -.362403
vertex .0244827 .123649 -.363178
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0244827 .125973 -.363953
vertex .0244827 .123649 -.363178
vertex .0262147 .125198 -.362403
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.137797 .114753 -.344966
vertex -.139529 .113978 -.343417
vertex -.139529 .116302 -.344192
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.137797 .114753 -.344966
vertex -.139529 .116302 -.344192
vertex -.137797 .115527 -.342642
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.137797 .114753 -.344966
vertex -.137797 .115527 -.342642
vertex -.139529 .113978 -.343417
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.139529 .116302 -.344192
vertex -.139529 .113978 -.343417
vertex -.137797 .115527 -.342642
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.115756 .114753 -.344966
vertex -.117488 .113978 -.343417
vertex -.117488 .116302 -.344192
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.115756 .114753 -.344966
vertex -.117488 .116302 -.344192
vertex -.115756 .115527 -.342642
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.115756 .114753 -.344966
vertex -.115756 .115527 -.342642
vertex -.117488 .113978 -.343417
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.117488 .116302 -.344192
vertex -.117488 .113978 -.343417
vertex -.115756 .115527 -.342642
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0955602 .116998 -.35167
vertex -.0972923 .116223 -.350121
vertex -.0972923 .118547 -.350895
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0955602 .116998 -.35167
vertex -.0972923 .118547 -.350895
vertex -.0955602 .117772 -.349346
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0955602 .116998 -.35167
vertex -.0955602 .117772 -.349346
vertex -.0972923 .116223 -.350121
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0972923 .118547 -.350895
vertex -.0972923 .116223 -.350121
vertex -.0955602 .117772 -.349346
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0732922 .117319 -.352627
vertex -.0750242 .116544 -.351078
vertex -.0750242 .118868 -.351853
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0732922 .117319 -.352627
vertex -.0750242 .118868 -.351853
vertex -.0732922 .118093 -.350304
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0732922 .117319 -.352627
vertex -.0732922 .118093 -.350304
vertex -.0750242 .116544 -.351078
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0750242 .118868 -.351853
vertex -.0750242 .116544 -.351078
vertex -.0732922 .118093 -.350304
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0414934 .108018 -.324855
vertex -.0432254 .107243 -.323306
vertex -.0432254 .109567 -.32408
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0414934 .108018 -.324855
vertex -.0432254 .109567 -.32408
vertex -.0414934 .108792 -.322531
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0414934 .108018 -.324855
vertex -.0414934 .108792 -.322531
vertex -.0432254 .107243 -.323306
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0432254 .109567 -.32408
vertex -.0432254 .107243 -.323306
vertex -.0414934 .108792 -.322531
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.023747 .123092 -.369866
vertex -.025479 .122317 -.368316
vertex -.025479 .124641 -.369091
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.023747 .123092 -.369866
vertex -.025479 .124641 -.369091
vertex -.023747 .123866 -.367542
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.023747 .123092 -.369866
vertex -.023747 .123866 -.367542
vertex -.025479 .122317 -.368316
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.025479 .124641 -.369091
vertex -.025479 .122317 -.368316
vertex -.023747 .123866 -.367542
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00012808 .125337 -.376569
vertex -.00186013 .124562 -.37502
vertex -.00186013 .126886 -.375795
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00012808 .125337 -.376569
vertex -.00186013 .126886 -.375795
vertex -.00012808 .126111 -.374245
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00012808 .125337 -.376569
vertex -.00012808 .126111 -.374245
vertex -.00186013 .124562 -.37502
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00186013 .126886 -.375795
vertex -.00186013 .124562 -.37502
vertex -.00012808 .126111 -.374245
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0261482 .121488 -.365077
vertex .0244162 .120713 -.363528
vertex .0244162 .123037 -.364303
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0261482 .121488 -.365077
vertex .0244162 .123037 -.364303
vertex .0261482 .122263 -.362753
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0261482 .121488 -.365077
vertex .0261482 .122263 -.362753
vertex .0244162 .120713 -.363528
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0244162 .123037 -.364303
vertex .0244162 .120713 -.363528
vertex .0261482 .122263 -.362753
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0718463 .130468 -.391892
vertex .0701143 .129694 -.390343
vertex .0701143 .132017 -.391117
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0718463 .130468 -.391892
vertex .0701143 .132017 -.391117
vertex .0718463 .131243 -.389568
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0718463 .130468 -.391892
vertex .0718463 .131243 -.389568
vertex .0701143 .129694 -.390343
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0701143 .132017 -.391117
vertex .0701143 .129694 -.390343
vertex .0718463 .131243 -.389568
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.117524 .108836 -.335626
vertex -.119256 .108062 -.334077
vertex -.119256 .110385 -.334851
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.117524 .108836 -.335626
vertex -.119256 .110385 -.334851
vertex -.117524 .109611 -.333302
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.117524 .108836 -.335626
vertex -.117524 .109611 -.333302
vertex -.119256 .108062 -.334077
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.119256 .110385 -.334851
vertex -.119256 .108062 -.334077
vertex -.117524 .109611 -.333302
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.101457 .114799 -.353887
vertex -.103189 .114025 -.352338
vertex -.103189 .116349 -.353112
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.101457 .114799 -.353887
vertex -.103189 .116349 -.353112
vertex -.101457 .115574 -.351563
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.101457 .114799 -.353887
vertex -.101457 .115574 -.351563
vertex -.103189 .114025 -.352338
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.103189 .116349 -.353112
vertex -.103189 .114025 -.352338
vertex -.101457 .115574 -.351563
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0791425 .115113 -.354848
vertex -.0808745 .114339 -.353299
vertex -.0808745 .116663 -.354074
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0791425 .115113 -.354848
vertex -.0808745 .116663 -.354074
vertex -.0791425 .115888 -.352524
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0791425 .115113 -.354848
vertex -.0791425 .115888 -.352524
vertex -.0808745 .114339 -.353299
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0808745 .116663 -.354074
vertex -.0808745 .114339 -.353299
vertex -.0791425 .115888 -.352524
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0489783 .10507 -.324093
vertex -.0507104 .104295 -.322544
vertex -.0507104 .106619 -.323318
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0489783 .10507 -.324093
vertex -.0507104 .106619 -.323318
vertex -.0489783 .105844 -.321769
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0489783 .10507 -.324093
vertex -.0489783 .105844 -.321769
vertex -.0507104 .104295 -.322544
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0507104 .106619 -.323318
vertex -.0507104 .104295 -.322544
vertex -.0489783 .105844 -.321769
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0292597 .11888 -.366381
vertex -.0309918 .118105 -.364832
vertex -.0309918 .120429 -.365607
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0292597 .11888 -.366381
vertex -.0309918 .120429 -.365607
vertex -.0292597 .119654 -.364058
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0292597 .11888 -.366381
vertex -.0292597 .119654 -.364058
vertex -.0309918 .118105 -.364832
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0309918 .120429 -.365607
vertex -.0309918 .118105 -.364832
vertex -.0292597 .119654 -.364058
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00312617 .122332 -.376954
vertex -.00485822 .121558 -.375404
vertex -.00485822 .123881 -.376179
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00312617 .122332 -.376954
vertex -.00485822 .123881 -.376179
vertex -.00312617 .123107 -.37463
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00312617 .122332 -.376954
vertex -.00312617 .123107 -.37463
vertex -.00485822 .121558 -.375404
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00485822 .123881 -.376179
vertex -.00485822 .121558 -.375404
vertex -.00312617 .123107 -.37463
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .020367 .119193 -.367342
vertex .0186349 .118419 -.365793
vertex .0186349 .120743 -.366568
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .020367 .119193 -.367342
vertex .0186349 .120743 -.366568
vertex .020367 .119968 -.365019
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .020367 .119193 -.367342
vertex .020367 .119968 -.365019
vertex .0186349 .118419 -.365793
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0186349 .120743 -.366568
vertex .0186349 .118419 -.365793
vertex .020367 .119968 -.365019
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0685489 .127354 -.392331
vertex .0668169 .126579 -.390782
vertex .0668169 .128903 -.391557
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0685489 .127354 -.392331
vertex .0668169 .128903 -.391557
vertex .0685489 .128128 -.390007
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0685489 .127354 -.392331
vertex .0685489 .128128 -.390007
vertex .0668169 .126579 -.390782
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0668169 .128903 -.391557
vertex .0668169 .126579 -.390782
vertex .0685489 .128128 -.390007
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.119151 .105526 -.333927
vertex -.120883 .104752 -.332378
vertex -.120883 .107076 -.333152
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.119151 .105526 -.333927
vertex -.120883 .107076 -.333152
vertex -.119151 .106301 -.331603
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.119151 .105526 -.333927
vertex -.119151 .106301 -.331603
vertex -.120883 .104752 -.332378
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.120883 .107076 -.333152
vertex -.120883 .104752 -.332378
vertex -.119151 .106301 -.331603
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.102841 .110745 -.350324
vertex -.104573 .109971 -.348775
vertex -.104573 .112295 -.349549
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.102841 .110745 -.350324
vertex -.104573 .112295 -.349549
vertex -.102841 .11152 -.348
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.102841 .110745 -.350324
vertex -.102841 .11152 -.348
vertex -.104573 .109971 -.348775
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.104573 .112295 -.349549
vertex -.104573 .109971 -.348775
vertex -.102841 .11152 -.348
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0819666 .112587 -.356111
vertex -.0836986 .111813 -.354562
vertex -.0836986 .114137 -.355337
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0819666 .112587 -.356111
vertex -.0836986 .114137 -.355337
vertex -.0819666 .113362 -.353787
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0819666 .112587 -.356111
vertex -.0819666 .113362 -.353787
vertex -.0836986 .111813 -.354562
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0836986 .114137 -.355337
vertex -.0836986 .111813 -.354562
vertex -.0819666 .113362 -.353787
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0489783 .102763 -.325246
vertex -.0507104 .101989 -.323697
vertex -.0507104 .104313 -.324471
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0489783 .102763 -.325246
vertex -.0507104 .104313 -.324471
vertex -.0489783 .103538 -.322922
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0489783 .102763 -.325246
vertex -.0489783 .103538 -.322922
vertex -.0507104 .101989 -.323697
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0507104 .104313 -.324471
vertex -.0507104 .101989 -.323697
vertex -.0489783 .103538 -.322922
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0212424 .120262 -.380224
vertex -.0229745 .119488 -.378675
vertex -.0229745 .121811 -.37945
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0212424 .120262 -.380224
vertex -.0229745 .121811 -.37945
vertex -.0212424 .121037 -.377901
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0212424 .120262 -.380224
vertex -.0212424 .121037 -.377901
vertex -.0229745 .119488 -.378675
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0229745 .121811 -.37945
vertex -.0229745 .119488 -.378675
vertex -.0212424 .121037 -.377901
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00284939 .118113 -.373473
vertex .00111734 .117339 -.371924
vertex .00111734 .119662 -.372698
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00284939 .118113 -.373473
vertex .00111734 .119662 -.372698
vertex .00284939 .118888 -.371149
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00284939 .118113 -.373473
vertex .00284939 .118888 -.371149
vertex .00111734 .117339 -.371924
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00111734 .119662 -.372698
vertex .00111734 .117339 -.371924
vertex .00284939 .118888 -.371149
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0260151 .115657 -.365756
vertex .0242831 .114883 -.364207
vertex .0242831 .117206 -.364982
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0260151 .115657 -.365756
vertex .0242831 .117206 -.364982
vertex .0260151 .116432 -.363433
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0260151 .115657 -.365756
vertex .0260151 .116432 -.363433
vertex .0242831 .114883 -.364207
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0242831 .117206 -.364982
vertex .0242831 .114883 -.364207
vertex .0260151 .116432 -.363433
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0742508 .123639 -.390834
vertex .0725188 .122865 -.389285
vertex .0725188 .125188 -.39006
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0742508 .123639 -.390834
vertex .0725188 .125188 -.39006
vertex .0742508 .124414 -.388511
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0742508 .123639 -.390834
vertex .0742508 .124414 -.388511
vertex .0725188 .122865 -.389285
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0725188 .125188 -.39006
vertex .0725188 .122865 -.389285
vertex .0742508 .124414 -.388511
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.115823 .102558 -.333175
vertex -.117555 .101783 -.331626
vertex -.117555 .104107 -.3324
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.115823 .102558 -.333175
vertex -.117555 .104107 -.3324
vertex -.115823 .103333 -.330851
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.115823 .102558 -.333175
vertex -.115823 .103333 -.330851
vertex -.117555 .101783 -.331626
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.117555 .104107 -.3324
vertex -.117555 .101783 -.331626
vertex -.115823 .103333 -.330851
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0995049 .10766 -.34963
vertex -.101237 .106886 -.348081
vertex -.101237 .10921 -.348856
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0995049 .10766 -.34963
vertex -.101237 .10921 -.348856
vertex -.0995049 .108435 -.347307
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0995049 .10766 -.34963
vertex -.0995049 .108435 -.347307
vertex -.101237 .106886 -.348081
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.101237 .10921 -.348856
vertex -.101237 .106886 -.348081
vertex -.0995049 .108435 -.347307
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0793599 .110362 -.358342
vertex -.0810919 .109587 -.356793
vertex -.0810919 .111911 -.357567
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0793599 .110362 -.358342
vertex -.0810919 .111911 -.357567
vertex -.0793599 .111136 -.356018
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0793599 .110362 -.358342
vertex -.0793599 .111136 -.356018
vertex -.0810919 .109587 -.356793
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0810919 .111911 -.357567
vertex -.0810919 .109587 -.356793
vertex -.0793599 .111136 -.356018
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0463998 .100457 -.326399
vertex -.0481318 .0996825 -.32485
vertex -.0481318 .102006 -.325625
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0463998 .100457 -.326399
vertex -.0481318 .102006 -.325625
vertex -.0463998 .101232 -.324075
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0463998 .100457 -.326399
vertex -.0463998 .101232 -.324075
vertex -.0481318 .0996825 -.32485
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0481318 .102006 -.325625
vertex -.0481318 .0996825 -.32485
vertex -.0463998 .101232 -.324075
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0281498 .109461 -.355438
vertex -.0298819 .108687 -.353889
vertex -.0298819 .11101 -.354664
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0281498 .109461 -.355438
vertex -.0298819 .11101 -.354664
vertex -.0281498 .110236 -.353114
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0281498 .109461 -.355438
vertex -.0281498 .110236 -.353114
vertex -.0298819 .108687 -.353889
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0298819 .11101 -.354664
vertex -.0298819 .108687 -.353889
vertex -.0281498 .110236 -.353114
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00609102 .116364 -.377701
vertex -.00782307 .11559 -.376152
vertex -.00782307 .117913 -.376927
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00609102 .116364 -.377701
vertex -.00782307 .117913 -.376927
vertex -.00609102 .117139 -.375378
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00609102 .116364 -.377701
vertex -.00609102 .117139 -.375378
vertex -.00782307 .11559 -.376152
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00782307 .117913 -.376927
vertex -.00782307 .11559 -.376152
vertex -.00609102 .117139 -.375378
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0173561 .113363 -.368022
vertex .015624 .112588 -.366473
vertex .015624 .114912 -.367247
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0173561 .113363 -.368022
vertex .015624 .114912 -.367247
vertex .0173561 .114137 -.365698
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0173561 .113363 -.368022
vertex .0173561 .114137 -.365698
vertex .015624 .112588 -.366473
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .015624 .114912 -.367247
vertex .015624 .112588 -.366473
vertex .0173561 .114137 -.365698
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0682163 .121166 -.393189
vertex .0664843 .120392 -.39164
vertex .0664843 .122715 -.392414
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0682163 .121166 -.393189
vertex .0664843 .122715 -.392414
vertex .0682163 .121941 -.390865
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0682163 .121166 -.393189
vertex .0682163 .121941 -.390865
vertex .0664843 .120392 -.39164
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0664843 .122715 -.392414
vertex .0664843 .120392 -.39164
vertex .0682163 .121941 -.390865
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.130621 .105776 -.352809
vertex -.132353 .105001 -.351259
vertex -.132353 .107325 -.352034
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.130621 .105776 -.352809
vertex -.132353 .107325 -.352034
vertex -.130621 .10655 -.350485
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.130621 .105776 -.352809
vertex -.130621 .10655 -.350485
vertex -.132353 .105001 -.351259
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.132353 .107325 -.352034
vertex -.132353 .105001 -.351259
vertex -.130621 .10655 -.350485
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.102677 .102843 -.343095
vertex -.104409 .102069 -.341545
vertex -.104409 .104392 -.34232
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.102677 .102843 -.343095
vertex -.104409 .104392 -.34232
vertex -.102677 .103618 -.340771
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.102677 .102843 -.343095
vertex -.102677 .103618 -.340771
vertex -.104409 .102069 -.341545
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.104409 .104392 -.34232
vertex -.104409 .102069 -.341545
vertex -.102677 .103618 -.340771
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0824168 .108122 -.36058
vertex -.0841488 .107347 -.359031
vertex -.0841488 .109671 -.359805
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0824168 .108122 -.36058
vertex -.0841488 .109671 -.359805
vertex -.0824168 .108897 -.358256
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0824168 .108122 -.36058
vertex -.0824168 .108897 -.358256
vertex -.0841488 .107347 -.359031
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0841488 .109671 -.359805
vertex -.0841488 .107347 -.359031
vertex -.0824168 .108897 -.358256
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0463998 .0981508 -.327552
vertex -.0481318 .0973762 -.326003
vertex -.0481318 .0997 -.326778
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0463998 .0981508 -.327552
vertex -.0481318 .0997 -.326778
vertex -.0463998 .0989254 -.325228
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0463998 .0981508 -.327552
vertex -.0463998 .0989254 -.325228
vertex -.0481318 .0973762 -.326003
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0481318 .0997 -.326778
vertex -.0481318 .0973762 -.326003
vertex -.0463998 .0989254 -.325228
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0254127 .107242 -.357666
vertex -.0271447 .106468 -.356116
vertex -.0271447 .108791 -.356891
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0254127 .107242 -.357666
vertex -.0271447 .108791 -.356891
vertex -.0254127 .108017 -.355342
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0254127 .107242 -.357666
vertex -.0254127 .108017 -.355342
vertex -.0271447 .106468 -.356116
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0271447 .108791 -.356891
vertex -.0271447 .106468 -.356116
vertex -.0254127 .108017 -.355342
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00308533 .113108 -.377094
vertex -.00481738 .112333 -.375544
vertex -.00481738 .114657 -.376319
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00308533 .113108 -.377094
vertex -.00481738 .114657 -.376319
vertex -.00308533 .113882 -.37477
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00308533 .113108 -.377094
vertex -.00308533 .113882 -.37477
vertex -.00481738 .112333 -.375544
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00481738 .114657 -.376319
vertex -.00481738 .112333 -.375544
vertex -.00308533 .113882 -.37477
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0201622 .110175 -.36738
vertex .0184302 .1094 -.36583
vertex .0184302 .111724 -.366605
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0201622 .110175 -.36738
vertex .0184302 .111724 -.366605
vertex .0201622 .110949 -.365056
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0201622 .110175 -.36738
vertex .0201622 .110949 -.365056
vertex .0184302 .1094 -.36583
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0184302 .111724 -.366605
vertex .0184302 .1094 -.36583
vertex .0201622 .110949 -.365056
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0708025 .117507 -.391665
vertex .0690704 .116732 -.390115
vertex .0690704 .119056 -.39089
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0708025 .117507 -.391665
vertex .0690704 .119056 -.39089
vertex .0708025 .118281 -.389341
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0708025 .117507 -.391665
vertex .0708025 .118281 -.389341
vertex .0690704 .116732 -.390115
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0690704 .119056 -.39089
vertex .0690704 .116732 -.390115
vertex .0708025 .118281 -.389341
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.130985 .103577 -.355026
vertex -.132717 .102803 -.353477
vertex -.132717 .105127 -.354251
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.130985 .103577 -.355026
vertex -.132717 .105127 -.354251
vertex -.130985 .104352 -.352702
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.130985 .103577 -.355026
vertex -.130985 .104352 -.352702
vertex -.132717 .102803 -.353477
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.132717 .105127 -.354251
vertex -.132717 .102803 -.353477
vertex -.130985 .104352 -.352702
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0996893 .100141 -.343328
vertex -.101421 .099366 -.341779
vertex -.101421 .10169 -.342553
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0996893 .100141 -.343328
vertex -.101421 .10169 -.342553
vertex -.0996893 .100915 -.341004
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0996893 .100141 -.343328
vertex -.0996893 .100915 -.341004
vertex -.101421 .099366 -.341779
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.101421 .10169 -.342553
vertex -.101421 .099366 -.341779
vertex -.0996893 .100915 -.341004
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0821917 .105296 -.360875
vertex -.0839237 .104521 -.359326
vertex -.0839237 .106845 -.3601
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0821917 .105296 -.360875
vertex -.0839237 .106845 -.3601
vertex -.0821917 .10607 -.358551
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0821917 .105296 -.360875
vertex -.0821917 .10607 -.358551
vertex -.0839237 .104521 -.359326
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0839237 .106845 -.3601
vertex -.0839237 .104521 -.359326
vertex -.0821917 .10607 -.358551
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.041368 .0961309 -.32968
vertex -.0431001 .0953563 -.328131
vertex -.0431001 .0976801 -.328906
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.041368 .0961309 -.32968
vertex -.0431001 .0976801 -.328906
vertex -.041368 .0969055 -.327356
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.041368 .0961309 -.32968
vertex -.041368 .0969055 -.327356
vertex -.0431001 .0953563 -.328131
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0431001 .0976801 -.328906
vertex -.0431001 .0953563 -.328131
vertex -.041368 .0969055 -.327356
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0212424 .11217 -.384271
vertex -.0229745 .111395 -.382722
vertex -.0229745 .113719 -.383496
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0212424 .11217 -.384271
vertex -.0229745 .113719 -.383496
vertex -.0212424 .112944 -.381947
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0212424 .11217 -.384271
vertex -.0212424 .112944 -.381947
vertex -.0229745 .111395 -.382722
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0229745 .113719 -.383496
vertex -.0229745 .111395 -.382722
vertex -.0212424 .112944 -.381947
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00282883 .109019 -.373548
vertex .00109678 .108245 -.371999
vertex .00109678 .110568 -.372773
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00282883 .109019 -.373548
vertex .00109678 .110568 -.372773
vertex .00282883 .109794 -.371224
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00282883 .109019 -.373548
vertex .00282883 .109794 -.371224
vertex .00109678 .108245 -.371999
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00109678 .110568 -.372773
vertex .00109678 .108245 -.371999
vertex .00282883 .109794 -.371224
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0263478 .109306 -.374523
vertex .0246158 .108531 -.372973
vertex .0246158 .110855 -.373748
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0263478 .109306 -.374523
vertex .0246158 .110855 -.373748
vertex .0263478 .11008 -.372199
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0263478 .109306 -.374523
vertex .0263478 .11008 -.372199
vertex .0246158 .108531 -.372973
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0246158 .110855 -.373748
vertex .0246158 .108531 -.372973
vertex .0263478 .11008 -.372199
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.148108 .101365 -.35725
vertex -.14984 .100591 -.355701
vertex -.14984 .102915 -.356475
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.148108 .101365 -.35725
vertex -.14984 .102915 -.356475
vertex -.148108 .10214 -.354926
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.148108 .101365 -.35725
vertex -.148108 .10214 -.354926
vertex -.14984 .100591 -.355701
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.14984 .102915 -.356475
vertex -.14984 .100591 -.355701
vertex -.148108 .10214 -.354926
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.111196 .0938177 -.330837
vertex -.112928 .0930431 -.329288
vertex -.112928 .0953669 -.330062
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.111196 .0938177 -.330837
vertex -.112928 .0953669 -.330062
vertex -.111196 .0945923 -.328513
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.111196 .0938177 -.330837
vertex -.111196 .0945923 -.328513
vertex -.112928 .0930431 -.329288
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.112928 .0953669 -.330062
vertex -.112928 .0930431 -.329288
vertex -.111196 .0945923 -.328513
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.094302 .0977313 -.344533
vertex -.096034 .0969567 -.342983
vertex -.096034 .0992805 -.343758
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.094302 .0977313 -.344533
vertex -.096034 .0992805 -.343758
vertex -.094302 .0985059 -.342209
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.094302 .0977313 -.344533
vertex -.094302 .0985059 -.342209
vertex -.096034 .0969567 -.342983
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.096034 .0992805 -.343758
vertex -.096034 .0969567 -.342983
vertex -.094302 .0985059 -.342209
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0723627 .113945 -.401272
vertex -.0740947 .11317 -.399723
vertex -.0740947 .115494 -.400497
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0723627 .113945 -.401272
vertex -.0740947 .115494 -.400497
vertex -.0723627 .114719 -.398948
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0723627 .113945 -.401272
vertex -.0723627 .114719 -.398948
vertex -.0740947 .11317 -.399723
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0740947 .115494 -.400497
vertex -.0740947 .11317 -.399723
vertex -.0723627 .114719 -.398948
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0388995 .0940973 -.331815
vertex -.0406315 .0933227 -.330266
vertex -.0406315 .0956465 -.331041
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0388995 .0940973 -.331815
vertex -.0406315 .0956465 -.331041
vertex -.0388995 .0948719 -.329491
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0388995 .0940973 -.331815
vertex -.0388995 .0948719 -.329491
vertex -.0406315 .0933227 -.330266
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0406315 .0956465 -.331041
vertex -.0406315 .0933227 -.330266
vertex -.0388995 .0948719 -.329491
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0168662 .101645 -.358228
vertex -.0185982 .10087 -.356679
vertex -.0185982 .103194 -.357454
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0168662 .101645 -.358228
vertex -.0185982 .103194 -.357454
vertex -.0168662 .102419 -.355904
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0168662 .101645 -.358228
vertex -.0168662 .102419 -.355904
vertex -.0185982 .10087 -.356679
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0185982 .103194 -.357454
vertex -.0185982 .10087 -.356679
vertex -.0168662 .102419 -.355904
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00573477 .105838 -.372902
vertex .00400271 .105063 -.371353
vertex .00400271 .107387 -.372128
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00573477 .105838 -.372902
vertex .00400271 .107387 -.372128
vertex .00573477 .106613 -.370578
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00573477 .105838 -.372902
vertex .00573477 .106613 -.370578
vertex .00400271 .105063 -.371353
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00400271 .107387 -.372128
vertex .00400271 .105063 -.371353
vertex .00573477 .106613 -.370578
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0295839 .107795 -.37975
vertex .0278518 .10702 -.378201
vertex .0278518 .109344 -.378975
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0295839 .107795 -.37975
vertex .0278518 .109344 -.378975
vertex .0295839 .108569 -.377426
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0295839 .107795 -.37975
vertex .0295839 .108569 -.377426
vertex .0278518 .10702 -.378201
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0278518 .109344 -.378975
vertex .0278518 .10702 -.378201
vertex .0295839 .108569 -.377426
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.148518 .0991395 -.359481
vertex -.15025 .0983649 -.357932
vertex -.15025 .100689 -.358706
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.148518 .0991395 -.359481
vertex -.15025 .100689 -.358706
vertex -.148518 .0999141 -.357157
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.148518 .0991395 -.359481
vertex -.148518 .0999141 -.357157
vertex -.15025 .0983649 -.357932
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.15025 .100689 -.358706
vertex -.15025 .0983649 -.357932
vertex -.148518 .0999141 -.357157
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.125761 .0988668 -.358499
vertex -.127493 .0980922 -.35695
vertex -.127493 .100416 -.357725
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.125761 .0988668 -.358499
vertex -.127493 .100416 -.357725
vertex -.125761 .0996414 -.356175
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.125761 .0988668 -.358499
vertex -.125761 .0996414 -.356175
vertex -.127493 .0980922 -.35695
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.127493 .100416 -.357725
vertex -.127493 .0980922 -.35695
vertex -.125761 .0996414 -.356175
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0953228 .093686 -.339847
vertex -.0970549 .0929114 -.338298
vertex -.0970549 .0952352 -.339072
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0953228 .093686 -.339847
vertex -.0970549 .0952352 -.339072
vertex -.0953228 .0944606 -.337523
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0953228 .093686 -.339847
vertex -.0953228 .0944606 -.337523
vertex -.0970549 .0929114 -.338298
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0970549 .0952352 -.339072
vertex -.0970549 .0929114 -.338298
vertex -.0953228 .0944606 -.337523
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0561999 .0904138 -.328067
vertex -.0579319 .0896392 -.326517
vertex -.0579319 .091963 -.327292
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0561999 .0904138 -.328067
vertex -.0579319 .091963 -.327292
vertex -.0561999 .0911884 -.325743
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0561999 .0904138 -.328067
vertex -.0561999 .0911884 -.325743
vertex -.0579319 .0896392 -.326517
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0579319 .091963 -.327292
vertex -.0579319 .0896392 -.326517
vertex -.0561999 .0911884 -.325743
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0363056 .0917772 -.332975
vertex -.0380376 .0910026 -.331426
vertex -.0380376 .0933264 -.332201
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0363056 .0917772 -.332975
vertex -.0380376 .0933264 -.332201
vertex -.0363056 .0925518 -.330651
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0363056 .0917772 -.332975
vertex -.0363056 .0925518 -.330651
vertex -.0380376 .0910026 -.331426
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0380376 .0933264 -.332201
vertex -.0380376 .0910026 -.331426
vertex -.0363056 .0925518 -.330651
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00897406 .104593 -.379115
vertex -.0107061 .103818 -.377566
vertex -.0107061 .106142 -.37834
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00897406 .104593 -.379115
vertex -.0107061 .106142 -.37834
vertex -.00897406 .105368 -.376791
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00897406 .104593 -.379115
vertex -.00897406 .105368 -.376791
vertex -.0107061 .103818 -.377566
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0107061 .106142 -.37834
vertex -.0107061 .103818 -.377566
vertex -.00897406 .105368 -.376791
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .01434 .102139 -.37028
vertex .012608 .101364 -.36873
vertex .012608 .103688 -.369505
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .01434 .102139 -.37028
vertex .012608 .103688 -.369505
vertex .01434 .102914 -.367956
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .01434 .102139 -.37028
vertex .01434 .102914 -.367956
vertex .012608 .101364 -.36873
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .012608 .103688 -.369505
vertex .012608 .101364 -.36873
vertex .01434 .102914 -.367956
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0673848 .108683 -.39384
vertex .0656528 .107909 -.392291
vertex .0656528 .110232 -.393066
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0673848 .108683 -.39384
vertex .0656528 .110232 -.393066
vertex .0673848 .109458 -.391516
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0673848 .108683 -.39384
vertex .0673848 .109458 -.391516
vertex .0656528 .107909 -.392291
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0656528 .110232 -.393066
vertex .0656528 .107909 -.392291
vertex .0673848 .109458 -.391516
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.140501 .0968999 -.361719
vertex -.142233 .0961253 -.36017
vertex -.142233 .0984491 -.360944
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.140501 .0968999 -.361719
vertex -.142233 .0984491 -.360944
vertex -.140501 .0976745 -.359395
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.140501 .0968999 -.361719
vertex -.140501 .0976745 -.359395
vertex -.142233 .0961253 -.36017
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.142233 .0984491 -.360944
vertex -.142233 .0961253 -.36017
vertex -.140501 .0976745 -.359395
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.10158 .0875965 -.327239
vertex -.103312 .0868219 -.32569
vertex -.103312 .0891457 -.326465
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.10158 .0875965 -.327239
vertex -.103312 .0891457 -.326465
vertex -.10158 .0883711 -.324915
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.10158 .0875965 -.327239
vertex -.10158 .0883711 -.324915
vertex -.103312 .0868219 -.32569
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.103312 .0891457 -.326465
vertex -.103312 .0868219 -.32569
vertex -.10158 .0883711 -.324915
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0859729 .0926469 -.345957
vertex -.087705 .0918723 -.344408
vertex -.087705 .0941961 -.345182
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0859729 .0926469 -.345957
vertex -.087705 .0941961 -.345182
vertex -.0859729 .0934215 -.343633
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0859729 .0926469 -.345957
vertex -.0859729 .0934215 -.343633
vertex -.087705 .0918723 -.344408
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.087705 .0941961 -.345182
vertex -.087705 .0918723 -.344408
vertex -.0859729 .0934215 -.343633
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.04883 .0886597 -.33118
vertex -.050562 .0878851 -.329631
vertex -.050562 .0902089 -.330405
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.04883 .0886597 -.33118
vertex -.050562 .0902089 -.330405
vertex -.04883 .0894343 -.328856
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.04883 .0886597 -.33118
vertex -.04883 .0894343 -.328856
vertex -.050562 .0878851 -.329631
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.050562 .0902089 -.330405
vertex -.050562 .0878851 -.329631
vertex -.04883 .0894343 -.328856
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0244102 .0934444 -.348912
vertex -.0261423 .0926698 -.347363
vertex -.0261423 .0949936 -.348137
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0244102 .0934444 -.348912
vertex -.0261423 .0949936 -.348137
vertex -.0244102 .094219 -.346588
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0244102 .0934444 -.348912
vertex -.0244102 .094219 -.346588
vertex -.0261423 .0926698 -.347363
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0261423 .0949936 -.348137
vertex -.0261423 .0926698 -.347363
vertex -.0244102 .094219 -.346588
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -9.76483e-5 .100621 -.375511
vertex -.0018297 .0998467 -.373961
vertex -.0018297 .10217 -.374736
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -9.76483e-5 .100621 -.375511
vertex -.0018297 .10217 -.374736
vertex -9.76483e-5 .101396 -.373187
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -9.76483e-5 .100621 -.375511
vertex -9.76483e-5 .101396 -.373187
vertex -.0018297 .0998467 -.373961
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0018297 .10217 -.374736
vertex -.0018297 .0998467 -.373961
vertex -9.76483e-5 .101396 -.373187
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0524202 .106735 -.398168
vertex .0506881 .10596 -.396619
vertex .0506881 .108284 -.397394
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0524202 .106735 -.398168
vertex .0506881 .108284 -.397394
vertex .0524202 .10751 -.395845
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0524202 .106735 -.398168
vertex .0524202 .10751 -.395845
vertex .0506881 .10596 -.396619
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0506881 .108284 -.397394
vertex .0506881 .10596 -.396619
vertex .0524202 .10751 -.395845
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0756473 .104608 -.390287
vertex .0739153 .103834 -.388738
vertex .0739153 .106158 -.389513
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0756473 .104608 -.390287
vertex .0739153 .106158 -.389513
vertex .0756473 .105383 -.387964
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0756473 .104608 -.390287
vertex .0756473 .105383 -.387964
vertex .0739153 .103834 -.388738
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0739153 .106158 -.389513
vertex .0739153 .103834 -.388738
vertex .0756473 .105383 -.387964
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.135254 .0946466 -.363963
vertex -.136986 .093872 -.362414
vertex -.136986 .0961958 -.363189
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.135254 .0946466 -.363963
vertex -.136986 .0961958 -.363189
vertex -.135254 .0954212 -.36164
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.135254 .0946466 -.363963
vertex -.135254 .0954212 -.36164
vertex -.136986 .093872 -.362414
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.136986 .0961958 -.363189
vertex -.136986 .093872 -.362414
vertex -.135254 .0954212 -.36164
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0987382 .0850656 -.327387
vertex -.10047 .084291 -.325838
vertex -.10047 .0866147 -.326612
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0987382 .0850656 -.327387
vertex -.10047 .0866147 -.326612
vertex -.0987382 .0858401 -.325063
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0987382 .0850656 -.327387
vertex -.0987382 .0858401 -.325063
vertex -.10047 .084291 -.325838
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.10047 .0866147 -.326612
vertex -.10047 .084291 -.325838
vertex -.0987382 .0858401 -.325063
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0825656 .0894677 -.344192
vertex -.0842977 .0886931 -.342643
vertex -.0842977 .0910169 -.343418
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0825656 .0894677 -.344192
vertex -.0842977 .0910169 -.343418
vertex -.0825656 .0902423 -.341868
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0825656 .0894677 -.344192
vertex -.0825656 .0902423 -.341868
vertex -.0842977 .0886931 -.342643
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0842977 .0910169 -.343418
vertex -.0842977 .0886931 -.342643
vertex -.0825656 .0902423 -.341868
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.04883 .0863603 -.332329
vertex -.050562 .0855857 -.33078
vertex -.050562 .0879095 -.331555
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.04883 .0863603 -.332329
vertex -.050562 .0879095 -.331555
vertex -.04883 .0871349 -.330006
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.04883 .0863603 -.332329
vertex -.04883 .0871349 -.330006
vertex -.050562 .0855857 -.33078
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.050562 .0879095 -.331555
vertex -.050562 .0855857 -.33078
vertex -.04883 .0871349 -.330006
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0291012 .0974951 -.374838
vertex -.0308332 .0967205 -.373288
vertex -.0308332 .0990443 -.374063
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0291012 .0974951 -.374838
vertex -.0308332 .0990443 -.374063
vertex -.0291012 .0982697 -.372514
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0291012 .0974951 -.374838
vertex -.0291012 .0982697 -.372514
vertex -.0308332 .0967205 -.373288
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0308332 .0990443 -.374063
vertex -.0308332 .0967205 -.373288
vertex -.0291012 .0982697 -.372514
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -9.00403e-5 .0972361 -.373849
vertex -.00182209 .0964615 -.3723
vertex -.00182209 .0987853 -.373074
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -9.00403e-5 .0972361 -.373849
vertex -.00182209 .0987853 -.373074
vertex -9.00403e-5 .0980107 -.371525
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -9.00403e-5 .0972361 -.373849
vertex -9.00403e-5 .0980107 -.371525
vertex -.00182209 .0964615 -.3723
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00182209 .0987853 -.373074
vertex -.00182209 .0964615 -.3723
vertex -9.00403e-5 .0980107 -.371525
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0235852 .0995666 -.382746
vertex .0218531 .098792 -.381197
vertex .0218531 .101116 -.381971
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0235852 .0995666 -.382746
vertex .0218531 .101116 -.381971
vertex .0235852 .100341 -.380422
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0235852 .0995666 -.382746
vertex .0235852 .100341 -.380422
vertex .0218531 .098792 -.381197
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0218531 .101116 -.381971
vertex .0218531 .098792 -.381197
vertex .0235852 .100341 -.380422
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .072616 .101897 -.391643
vertex .0708839 .101123 -.390094
vertex .0708839 .103446 -.390868
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .072616 .101897 -.391643
vertex .0708839 .103446 -.390868
vertex .072616 .102672 -.389319
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .072616 .101897 -.391643
vertex .072616 .102672 -.389319
vertex .0708839 .101123 -.390094
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0708839 .103446 -.390868
vertex .0708839 .101123 -.390094
vertex .072616 .102672 -.389319
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.138449 .0923796 -.366215
vertex -.140181 .091605 -.364666
vertex -.140181 .0939288 -.36544
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.138449 .0923796 -.366215
vertex -.140181 .0939288 -.36544
vertex -.138449 .0931542 -.363891
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.138449 .0923796 -.366215
vertex -.138449 .0931542 -.363891
vertex -.140181 .091605 -.364666
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.140181 .0939288 -.36544
vertex -.140181 .091605 -.364666
vertex -.138449 .0931542 -.363891
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.100652 .0822963 -.326535
vertex -.102384 .0815217 -.324986
vertex -.102384 .0838454 -.325761
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.100652 .0822963 -.326535
vertex -.102384 .0838454 -.325761
vertex -.100652 .0830708 -.324211
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.100652 .0822963 -.326535
vertex -.100652 .0830708 -.324211
vertex -.102384 .0815217 -.324986
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.102384 .0838454 -.325761
vertex -.102384 .0815217 -.324986
vertex -.100652 .0830708 -.324211
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0842362 .0860775 -.341415
vertex -.0859682 .0853029 -.339866
vertex -.0859682 .0876267 -.340641
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0842362 .0860775 -.341415
vertex -.0859682 .0876267 -.340641
vertex -.0842362 .0868521 -.339091
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0842362 .0860775 -.341415
vertex -.0842362 .0868521 -.339091
vertex -.0859682 .0853029 -.339866
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0859682 .0876267 -.340641
vertex -.0859682 .0853029 -.339866
vertex -.0842362 .0868521 -.339091
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0561999 .0835567 -.331495
vertex -.0579319 .0827821 -.329946
vertex -.0579319 .0851059 -.330721
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0561999 .0835567 -.331495
vertex -.0579319 .0851059 -.330721
vertex -.0561999 .0843313 -.329171
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0561999 .0835567 -.331495
vertex -.0561999 .0843313 -.329171
vertex -.0579319 .0827821 -.329946
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0579319 .0851059 -.330721
vertex -.0579319 .0827821 -.329946
vertex -.0561999 .0843313 -.329171
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0377014 .0946484 -.375143
vertex -.0394335 .0938738 -.373594
vertex -.0394335 .0961976 -.374368
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0377014 .0946484 -.375143
vertex -.0394335 .0961976 -.374368
vertex -.0377014 .095423 -.372819
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0377014 .0946484 -.375143
vertex -.0377014 .095423 -.372819
vertex -.0394335 .0938738 -.373594
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0394335 .0961976 -.374368
vertex -.0394335 .0938738 -.373594
vertex -.0377014 .095423 -.372819
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0114301 .0928838 -.368199
vertex -.0131622 .0921092 -.36665
vertex -.0131622 .094433 -.367424
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0114301 .0928838 -.368199
vertex -.0131622 .094433 -.367424
vertex -.0114301 .0936584 -.365875
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0114301 .0928838 -.368199
vertex -.0114301 .0936584 -.365875
vertex -.0131622 .0921092 -.36665
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0131622 .094433 -.367424
vertex -.0131622 .0921092 -.36665
vertex -.0114301 .0936584 -.365875
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0114827 .0946484 -.375143
vertex .00975064 .0938738 -.373594
vertex .00975064 .0961976 -.374368
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0114827 .0946484 -.375143
vertex .00975064 .0961976 -.374368
vertex .0114827 .095423 -.372819
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0114827 .0946484 -.375143
vertex .0114827 .095423 -.372819
vertex .00975064 .0938738 -.373594
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00975064 .0961976 -.374368
vertex .00975064 .0938738 -.373594
vertex .0114827 .095423 -.372819
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0638393 .09969 -.394983
vertex .0621072 .0989154 -.393434
vertex .0621072 .101239 -.394208
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0638393 .09969 -.394983
vertex .0621072 .101239 -.394208
vertex .0638393 .100465 -.392659
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0638393 .09969 -.394983
vertex .0638393 .100465 -.392659
vertex .0621072 .0989154 -.393434
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0621072 .101239 -.394208
vertex .0621072 .0989154 -.393434
vertex .0638393 .100465 -.392659
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.150155 .0900989 -.368473
vertex -.151887 .0893243 -.366924
vertex -.151887 .0916481 -.367699
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.150155 .0900989 -.368473
vertex -.151887 .0916481 -.367699
vertex -.150155 .0908735 -.36615
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.150155 .0900989 -.368473
vertex -.150155 .0908735 -.36615
vertex -.151887 .0893243 -.366924
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.151887 .0916481 -.367699
vertex -.151887 .0893243 -.366924
vertex -.150155 .0908735 -.36615
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.127152 .0898536 -.367478
vertex -.128884 .0890791 -.365929
vertex -.128884 .0914028 -.366703
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.127152 .0898536 -.367478
vertex -.128884 .0914028 -.366703
vertex -.127152 .0906282 -.365154
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.127152 .0898536 -.367478
vertex -.127152 .0906282 -.365154
vertex -.128884 .0890791 -.365929
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.128884 .0914028 -.366703
vertex -.128884 .0890791 -.365929
vertex -.127152 .0906282 -.365154
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0931002 .0800448 -.327661
vertex -.0948323 .0792702 -.326112
vertex -.0948323 .081594 -.326886
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0931002 .0800448 -.327661
vertex -.0948323 .081594 -.326886
vertex -.0931002 .0808194 -.325337
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0931002 .0800448 -.327661
vertex -.0931002 .0808194 -.325337
vertex -.0948323 .0792702 -.326112
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0948323 .081594 -.326886
vertex -.0948323 .0792702 -.326112
vertex -.0931002 .0808194 -.325337
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0634776 .0807805 -.330647
vertex -.0652097 .0800059 -.329098
vertex -.0652097 .0823297 -.329873
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0634776 .0807805 -.330647
vertex -.0652097 .0823297 -.329873
vertex -.0634776 .0815551 -.328323
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0634776 .0807805 -.330647
vertex -.0634776 .0815551 -.328323
vertex -.0652097 .0800059 -.329098
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0652097 .0823297 -.329873
vertex -.0652097 .0800059 -.329098
vertex -.0634776 .0815551 -.328323
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0436883 .0817614 -.334629
vertex -.0454203 .0809868 -.33308
vertex -.0454203 .0833106 -.333854
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0436883 .0817614 -.334629
vertex -.0454203 .0833106 -.333854
vertex -.0436883 .082536 -.332305
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0436883 .0817614 -.334629
vertex -.0436883 .082536 -.332305
vertex -.0454203 .0809868 -.33308
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0454203 .0833106 -.333854
vertex -.0454203 .0809868 -.33308
vertex -.0436883 .082536 -.332305
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0234273 .0927963 -.379423
vertex -.0251594 .0920217 -.377874
vertex -.0251594 .0943455 -.378648
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0234273 .0927963 -.379423
vertex -.0251594 .0943455 -.378648
vertex -.0234273 .0935709 -.377099
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0234273 .0927963 -.379423
vertex -.0234273 .0935709 -.377099
vertex -.0251594 .0920217 -.377874
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0251594 .0943455 -.378648
vertex -.0251594 .0920217 -.377874
vertex -.0234273 .0935709 -.377099
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00278259 .0910797 -.372455
vertex .00105054 .0903051 -.370906
vertex .00105054 .0926289 -.37168
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00278259 .0910797 -.372455
vertex .00105054 .0926289 -.37168
vertex .00278259 .0918543 -.370131
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00278259 .0910797 -.372455
vertex .00278259 .0918543 -.370131
vertex .00105054 .0903051 -.370906
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00105054 .0926289 -.37168
vertex .00105054 .0903051 -.370906
vertex .00278259 .0918543 -.370131
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0519085 .0974555 -.398336
vertex .0501764 .0966809 -.396787
vertex .0501764 .0990047 -.397562
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0519085 .0974555 -.398336
vertex .0501764 .0990047 -.397562
vertex .0519085 .0982301 -.396012
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0519085 .0974555 -.398336
vertex .0519085 .0982301 -.396012
vertex .0501764 .0966809 -.396787
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0501764 .0990047 -.397562
vertex .0501764 .0966809 -.396787
vertex .0519085 .0982301 -.396012
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0747007 .0952485 -.389377
vertex .0729687 .0944739 -.387828
vertex .0729687 .0967977 -.388603
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0747007 .0952485 -.389377
vertex .0729687 .0967977 -.388603
vertex .0747007 .0960231 -.387053
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0747007 .0952485 -.389377
vertex .0747007 .0960231 -.387053
vertex .0729687 .0944739 -.387828
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0729687 .0967977 -.388603
vertex .0729687 .0944739 -.387828
vertex .0747007 .0960231 -.387053
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.136367 .0878044 -.370739
vertex -.138099 .0870298 -.369189
vertex -.138099 .0893536 -.369964
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.136367 .0878044 -.370739
vertex -.138099 .0893536 -.369964
vertex -.136367 .088579 -.368415
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.136367 .0878044 -.370739
vertex -.136367 .088579 -.368415
vertex -.138099 .0870298 -.369189
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.138099 .0893536 -.369964
vertex -.138099 .0870298 -.369189
vertex -.136367 .088579 -.368415
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0978327 .0775551 -.327788
vertex -.0995647 .0767805 -.326239
vertex -.0995647 .0791043 -.327013
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0978327 .0775551 -.327788
vertex -.0995647 .0791043 -.327013
vertex -.0978327 .0783297 -.325464
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0978327 .0775551 -.327788
vertex -.0978327 .0783297 -.325464
vertex -.0995647 .0767805 -.326239
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0995647 .0791043 -.327013
vertex -.0995647 .0767805 -.326239
vertex -.0978327 .0783297 -.325464
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0799208 .0797003 -.336778
vertex -.0816529 .0789257 -.335228
vertex -.0816529 .0812495 -.336003
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0799208 .0797003 -.336778
vertex -.0816529 .0812495 -.336003
vertex -.0799208 .0804749 -.334454
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0799208 .0797003 -.336778
vertex -.0799208 .0804749 -.334454
vertex -.0816529 .0789257 -.335228
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0816529 .0812495 -.336003
vertex -.0816529 .0789257 -.335228
vertex -.0799208 .0804749 -.334454
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0536443 .0789852 -.333781
vertex -.0553764 .0782106 -.332232
vertex -.0553764 .0805344 -.333006
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0536443 .0789852 -.333781
vertex -.0553764 .0805344 -.333006
vertex -.0536443 .0797598 -.331457
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0536443 .0789852 -.333781
vertex -.0536443 .0797598 -.331457
vertex -.0553764 .0782106 -.332232
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0553764 .0805344 -.333006
vertex -.0553764 .0782106 -.332232
vertex -.0536443 .0797598 -.331457
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0348082 .0894729 -.377731
vertex -.0365403 .0886983 -.376181
vertex -.0365403 .0910221 -.376956
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0348082 .0894729 -.377731
vertex -.0365403 .0910221 -.376956
vertex -.0348082 .0902475 -.375407
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0348082 .0894729 -.377731
vertex -.0348082 .0902475 -.375407
vertex -.0365403 .0886983 -.376181
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0365403 .0910221 -.376956
vertex -.0365403 .0886983 -.376181
vertex -.0348082 .0902475 -.375407
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.011696 .0897112 -.378729
vertex -.0134281 .0889366 -.37718
vertex -.0134281 .0912604 -.377955
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.011696 .0897112 -.378729
vertex -.0134281 .0912604 -.377955
vertex -.011696 .0904858 -.376406
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.011696 .0897112 -.378729
vertex -.011696 .0904858 -.376406
vertex -.0134281 .0889366 -.37718
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0134281 .0912604 -.377955
vertex -.0134281 .0889366 -.37718
vertex -.011696 .0904858 -.376406
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0229547 .0787469 -.332782
vertex .0212226 .0779723 -.331233
vertex .0212226 .0802961 -.332008
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0229547 .0787469 -.332782
vertex .0212226 .0802961 -.332008
vertex .0229547 .0795215 -.330458
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0229547 .0787469 -.332782
vertex .0229547 .0795215 -.330458
vertex .0212226 .0779723 -.331233
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0212226 .0802961 -.332008
vertex .0212226 .0779723 -.331233
vertex .0229547 .0795215 -.330458
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0633634 .0935249 -.394711
vertex .0616314 .0927503 -.393162
vertex .0616314 .0950741 -.393937
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0633634 .0935249 -.394711
vertex .0616314 .0950741 -.393937
vertex .0633634 .0942995 -.392387
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0633634 .0935249 -.394711
vertex .0633634 .0942995 -.392387
vertex .0616314 .0927503 -.393162
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0616314 .0950741 -.393937
vertex .0616314 .0927503 -.393162
vertex .0633634 .0942995 -.392387
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.147724 .0852647 -.372008
vertex -.149456 .0844901 -.370459
vertex -.149456 .0868139 -.371234
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.147724 .0852647 -.372008
vertex -.149456 .0868139 -.371234
vertex -.147724 .0860393 -.369685
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.147724 .0852647 -.372008
vertex -.147724 .0860393 -.369685
vertex -.149456 .0844901 -.370459
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.149456 .0868139 -.371234
vertex -.149456 .0844901 -.370459
vertex -.147724 .0860393 -.369685
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.125349 .0854962 -.373011
vertex -.127081 .0847216 -.371462
vertex -.127081 .0870454 -.372236
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.125349 .0854962 -.373011
vertex -.127081 .0870454 -.372236
vertex -.125349 .0862708 -.370687
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.125349 .0854962 -.373011
vertex -.125349 .0862708 -.370687
vertex -.127081 .0847216 -.371462
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.127081 .0870454 -.372236
vertex -.127081 .0847216 -.371462
vertex -.125349 .0862708 -.370687
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0877948 .0753105 -.32891
vertex -.0895268 .0745359 -.327361
vertex -.0895268 .0768597 -.328136
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0877948 .0753105 -.32891
vertex -.0895268 .0768597 -.328136
vertex -.0877948 .0760851 -.326586
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0877948 .0753105 -.32891
vertex -.0877948 .0760851 -.326586
vertex -.0895268 .0745359 -.327361
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0895268 .0768597 -.328136
vertex -.0895268 .0745359 -.327361
vertex -.0877948 .0760851 -.326586
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0681385 .0757735 -.330915
vertex -.0698705 .0749989 -.329365
vertex -.0698705 .0773227 -.33014
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0681385 .0757735 -.330915
vertex -.0698705 .0773227 -.33014
vertex -.0681385 .0765481 -.328591
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0681385 .0757735 -.330915
vertex -.0681385 .0765481 -.328591
vertex -.0698705 .0749989 -.329365
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0698705 .0773227 -.33014
vertex -.0698705 .0749989 -.329365
vertex -.0681385 .0765481 -.328591
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.051886 .0864222 -.37702
vertex -.0536181 .0856476 -.375471
vertex -.0536181 .0879713 -.376245
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.051886 .0864222 -.37702
vertex -.0536181 .0879713 -.376245
vertex -.051886 .0871968 -.374696
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.051886 .0864222 -.37702
vertex -.051886 .0871968 -.374696
vertex -.0536181 .0856476 -.375471
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0536181 .0879713 -.376245
vertex -.0536181 .0856476 -.375471
vertex -.051886 .0871968 -.374696
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0291804 .0873481 -.381029
vertex -.0309125 .0865735 -.37948
vertex -.0309125 .0888973 -.380254
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0291804 .0873481 -.381029
vertex -.0309125 .0888973 -.380254
vertex -.0291804 .0881227 -.378705
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0291804 .0873481 -.381029
vertex -.0291804 .0881227 -.378705
vertex -.0309125 .0865735 -.37948
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0309125 .0888973 -.380254
vertex -.0309125 .0865735 -.37948
vertex -.0291804 .0881227 -.378705
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -7.73604e-5 .0857277 -.374013
vertex -.00180941 .0849531 -.372464
vertex -.00180941 .0872769 -.373238
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -7.73604e-5 .0857277 -.374013
vertex -.00180941 .0872769 -.373238
vertex -7.73604e-5 .0865023 -.371689
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -7.73604e-5 .0857277 -.374013
vertex -7.73604e-5 .0865023 -.371689
vertex -.00180941 .0849531 -.372464
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00180941 .0872769 -.373238
vertex -.00180941 .0849531 -.372464
vertex -7.73604e-5 .0865023 -.371689
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0545637 .0912835 -.398068
vertex .0528316 .0905089 -.396519
vertex .0528316 .0928327 -.397293
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0545637 .0912835 -.398068
vertex .0528316 .0928327 -.397293
vertex .0545637 .0920581 -.395744
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0545637 .0912835 -.398068
vertex .0545637 .0920581 -.395744
vertex .0528316 .0905089 -.396519
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0528316 .0928327 -.397293
vertex .0528316 .0905089 -.396519
vertex .0545637 .0920581 -.395744
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.157353 .0818265 -.369255
vertex -.159085 .0810519 -.367706
vertex -.159085 .0833757 -.368481
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.157353 .0818265 -.369255
vertex -.159085 .0833757 -.368481
vertex -.157353 .0826011 -.366932
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.157353 .0818265 -.369255
vertex -.157353 .0826011 -.366932
vertex -.159085 .0810519 -.367706
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.159085 .0833757 -.368481
vertex -.159085 .0810519 -.367706
vertex -.157353 .0826011 -.366932
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.136737 .0829496 -.374284
vertex -.13847 .082175 -.372735
vertex -.13847 .0844988 -.373509
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.136737 .0829496 -.374284
vertex -.13847 .0844988 -.373509
vertex -.136737 .0837242 -.37196
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.136737 .0829496 -.374284
vertex -.136737 .0837242 -.37196
vertex -.13847 .082175 -.372735
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.13847 .0844988 -.373509
vertex -.13847 .082175 -.372735
vertex -.136737 .0837242 -.37196
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0981345 .0732906 -.331038
vertex -.0998666 .072516 -.329489
vertex -.0998666 .0748398 -.330263
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0981345 .0732906 -.331038
vertex -.0998666 .0748398 -.330263
vertex -.0981345 .0740652 -.328714
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0981345 .0732906 -.331038
vertex -.0981345 .0740652 -.328714
vertex -.0998666 .072516 -.329489
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0998666 .0748398 -.330263
vertex -.0998666 .072516 -.329489
vertex -.0981345 .0740652 -.328714
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0782378 .0735153 -.332044
vertex -.0799698 .0727407 -.330495
vertex -.0799698 .0750645 -.331269
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0782378 .0735153 -.332044
vertex -.0799698 .0750645 -.331269
vertex -.0782378 .0742899 -.32972
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0782378 .0735153 -.332044
vertex -.0782378 .0742899 -.32972
vertex -.0799698 .0727407 -.330495
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0799698 .0750645 -.331269
vertex -.0799698 .0727407 -.330495
vertex -.0782378 .0742899 -.32972
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0583973 .0739645 -.334055
vertex -.0601293 .0731899 -.332506
vertex -.0601293 .0755137 -.333281
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0583973 .0739645 -.334055
vertex -.0601293 .0755137 -.333281
vertex -.0583973 .0747391 -.331731
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0583973 .0739645 -.334055
vertex -.0583973 .0747391 -.331731
vertex -.0601293 .0731899 -.332506
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0601293 .0755137 -.333281
vertex -.0601293 .0731899 -.332506
vertex -.0583973 .0747391 -.331731
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0404846 .0840728 -.379313
vertex -.0422167 .0832982 -.377763
vertex -.0422167 .085622 -.378538
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0404846 .0840728 -.379313
vertex -.0422167 .085622 -.378538
vertex -.0404846 .0848474 -.376989
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0404846 .0840728 -.379313
vertex -.0404846 .0848474 -.376989
vertex -.0422167 .0832982 -.377763
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0422167 .085622 -.378538
vertex -.0422167 .0832982 -.377763
vertex -.0404846 .0848474 -.376989
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0147196 .0851959 -.384341
vertex -.0164516 .0844213 -.382792
vertex -.0164516 .0867451 -.383567
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0147196 .0851959 -.384341
vertex -.0164516 .0867451 -.383567
vertex -.0147196 .0859705 -.382017
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0147196 .0851959 -.384341
vertex -.0147196 .0859705 -.382017
vertex -.0164516 .0844213 -.382792
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0164516 .0867451 -.383567
vertex -.0164516 .0844213 -.382792
vertex -.0147196 .0859705 -.382017
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0427224 .0892392 -.402444
vertex .0409904 .0884646 -.400895
vertex .0409904 .0907884 -.40167
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0427224 .0892392 -.402444
vertex .0409904 .0907884 -.40167
vertex .0427224 .0900138 -.40012
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0427224 .0892392 -.402444
vertex .0427224 .0900138 -.40012
vertex .0409904 .0884646 -.400895
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0409904 .0907884 -.40167
vertex .0409904 .0884646 -.400895
vertex .0427224 .0900138 -.40012
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0657219 .0872176 -.393393
vertex .0639898 .086443 -.391844
vertex .0639898 .0887668 -.392618
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0657219 .0872176 -.393393
vertex .0639898 .0887668 -.392618
vertex .0657219 .0879922 -.391069
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0657219 .0872176 -.393393
vertex .0657219 .0879922 -.391069
vertex .0639898 .086443 -.391844
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0639898 .0887668 -.392618
vertex .0639898 .086443 -.391844
vertex .0657219 .0879922 -.391069
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.147724 .0801853 -.374548
vertex -.149456 .0794107 -.372999
vertex -.149456 .0817345 -.373774
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.147724 .0801853 -.374548
vertex -.149456 .0817345 -.373774
vertex -.147724 .0809599 -.372224
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.147724 .0801853 -.374548
vertex -.147724 .0809599 -.372224
vertex -.149456 .0794107 -.372999
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.149456 .0817345 -.373774
vertex -.149456 .0794107 -.372999
vertex -.147724 .0809599 -.372224
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.125689 .0806208 -.376566
vertex -.127421 .0798462 -.375017
vertex -.127421 .08217 -.375792
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.125689 .0806208 -.376566
vertex -.127421 .08217 -.375792
vertex -.125689 .0813954 -.374243
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.125689 .0806208 -.376566
vertex -.125689 .0813954 -.374243
vertex -.127421 .0798462 -.375017
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.127421 .08217 -.375792
vertex -.127421 .0798462 -.375017
vertex -.125689 .0813954 -.374243
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0883371 .071257 -.333173
vertex -.0900691 .0704824 -.331624
vertex -.0900691 .0728062 -.332398
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0883371 .071257 -.333173
vertex -.0900691 .0728062 -.332398
vertex -.0883371 .0720316 -.330849
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0883371 .071257 -.333173
vertex -.0883371 .0720316 -.330849
vertex -.0900691 .0704824 -.331624
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0900691 .0728062 -.332398
vertex -.0900691 .0704824 -.331624
vertex -.0883371 .0720316 -.330849
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0681385 .071257 -.333173
vertex -.0698705 .0704824 -.331624
vertex -.0698705 .0728062 -.332398
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0681385 .071257 -.333173
vertex -.0698705 .0728062 -.332398
vertex -.0681385 .0720316 -.330849
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0681385 .071257 -.333173
vertex -.0681385 .0720316 -.330849
vertex -.0698705 .0704824 -.331624
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0698705 .0728062 -.332398
vertex -.0698705 .0704824 -.331624
vertex -.0681385 .0720316 -.330849
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0516047 .0808386 -.377576
vertex -.0533367 .080064 -.376026
vertex -.0533367 .0823878 -.376801
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0516047 .0808386 -.377576
vertex -.0533367 .0823878 -.376801
vertex -.0516047 .0816132 -.375252
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0516047 .0808386 -.377576
vertex -.0516047 .0816132 -.375252
vertex -.0533367 .080064 -.376026
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0533367 .0823878 -.376801
vertex -.0533367 .080064 -.376026
vertex -.0516047 .0816132 -.375252
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0291804 .0821452 -.383631
vertex -.0309125 .0813706 -.382081
vertex -.0309125 .0836944 -.382856
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0291804 .0821452 -.383631
vertex -.0309125 .0836944 -.382856
vertex -.0291804 .0829198 -.381307
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0291804 .0821452 -.383631
vertex -.0291804 .0829198 -.381307
vertex -.0309125 .0813706 -.382081
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0309125 .0836944 -.382856
vertex -.0309125 .0813706 -.382081
vertex -.0291804 .0829198 -.381307
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00333037 .0891136 -.415923
vertex -.00506242 .088339 -.414374
vertex -.00506242 .0906628 -.415149
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00333037 .0891136 -.415923
vertex -.00506242 .0906628 -.415149
vertex -.00333037 .0898882 -.4136
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00333037 .0891136 -.415923
vertex -.00333037 .0898882 -.4136
vertex -.00506242 .088339 -.414374
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00506242 .0906628 -.415149
vertex -.00506242 .088339 -.414374
vertex -.00333037 .0898882 -.4136
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0571728 .0851939 -.397759
vertex .0554408 .0844193 -.396209
vertex .0554408 .0867431 -.396984
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0571728 .0851939 -.397759
vertex .0554408 .0867431 -.396984
vertex .0571728 .0859685 -.395435
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0571728 .0851939 -.397759
vertex .0571728 .0859685 -.395435
vertex .0554408 .0844193 -.396209
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0554408 .0867431 -.396984
vertex .0554408 .0844193 -.396209
vertex .0571728 .0859685 -.395435
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.15327 .0761693 -.36873
vertex -.155003 .0753947 -.367181
vertex -.155003 .0777185 -.367955
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.15327 .0761693 -.36873
vertex -.155003 .0777185 -.367955
vertex -.15327 .0769439 -.366406
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.15327 .0761693 -.36873
vertex -.15327 .0769439 -.366406
vertex -.155003 .0753947 -.367181
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.155003 .0777185 -.367955
vertex -.155003 .0753947 -.367181
vertex -.15327 .0769439 -.366406
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.134254 .0780674 -.377843
vertex -.135986 .0772928 -.376294
vertex -.135986 .0796166 -.377069
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.134254 .0780674 -.377843
vertex -.135986 .0796166 -.377069
vertex -.134254 .078842 -.375519
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.134254 .0780674 -.377843
vertex -.134254 .078842 -.375519
vertex -.135986 .0772928 -.376294
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.135986 .0796166 -.377069
vertex -.135986 .0772928 -.376294
vertex -.134254 .078842 -.375519
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0956174 .0687878 -.333289
vertex -.0973494 .0680132 -.33174
vertex -.0973494 .070337 -.332515
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0956174 .0687878 -.333289
vertex -.0973494 .070337 -.332515
vertex -.0956174 .0695624 -.330966
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0956174 .0687878 -.333289
vertex -.0956174 .0695624 -.330966
vertex -.0973494 .0680132 -.33174
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0973494 .070337 -.332515
vertex -.0973494 .0680132 -.33174
vertex -.0956174 .0695624 -.330966
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.072963 .0687878 -.333289
vertex -.0746951 .0680132 -.33174
vertex -.0746951 .070337 -.332515
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.072963 .0687878 -.333289
vertex -.0746951 .070337 -.332515
vertex -.072963 .0695624 -.330966
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.072963 .0687878 -.333289
vertex -.072963 .0695624 -.330966
vertex -.0746951 .0680132 -.33174
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0746951 .070337 -.332515
vertex -.0746951 .0680132 -.33174
vertex -.072963 .0695624 -.330966
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0573296 .0782783 -.378856
vertex -.0590617 .0775037 -.377307
vertex -.0590617 .0798275 -.378081
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0573296 .0782783 -.378856
vertex -.0590617 .0798275 -.378081
vertex -.0573296 .0790529 -.376532
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0573296 .0782783 -.378856
vertex -.0573296 .0790529 -.376532
vertex -.0590617 .0775037 -.377307
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0590617 .0798275 -.378081
vertex -.0590617 .0775037 -.377307
vertex -.0573296 .0790529 -.376532
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0348082 .0791219 -.382906
vertex -.0365403 .0783473 -.381357
vertex -.0365403 .0806711 -.382132
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0348082 .0791219 -.382906
vertex -.0365403 .0806711 -.382132
vertex -.0348082 .0798965 -.380582
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0348082 .0791219 -.382906
vertex -.0348082 .0798965 -.380582
vertex -.0365403 .0783473 -.381357
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0365403 .0806711 -.382132
vertex -.0365403 .0783473 -.381357
vertex -.0348082 .0798965 -.380582
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00966415 .0865034 -.418347
vertex -.0113962 .0857288 -.416797
vertex -.0113962 .0880526 -.417572
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00966415 .0865034 -.418347
vertex -.0113962 .0880526 -.417572
vertex -.00966415 .087278 -.416023
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00966415 .0865034 -.418347
vertex -.00966415 .087278 -.416023
vertex -.0113962 .0857288 -.416797
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0113962 .0880526 -.417572
vertex -.0113962 .0857288 -.416797
vertex -.00966415 .087278 -.416023
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0511409 .0824963 -.399107
vertex .0494089 .0817217 -.397558
vertex .0494089 .0840455 -.398333
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0511409 .0824963 -.399107
vertex .0494089 .0840455 -.398333
vertex .0511409 .0832709 -.396784
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0511409 .0824963 -.399107
vertex .0511409 .0832709 -.396784
vertex .0494089 .0817217 -.397558
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0494089 .0840455 -.398333
vertex .0494089 .0817217 -.397558
vertex .0511409 .0832709 -.396784
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0735648 .0805982 -.389994
vertex .0718327 .0798236 -.388445
vertex .0718327 .0821474 -.38922
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0735648 .0805982 -.389994
vertex .0718327 .0821474 -.38922
vertex .0735648 .0813728 -.38767
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0735648 .0805982 -.389994
vertex .0735648 .0813728 -.38767
vertex .0718327 .0798236 -.388445
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0718327 .0821474 -.38922
vertex .0718327 .0798236 -.388445
vertex .0735648 .0813728 -.38767
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.136737 .07531 -.378104
vertex -.13847 .0745354 -.376555
vertex -.13847 .0768591 -.377329
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.136737 .07531 -.378104
vertex -.13847 .0768591 -.377329
vertex -.136737 .0760845 -.37578
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.136737 .07531 -.378104
vertex -.136737 .0760845 -.37578
vertex -.13847 .0745354 -.376555
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.13847 .0768591 -.377329
vertex -.13847 .0745354 -.376555
vertex -.136737 .0760845 -.37578
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0962057 .0669445 -.336447
vertex -.0979377 .0661699 -.334898
vertex -.0979377 .0684937 -.335673
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0962057 .0669445 -.336447
vertex -.0979377 .0684937 -.335673
vertex -.0962057 .0677191 -.334123
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0962057 .0669445 -.336447
vertex -.0962057 .0677191 -.334123
vertex -.0979377 .0661699 -.334898
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0979377 .0684937 -.335673
vertex -.0979377 .0661699 -.334898
vertex -.0962057 .0677191 -.334123
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0752474 .0663324 -.333399
vertex -.0769795 .0655578 -.33185
vertex -.0769795 .0678816 -.332625
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0752474 .0663324 -.333399
vertex -.0769795 .0678816 -.332625
vertex -.0752474 .067107 -.331075
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0752474 .0663324 -.333399
vertex -.0752474 .067107 -.331075
vertex -.0769795 .0655578 -.33185
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0769795 .0678816 -.332625
vertex -.0769795 .0655578 -.33185
vertex -.0752474 .067107 -.331075
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0571736 .075514 -.37912
vertex -.0589057 .0747394 -.377571
vertex -.0589057 .0770632 -.378345
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0571736 .075514 -.37912
vertex -.0589057 .0770632 -.378345
vertex -.0571736 .0762886 -.376796
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0571736 .075514 -.37912
vertex -.0571736 .0762886 -.376796
vertex -.0589057 .0747394 -.377571
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0589057 .0770632 -.378345
vertex -.0589057 .0747394 -.377571
vertex -.0571736 .0762886 -.376796
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0348082 .0765342 -.3842
vertex -.0365403 .0757596 -.382651
vertex -.0365403 .0780834 -.383425
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0348082 .0765342 -.3842
vertex -.0365403 .0780834 -.383425
vertex -.0348082 .0773088 -.381876
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0348082 .0765342 -.3842
vertex -.0348082 .0773088 -.381876
vertex -.0365403 .0757596 -.382651
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0365403 .0780834 -.383425
vertex -.0365403 .0757596 -.382651
vertex -.0348082 .0773088 -.381876
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0304001 .0808189 -.405536
vertex .0286681 .0800443 -.403987
vertex .0286681 .0823681 -.404762
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0304001 .0808189 -.405536
vertex .0286681 .0823681 -.404762
vertex .0304001 .0815935 -.403213
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0304001 .0808189 -.405536
vertex .0304001 .0815935 -.403213
vertex .0286681 .0800443 -.403987
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0286681 .0823681 -.404762
vertex .0286681 .0800443 -.403987
vertex .0304001 .0815935 -.403213
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0537501 .0791866 -.397408
vertex .052018 .078412 -.395859
vertex .052018 .0807358 -.396634
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0537501 .0791866 -.397408
vertex .052018 .0807358 -.396634
vertex .0537501 .0799612 -.395084
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0537501 .0791866 -.397408
vertex .0537501 .0799612 -.395084
vertex .052018 .078412 -.395859
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .052018 .0807358 -.396634
vertex .052018 .078412 -.395859
vertex .0537501 .0799612 -.395084
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0761177 .0775543 -.38928
vertex .0743857 .0767798 -.387731
vertex .0743857 .0791035 -.388505
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0761177 .0775543 -.38928
vertex .0743857 .0791035 -.388505
vertex .0761177 .0783289 -.386956
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0761177 .0775543 -.38928
vertex .0761177 .0783289 -.386956
vertex .0743857 .0767798 -.387731
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0743857 .0791035 -.388505
vertex .0743857 .0767798 -.387731
vertex .0761177 .0783289 -.386956
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.131754 .0731577 -.381416
vertex -.133486 .0723831 -.379867
vertex -.133486 .0747069 -.380641
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.131754 .0731577 -.381416
vertex -.133486 .0747069 -.380641
vertex -.131754 .0739323 -.379092
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.131754 .0731577 -.381416
vertex -.131754 .0739323 -.379092
vertex -.133486 .0723831 -.379867
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.133486 .0747069 -.380641
vertex -.133486 .0723831 -.379867
vertex -.131754 .0739323 -.379092
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0939597 .0648765 -.338599
vertex -.0956917 .0641019 -.33705
vertex -.0956917 .0664257 -.337825
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0939597 .0648765 -.338599
vertex -.0956917 .0664257 -.337825
vertex -.0939597 .0656511 -.336275
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0939597 .0648765 -.338599
vertex -.0939597 .0656511 -.336275
vertex -.0956917 .0641019 -.33705
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0956917 .0664257 -.337825
vertex -.0956917 .0641019 -.33705
vertex -.0939597 .0656511 -.336275
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.072738 .0640878 -.334521
vertex -.07447 .0633132 -.332972
vertex -.07447 .065637 -.333747
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.072738 .0640878 -.334521
vertex -.07447 .065637 -.333747
vertex -.072738 .0648624 -.332198
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.072738 .0640878 -.334521
vertex -.072738 .0648624 -.332198
vertex -.07447 .0633132 -.332972
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.07447 .065637 -.333747
vertex -.07447 .0633132 -.332972
vertex -.072738 .0648624 -.332198
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0543188 .0729606 -.380397
vertex -.0560509 .072186 -.378847
vertex -.0560509 .0745098 -.379622
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0543188 .0729606 -.380397
vertex -.0560509 .0745098 -.379622
vertex -.0543188 .0737352 -.378073
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0543188 .0729606 -.380397
vertex -.0543188 .0737352 -.378073
vertex -.0560509 .072186 -.378847
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0560509 .0745098 -.379622
vertex -.0560509 .072186 -.378847
vertex -.0543188 .0737352 -.378073
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0319151 .0739464 -.385494
vertex -.0336471 .0731718 -.383945
vertex -.0336471 .0754956 -.384719
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0319151 .0739464 -.385494
vertex -.0336471 .0754956 -.384719
vertex -.0319151 .074721 -.38317
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0319151 .0739464 -.385494
vertex -.0319151 .074721 -.38317
vertex -.0336471 .0731718 -.383945
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0336471 .0754956 -.384719
vertex -.0336471 .0731718 -.383945
vertex -.0319151 .074721 -.38317
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0421979 .0772983 -.402824
vertex .0404658 .0765237 -.401275
vertex .0404658 .0788475 -.40205
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0421979 .0772983 -.402824
vertex .0404658 .0788475 -.40205
vertex .0421979 .0780729 -.400501
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0421979 .0772983 -.402824
vertex .0421979 .0780729 -.400501
vertex .0404658 .0765237 -.401275
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0404658 .0788475 -.40205
vertex .0404658 .0765237 -.401275
vertex .0421979 .0780729 -.400501
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0648904 .0755238 -.393649
vertex .0631583 .0747492 -.3921
vertex .0631583 .077073 -.392875
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0648904 .0755238 -.393649
vertex .0631583 .077073 -.392875
vertex .0648904 .0762984 -.391326
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0648904 .0755238 -.393649
vertex .0648904 .0762984 -.391326
vertex .0631583 .0747492 -.3921
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0631583 .077073 -.392875
vertex .0631583 .0747492 -.3921
vertex .0648904 .0762984 -.391326
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.138569 .0683138 -.370422
vertex -.140301 .0675392 -.368872
vertex -.140301 .069863 -.369647
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.138569 .0683138 -.370422
vertex -.140301 .069863 -.369647
vertex -.138569 .0690884 -.368098
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.138569 .0683138 -.370422
vertex -.138569 .0690884 -.368098
vertex -.140301 .0675392 -.368872
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.140301 .069863 -.369647
vertex -.140301 .0675392 -.368872
vertex -.138569 .0690884 -.368098
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.10412 .0626045 -.339735
vertex -.105852 .0618299 -.338186
vertex -.105852 .0641537 -.338961
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.10412 .0626045 -.339735
vertex -.105852 .0641537 -.338961
vertex -.10412 .0633791 -.337411
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.10412 .0626045 -.339735
vertex -.10412 .0633791 -.337411
vertex -.105852 .0618299 -.338186
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.105852 .0641537 -.338961
vertex -.105852 .0618299 -.338186
vertex -.10412 .0633791 -.337411
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0800183 .061653 -.334621
vertex -.0817503 .0608784 -.333072
vertex -.0817503 .0632022 -.333846
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0800183 .061653 -.334621
vertex -.0817503 .0632022 -.333846
vertex -.0800183 .0624276 -.332297
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0800183 .061653 -.334621
vertex -.0800183 .0624276 -.332297
vertex -.0817503 .0608784 -.333072
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0817503 .0632022 -.333846
vertex -.0817503 .0608784 -.333072
vertex -.0800183 .0624276 -.332297
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.060564 .0622239 -.337689
vertex -.0622961 .0614493 -.33614
vertex -.0622961 .0637731 -.336915
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.060564 .0622239 -.337689
vertex -.0622961 .0637731 -.336915
vertex -.060564 .0629985 -.335366
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.060564 .0622239 -.337689
vertex -.060564 .0629985 -.335366
vertex -.0622961 .0614493 -.33614
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0622961 .0637731 -.336915
vertex -.0622961 .0614493 -.33614
vertex -.060564 .0629985 -.335366
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0458797 .0705975 -.382696
vertex -.0476117 .0698229 -.381147
vertex -.0476117 .0721466 -.381922
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0458797 .0705975 -.382696
vertex -.0476117 .0721466 -.381922
vertex -.0458797 .0713721 -.380372
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0458797 .0705975 -.382696
vertex -.0458797 .0713721 -.380372
vertex -.0476117 .0698229 -.381147
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0476117 .0721466 -.381922
vertex -.0476117 .0698229 -.381147
vertex -.0458797 .0713721 -.380372
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00945968 .076497 -.414405
vertex -.0111917 .0757224 -.412856
vertex -.0111917 .0780462 -.413631
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00945968 .076497 -.414405
vertex -.0111917 .0780462 -.413631
vertex -.00945968 .0772716 -.412082
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00945968 .076497 -.414405
vertex -.00945968 .0772716 -.412082
vertex -.0111917 .0757224 -.412856
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0111917 .0780462 -.413631
vertex -.0111917 .0757224 -.412856
vertex -.00945968 .0772716 -.412082
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0419881 .0742133 -.402131
vertex .040256 .0734387 -.400582
vertex .040256 .0757625 -.401356
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0419881 .0742133 -.402131
vertex .040256 .0757625 -.401356
vertex .0419881 .0749879 -.399807
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0419881 .0742133 -.402131
vertex .0419881 .0749879 -.399807
vertex .040256 .0734387 -.400582
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .040256 .0757625 -.401356
vertex .040256 .0734387 -.400582
vertex .0419881 .0749879 -.399807
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .067497 .0725005 -.392925
vertex .065765 .0717259 -.391376
vertex .065765 .0740497 -.39215
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .067497 .0725005 -.392925
vertex .065765 .0740497 -.39215
vertex .067497 .0732751 -.390601
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .067497 .0725005 -.392925
vertex .067497 .0732751 -.390601
vertex .065765 .0717259 -.391376
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .065765 .0740497 -.39215
vertex .065765 .0717259 -.391376
vertex .067497 .0732751 -.390601
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.137411 .0652855 -.368582
vertex -.139143 .0645109 -.367032
vertex -.139143 .0668347 -.367807
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.137411 .0652855 -.368582
vertex -.139143 .0668347 -.367807
vertex -.137411 .0660601 -.366258
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.137411 .0652855 -.368582
vertex -.137411 .0660601 -.366258
vertex -.139143 .0645109 -.367032
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.139143 .0668347 -.367807
vertex -.139143 .0645109 -.367032
vertex -.137411 .0660601 -.366258
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.104438 .060516 -.341898
vertex -.10617 .0597414 -.340348
vertex -.10617 .0620652 -.341123
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.104438 .060516 -.341898
vertex -.10617 .0620652 -.341123
vertex -.104438 .0612906 -.339574
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.104438 .060516 -.341898
vertex -.104438 .0612906 -.339574
vertex -.10617 .0597414 -.340348
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.10617 .0620652 -.341123
vertex -.10617 .0597414 -.340348
vertex -.104438 .0612906 -.339574
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0752474 .0595988 -.336766
vertex -.0769795 .0588242 -.335217
vertex -.0769795 .061148 -.335991
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0752474 .0595988 -.336766
vertex -.0769795 .061148 -.335991
vertex -.0752474 .0603734 -.334442
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0752474 .0595988 -.336766
vertex -.0752474 .0603734 -.334442
vertex -.0769795 .0588242 -.335217
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0769795 .061148 -.335991
vertex -.0769795 .0588242 -.335217
vertex -.0752474 .0603734 -.334442
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0598647 .0676703 -.381924
vertex -.0615968 .0668957 -.380375
vertex -.0615968 .0692195 -.381149
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0598647 .0676703 -.381924
vertex -.0615968 .0692195 -.381149
vertex -.0598647 .0684449 -.3796
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0598647 .0676703 -.381924
vertex -.0598647 .0684449 -.3796
vertex -.0615968 .0668957 -.380375
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0615968 .0692195 -.381149
vertex -.0615968 .0668957 -.380375
vertex -.0598647 .0684449 -.3796
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0375991 .0685875 -.387055
vertex -.0393312 .0678129 -.385506
vertex -.0393312 .0701367 -.386281
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0375991 .0685875 -.387055
vertex -.0393312 .0701367 -.386281
vertex -.0375991 .0693621 -.384732
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0375991 .0685875 -.387055
vertex -.0375991 .0693621 -.384732
vertex -.0393312 .0678129 -.385506
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0393312 .0701367 -.386281
vertex -.0393312 .0678129 -.385506
vertex -.0375991 .0693621 -.384732
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0241737 .0722564 -.407582
vertex .0224417 .0714818 -.406032
vertex .0224417 .0738056 -.406807
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0241737 .0722564 -.407582
vertex .0224417 .0738056 -.406807
vertex .0241737 .073031 -.405258
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0241737 .0722564 -.407582
vertex .0241737 .073031 -.405258
vertex .0224417 .0714818 -.406032
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0224417 .0738056 -.406807
vertex .0224417 .0714818 -.406032
vertex .0241737 .073031 -.405258
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0503733 .0706054 -.398345
vertex .0486413 .0698308 -.396796
vertex .0486413 .0721546 -.39757
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0503733 .0706054 -.398345
vertex .0486413 .0721546 -.39757
vertex .0503733 .07138 -.396021
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0503733 .0706054 -.398345
vertex .0503733 .07138 -.396021
vertex .0486413 .0698308 -.396796
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0486413 .0721546 -.39757
vertex .0486413 .0698308 -.396796
vertex .0503733 .07138 -.396021
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0728075 .0693213 -.391161
vertex .0710755 .0685467 -.389611
vertex .0710755 .0708705 -.390386
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0728075 .0693213 -.391161
vertex .0710755 .0708705 -.390386
vertex .0728075 .0700959 -.388837
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0728075 .0693213 -.391161
vertex .0728075 .0700959 -.388837
vertex .0710755 .0685467 -.389611
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0710755 .0708705 -.390386
vertex .0710755 .0685467 -.389611
vertex .0728075 .0700959 -.388837
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.129895 .0631813 -.37187
vertex -.131627 .0624067 -.370321
vertex -.131627 .0647305 -.371095
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.129895 .0631813 -.37187
vertex -.131627 .0647305 -.371095
vertex -.129895 .0639559 -.369546
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.129895 .0631813 -.37187
vertex -.129895 .0639559 -.369546
vertex -.131627 .0624067 -.370321
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.131627 .0647305 -.371095
vertex -.131627 .0624067 -.370321
vertex -.129895 .0639559 -.369546
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0973823 .0585903 -.345096
vertex -.0991144 .0578157 -.343547
vertex -.0991144 .0601395 -.344322
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0973823 .0585903 -.345096
vertex -.0991144 .0601395 -.344322
vertex -.0973823 .0593649 -.342773
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0973823 .0585903 -.345096
vertex -.0973823 .0593649 -.342773
vertex -.0991144 .0578157 -.343547
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0991144 .0601395 -.344322
vertex -.0991144 .0578157 -.343547
vertex -.0973823 .0593649 -.342773
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0650219 .064594 -.380108
vertex -.0667539 .0638194 -.378559
vertex -.0667539 .0661432 -.379333
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0650219 .064594 -.380108
vertex -.0667539 .0661432 -.379333
vertex -.0650219 .0653686 -.377784
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0650219 .064594 -.380108
vertex -.0650219 .0653686 -.377784
vertex -.0667539 .0638194 -.378559
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0667539 .0661432 -.379333
vertex -.0667539 .0638194 -.378559
vertex -.0650219 .0653686 -.377784
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0431349 .0656535 -.386286
vertex -.0448669 .0648789 -.384737
vertex -.0448669 .0672027 -.385512
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0431349 .0656535 -.386286
vertex -.0448669 .0672027 -.385512
vertex -.0431349 .0664281 -.383962
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0431349 .0656535 -.386286
vertex -.0431349 .0664281 -.383962
vertex -.0448669 .0648789 -.384737
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0448669 .0672027 -.385512
vertex -.0448669 .0648789 -.384737
vertex -.0431349 .0664281 -.383962
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0211859 .0697148 -.40997
vertex .0194539 .0689402 -.408421
vertex .0194539 .071264 -.409196
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0211859 .0697148 -.40997
vertex .0194539 .071264 -.409196
vertex .0211859 .0704894 -.407647
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0211859 .0697148 -.40997
vertex .0211859 .0704894 -.407647
vertex .0194539 .0689402 -.408421
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0194539 .071264 -.409196
vertex .0194539 .0689402 -.408421
vertex .0211859 .0704894 -.407647
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0474034 .067949 -.399673
vertex .0456714 .0671744 -.398124
vertex .0456714 .0694982 -.398898
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0474034 .067949 -.399673
vertex .0456714 .0694982 -.398898
vertex .0474034 .0687236 -.397349
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0474034 .067949 -.399673
vertex .0474034 .0687236 -.397349
vertex .0456714 .0671744 -.398124
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0456714 .0694982 -.398898
vertex .0456714 .0671744 -.398124
vertex .0474034 .0687236 -.397349
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0698913 .0667129 -.392465
vertex .0681593 .0659383 -.390916
vertex .0681593 .0682621 -.39169
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0698913 .0667129 -.392465
vertex .0681593 .0682621 -.39169
vertex .0698913 .0674875 -.390141
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0698913 .0667129 -.392465
vertex .0698913 .0674875 -.390141
vertex .0681593 .0659383 -.390916
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0681593 .0682621 -.39169
vertex .0681593 .0659383 -.390916
vertex .0698913 .0674875 -.390141
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.130432 .059692 -.366906
vertex -.132164 .0589174 -.365357
vertex -.132164 .0612412 -.366132
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.130432 .059692 -.366906
vertex -.132164 .0612412 -.366132
vertex -.130432 .0604666 -.364582
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.130432 .059692 -.366906
vertex -.130432 .0604666 -.364582
vertex -.132164 .0589174 -.365357
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.132164 .0612412 -.366132
vertex -.132164 .0589174 -.365357
vertex -.130432 .0604666 -.364582
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.102509 .0562977 -.346243
vertex -.104241 .0555231 -.344694
vertex -.104241 .0578469 -.345468
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.102509 .0562977 -.346243
vertex -.104241 .0578469 -.345468
vertex -.102509 .0570723 -.343919
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.102509 .0562977 -.346243
vertex -.102509 .0570723 -.343919
vertex -.104241 .0555231 -.344694
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.104241 .0578469 -.345468
vertex -.104241 .0555231 -.344694
vertex -.102509 .0570723 -.343919
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.059701 .0624074 -.383437
vertex -.0614331 .0616328 -.381888
vertex -.0614331 .0639566 -.382663
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.059701 .0624074 -.383437
vertex -.0614331 .0639566 -.382663
vertex -.059701 .063182 -.381113
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.059701 .0624074 -.383437
vertex -.059701 .063182 -.381113
vertex -.0614331 .0616328 -.381888
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0614331 .0639566 -.382663
vertex -.0614331 .0616328 -.381888
vertex -.059701 .063182 -.381113
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0375991 .0634257 -.389636
vertex -.0393312 .0626511 -.388087
vertex -.0393312 .0649749 -.388862
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0375991 .0634257 -.389636
vertex -.0393312 .0649749 -.388862
vertex -.0375991 .0642003 -.387312
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0375991 .0634257 -.389636
vertex -.0375991 .0642003 -.387312
vertex -.0393312 .0626511 -.388087
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0393312 .0649749 -.388862
vertex -.0393312 .0626511 -.388087
vertex -.0375991 .0642003 -.387312
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0299549 .0661412 -.406167
vertex .0282228 .0653666 -.404618
vertex .0282228 .0676904 -.405392
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0299549 .0661412 -.406167
vertex .0282228 .0676904 -.405392
vertex .0299549 .0669158 -.403843
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0299549 .0661412 -.406167
vertex .0299549 .0669158 -.403843
vertex .0282228 .0653666 -.404618
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0282228 .0676904 -.405392
vertex .0282228 .0653666 -.404618
vertex .0299549 .0669158 -.403843
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0555968 .064444 -.395835
vertex .0538648 .0636694 -.394286
vertex .0538648 .0659932 -.395061
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0555968 .064444 -.395835
vertex .0538648 .0659932 -.395061
vertex .0555968 .0652186 -.393511
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0555968 .064444 -.395835
vertex .0555968 .0652186 -.393511
vertex .0538648 .0636694 -.394286
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0538648 .0659932 -.395061
vertex .0538648 .0636694 -.394286
vertex .0555968 .0652186 -.393511
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.136613 .0564479 -.362938
vertex -.138345 .0556733 -.361389
vertex -.138345 .0579971 -.362164
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.136613 .0564479 -.362938
vertex -.138345 .0579971 -.362164
vertex -.136613 .0572225 -.360614
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.136613 .0564479 -.362938
vertex -.136613 .0572225 -.360614
vertex -.138345 .0556733 -.361389
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.138345 .0579971 -.362164
vertex -.138345 .0556733 -.361389
vertex -.136613 .0572225 -.360614
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.110198 .0540051 -.347389
vertex -.11193 .0532305 -.34584
vertex -.11193 .0555543 -.346614
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.110198 .0540051 -.347389
vertex -.11193 .0555543 -.346614
vertex -.110198 .0547797 -.345065
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.110198 .0540051 -.347389
vertex -.110198 .0547797 -.345065
vertex -.11193 .0532305 -.34584
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.11193 .0555543 -.346614
vertex -.11193 .0532305 -.34584
vertex -.110198 .0547797 -.345065
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0658157 .0533537 -.343243
vertex -.0675478 .0525791 -.341693
vertex -.0675478 .0549029 -.342468
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0658157 .0533537 -.343243
vertex -.0675478 .0549029 -.342468
vertex -.0658157 .0541283 -.340919
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0658157 .0533537 -.343243
vertex -.0658157 .0541283 -.340919
vertex -.0675478 .0525791 -.341693
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0675478 .0549029 -.342468
vertex -.0675478 .0525791 -.341693
vertex -.0658157 .0541283 -.340919
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0458797 .0603563 -.387817
vertex -.0476117 .0595817 -.386268
vertex -.0476117 .0619055 -.387042
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0458797 .0603563 -.387817
vertex -.0476117 .0619055 -.387042
vertex -.0458797 .0611309 -.385493
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0458797 .0603563 -.387817
vertex -.0458797 .0611309 -.385493
vertex -.0476117 .0595817 -.386268
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0476117 .0619055 -.387042
vertex -.0476117 .0595817 -.386268
vertex -.0458797 .0611309 -.385493
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0210324 .0637762 -.409586
vertex .0193003 .0630016 -.408036
vertex .0193003 .0653254 -.408811
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0210324 .0637762 -.409586
vertex .0193003 .0653254 -.408811
vertex .0210324 .0645508 -.407262
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0210324 .0637762 -.409586
vertex .0210324 .0645508 -.407262
vertex .0193003 .0630016 -.408036
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0193003 .0653254 -.408811
vertex .0193003 .0630016 -.408036
vertex .0210324 .0645508 -.407262
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0442083 .0623105 -.400256
vertex .0424763 .0615359 -.398707
vertex .0424763 .0638597 -.399481
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0442083 .0623105 -.400256
vertex .0424763 .0638597 -.399481
vertex .0442083 .0630851 -.397932
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0442083 .0623105 -.400256
vertex .0442083 .0630851 -.397932
vertex .0424763 .0615359 -.398707
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0424763 .0638597 -.399481
vertex .0424763 .0615359 -.398707
vertex .0442083 .0630851 -.397932
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0691647 .0608449 -.390927
vertex .0674327 .0600703 -.389377
vertex .0674327 .0623941 -.390152
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0691647 .0608449 -.390927
vertex .0674327 .0623941 -.390152
vertex .0691647 .0616195 -.388603
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0691647 .0608449 -.390927
vertex .0691647 .0616195 -.388603
vertex .0674327 .0600703 -.389377
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0674327 .0623941 -.390152
vertex .0674327 .0600703 -.389377
vertex .0691647 .0616195 -.388603
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.115673 .0518685 -.349575
vertex -.117405 .0510939 -.348026
vertex -.117405 .0534177 -.348801
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.115673 .0518685 -.349575
vertex -.117405 .0534177 -.348801
vertex -.115673 .0526431 -.347252
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.115673 .0518685 -.349575
vertex -.115673 .0526431 -.347252
vertex -.117405 .0510939 -.348026
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.117405 .0534177 -.348801
vertex -.117405 .0510939 -.348026
vertex -.115673 .0526431 -.347252
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0953921 .0520245 -.350615
vertex -.0971241 .0512499 -.349066
vertex -.0971241 .0535737 -.349841
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0953921 .0520245 -.350615
vertex -.0971241 .0535737 -.349841
vertex -.0953921 .0527991 -.348292
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0953921 .0520245 -.350615
vertex -.0953921 .0527991 -.348292
vertex -.0971241 .0512499 -.349066
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0971241 .0535737 -.349841
vertex -.0971241 .0512499 -.349066
vertex -.0953921 .0527991 -.348292
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0486092 .05764 -.388057
vertex -.0503412 .0568654 -.386508
vertex -.0503412 .0591892 -.387282
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0486092 .05764 -.388057
vertex -.0503412 .0591892 -.387282
vertex -.0486092 .0584146 -.385733
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0486092 .05764 -.388057
vertex -.0486092 .0584146 -.385733
vertex -.0503412 .0568654 -.386508
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0503412 .0591892 -.387282
vertex -.0503412 .0568654 -.386508
vertex -.0486092 .0584146 -.385733
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0150209 .0612277 -.411978
vertex .0132889 .0604531 -.410429
vertex .0132889 .0627769 -.411203
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0150209 .0612277 -.411978
vertex .0132889 .0627769 -.411203
vertex .0150209 .0620023 -.409654
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0150209 .0612277 -.411978
vertex .0150209 .0620023 -.409654
vertex .0132889 .0604531 -.410429
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0132889 .0627769 -.411203
vertex .0132889 .0604531 -.410429
vertex .0150209 .0620023 -.409654
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0383964 .0598239 -.402617
vertex .0366644 .0590493 -.401068
vertex .0366644 .061373 -.401843
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0383964 .0598239 -.402617
vertex .0366644 .061373 -.401843
vertex .0383964 .0605984 -.400294
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0383964 .0598239 -.402617
vertex .0383964 .0605984 -.400294
vertex .0366644 .0590493 -.401068
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0366644 .061373 -.401843
vertex .0366644 .0590493 -.401068
vertex .0383964 .0605984 -.400294
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0608254 .058576 -.394297
vertex .0590934 .0578014 -.392748
vertex .0590934 .0601252 -.393523
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0608254 .058576 -.394297
vertex .0590934 .0601252 -.393523
vertex .0608254 .0593506 -.391973
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0608254 .058576 -.394297
vertex .0608254 .0593506 -.391973
vertex .0590934 .0578014 -.392748
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0590934 .0601252 -.393523
vertex .0590934 .0578014 -.392748
vertex .0608254 .0593506 -.391973
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.129363 .0509112 -.360116
vertex -.131095 .0501366 -.358567
vertex -.131095 .0524603 -.359342
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.129363 .0509112 -.360116
vertex -.131095 .0524603 -.359342
vertex -.129363 .0516858 -.357793
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.129363 .0509112 -.360116
vertex -.129363 .0516858 -.357793
vertex -.131095 .0501366 -.358567
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.131095 .0524603 -.359342
vertex -.131095 .0501366 -.358567
vertex -.129363 .0516858 -.357793
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.103437 .0498673 -.352812
vertex -.105169 .0490927 -.351263
vertex -.105169 .0514165 -.352037
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.103437 .0498673 -.352812
vertex -.105169 .0514165 -.352037
vertex -.103437 .0506419 -.350488
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.103437 .0498673 -.352812
vertex -.103437 .0506419 -.350488
vertex -.105169 .0490927 -.351263
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.105169 .0514165 -.352037
vertex -.105169 .0490927 -.351263
vertex -.103437 .0506419 -.350488
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0486092 .0550866 -.389334
vertex -.0503412 .054312 -.387784
vertex -.0503412 .0566358 -.388559
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0486092 .0550866 -.389334
vertex -.0503412 .0566358 -.388559
vertex -.0486092 .0558612 -.38701
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0486092 .0550866 -.389334
vertex -.0486092 .0558612 -.38701
vertex -.0503412 .054312 -.387784
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0503412 .0566358 -.388559
vertex -.0503412 .054312 -.387784
vertex -.0486092 .0558612 -.38701
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0119614 .0583673 -.41229
vertex .0102294 .0575927 -.410741
vertex .0102294 .0599165 -.411515
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0119614 .0583673 -.41229
vertex .0102294 .0599165 -.411515
vertex .0119614 .0591419 -.409966
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0119614 .0583673 -.41229
vertex .0119614 .0591419 -.409966
vertex .0102294 .0575927 -.410741
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0102294 .0599165 -.411515
vertex .0102294 .0575927 -.410741
vertex .0119614 .0591419 -.409966
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0354342 .0571743 -.403942
vertex .0337021 .0563997 -.402393
vertex .0337021 .0587235 -.403168
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0354342 .0571743 -.403942
vertex .0337021 .0587235 -.403168
vertex .0354342 .0579489 -.401618
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0354342 .0571743 -.403942
vertex .0354342 .0579489 -.401618
vertex .0337021 .0563997 -.402393
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0337021 .0587235 -.403168
vertex .0337021 .0563997 -.402393
vertex .0354342 .0579489 -.401618
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0579246 .0559813 -.395594
vertex .0561925 .0552068 -.394045
vertex .0561925 .0575305 -.39482
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0579246 .0559813 -.395594
vertex .0561925 .0575305 -.39482
vertex .0579246 .0567559 -.393271
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0579246 .0559813 -.395594
vertex .0579246 .0567559 -.393271
vertex .0561925 .0552068 -.394045
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0561925 .0575305 -.39482
vertex .0561925 .0552068 -.394045
vertex .0579246 .0567559 -.393271
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.131231 .0482654 -.359203
vertex -.132963 .0474908 -.357654
vertex -.132963 .0498146 -.358429
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.131231 .0482654 -.359203
vertex -.132963 .0498146 -.358429
vertex -.131231 .04904 -.356879
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.131231 .0482654 -.359203
vertex -.131231 .04904 -.356879
vertex -.132963 .0474908 -.357654
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.132963 .0498146 -.358429
vertex -.132963 .0474908 -.357654
vertex -.131231 .04904 -.356879
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.10861 .0475541 -.353969
vertex -.110342 .0467795 -.352419
vertex -.110342 .0491033 -.353194
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.10861 .0475541 -.353969
vertex -.110342 .0491033 -.353194
vertex -.10861 .0483287 -.351645
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.10861 .0475541 -.353969
vertex -.10861 .0483287 -.351645
vertex -.110342 .0467795 -.352419
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.110342 .0491033 -.353194
vertex -.110342 .0467795 -.352419
vertex -.10861 .0483287 -.351645
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0457544 .0525332 -.39061
vertex -.0474864 .0517586 -.389061
vertex -.0474864 .0540824 -.389836
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0457544 .0525332 -.39061
vertex -.0474864 .0540824 -.389836
vertex -.0457544 .0533078 -.388287
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0457544 .0525332 -.39061
vertex -.0457544 .0533078 -.388287
vertex -.0474864 .0517586 -.389061
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0474864 .0540824 -.389836
vertex -.0474864 .0517586 -.389061
vertex -.0457544 .0533078 -.388287
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0119333 .0555206 -.412595
vertex .0102012 .054746 -.411046
vertex .0102012 .0570698 -.411821
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0119333 .0555206 -.412595
vertex .0102012 .0570698 -.411821
vertex .0119333 .0562952 -.410272
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0119333 .0555206 -.412595
vertex .0119333 .0562952 -.410272
vertex .0102012 .054746 -.411046
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0102012 .0570698 -.411821
vertex .0102012 .054746 -.411046
vertex .0119333 .0562952 -.410272
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0353446 .0543826 -.40422
vertex .0336126 .053608 -.402671
vertex .0336126 .0559318 -.403445
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0353446 .0543826 -.40422
vertex .0336126 .0559318 -.403445
vertex .0353446 .0551572 -.401896
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0353446 .0543826 -.40422
vertex .0353446 .0551572 -.401896
vertex .0336126 .053608 -.402671
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0336126 .0559318 -.403445
vertex .0336126 .053608 -.402671
vertex .0353446 .0551572 -.401896
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0577736 .0532445 -.395845
vertex .0560416 .0524699 -.394296
vertex .0560416 .0547937 -.39507
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0577736 .0532445 -.395845
vertex .0560416 .0547937 -.39507
vertex .0577736 .0540191 -.393521
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0577736 .0532445 -.395845
vertex .0577736 .0540191 -.393521
vertex .0560416 .0524699 -.394296
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0560416 .0547937 -.39507
vertex .0560416 .0524699 -.394296
vertex .0577736 .0540191 -.393521
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.128227 .0457825 -.359327
vertex -.129959 .0450079 -.357777
vertex -.129959 .0473317 -.358552
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.128227 .0457825 -.359327
vertex -.129959 .0473317 -.358552
vertex -.128227 .0465571 -.357003
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.128227 .0457825 -.359327
vertex -.128227 .0465571 -.357003
vertex -.129959 .0450079 -.357777
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.129959 .0473317 -.358552
vertex -.129959 .0450079 -.357777
vertex -.128227 .0465571 -.357003
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.106341 .0453763 -.356176
vertex -.108073 .0446017 -.354626
vertex -.108073 .0469255 -.355401
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.106341 .0453763 -.356176
vertex -.108073 .0469255 -.355401
vertex -.106341 .0461509 -.353852
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.106341 .0453763 -.356176
vertex -.106341 .0461509 -.353852
vertex -.108073 .0446017 -.354626
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.108073 .0469255 -.355401
vertex -.108073 .0446017 -.354626
vertex -.106341 .0461509 -.353852
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00014076 .0533646 -.418145
vertex -.00187281 .0525901 -.416596
vertex -.00187281 .0549138 -.417371
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00014076 .0533646 -.418145
vertex -.00187281 .0549138 -.417371
vertex -.00014076 .0541392 -.415822
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00014076 .0533646 -.418145
vertex -.00014076 .0541392 -.415822
vertex -.00187281 .0525901 -.416596
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00187281 .0549138 -.417371
vertex -.00187281 .0525901 -.416596
vertex -.00014076 .0541392 -.415822
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0265474 .0518753 -.406592
vertex .0248154 .0511007 -.405043
vertex .0248154 .0534245 -.405817
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0265474 .0518753 -.406592
vertex .0248154 .0534245 -.405817
vertex .0265474 .0526499 -.404268
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0265474 .0518753 -.406592
vertex .0265474 .0526499 -.404268
vertex .0248154 .0511007 -.405043
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0248154 .0534245 -.405817
vertex .0248154 .0511007 -.405043
vertex .0265474 .0526499 -.404268
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0493499 .0509275 -.399239
vertex .0476179 .0501529 -.39769
vertex .0476179 .0524767 -.398465
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0493499 .0509275 -.399239
vertex .0476179 .0524767 -.398465
vertex .0493499 .0517021 -.396916
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0493499 .0509275 -.399239
vertex .0493499 .0517021 -.396916
vertex .0476179 .0501529 -.39769
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0476179 .0524767 -.398465
vertex .0476179 .0501529 -.39769
vertex .0493499 .0517021 -.396916
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0711036 .0498444 -.390837
vertex .0693716 .0490698 -.389287
vertex .0693716 .0513936 -.390062
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0711036 .0498444 -.390837
vertex .0693716 .0513936 -.390062
vertex .0711036 .050619 -.388513
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0711036 .0498444 -.390837
vertex .0711036 .050619 -.388513
vertex .0693716 .0490698 -.389287
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0693716 .0513936 -.390062
vertex .0693716 .0490698 -.389287
vertex .0711036 .050619 -.388513
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.114462 .0431848 -.358389
vertex -.116195 .0424102 -.35684
vertex -.116195 .044734 -.357615
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.114462 .0431848 -.358389
vertex -.116195 .044734 -.357615
vertex -.114462 .0439594 -.356066
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.114462 .0431848 -.358389
vertex -.114462 .0439594 -.356066
vertex -.116195 .0424102 -.35684
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.116195 .044734 -.357615
vertex -.116195 .0424102 -.35684
vertex -.114462 .0439594 -.356066
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00930632 .0507682 -.420562
vertex -.0110384 .0499936 -.419013
vertex -.0110384 .0523173 -.419787
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00930632 .0507682 -.420562
vertex -.0110384 .0523173 -.419787
vertex -.00930632 .0515427 -.418238
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00930632 .0507682 -.420562
vertex -.00930632 .0515427 -.418238
vertex -.0110384 .0499936 -.419013
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0110384 .0523173 -.419787
vertex -.0110384 .0499936 -.419013
vertex -.00930632 .0515427 -.418238
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0148417 .0497399 -.412132
vertex .0131097 .0489653 -.410582
vertex .0131097 .0512891 -.411357
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0148417 .0497399 -.412132
vertex .0131097 .0512891 -.411357
vertex .0148417 .0505145 -.409808
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0148417 .0497399 -.412132
vertex .0148417 .0505145 -.409808
vertex .0131097 .0489653 -.410582
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0131097 .0512891 -.411357
vertex .0131097 .0489653 -.410582
vertex .0148417 .0505145 -.409808
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0379103 .0485831 -.402648
vertex .0361782 .0478085 -.401098
vertex .0361782 .0501323 -.401873
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0379103 .0485831 -.402648
vertex .0361782 .0501323 -.401873
vertex .0379103 .0493577 -.400324
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0379103 .0485831 -.402648
vertex .0379103 .0493577 -.400324
vertex .0361782 .0478085 -.401098
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0361782 .0501323 -.401873
vertex .0361782 .0478085 -.401098
vertex .0379103 .0493577 -.400324
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0627285 .0474263 -.393164
vertex .0609965 .0466517 -.391615
vertex .0609965 .0489755 -.392389
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0627285 .0474263 -.393164
vertex .0609965 .0489755 -.392389
vertex .0627285 .0482009 -.39084
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0627285 .0474263 -.393164
vertex .0627285 .0482009 -.39084
vertex .0609965 .0466517 -.391615
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0609965 .0489755 -.392389
vertex .0609965 .0466517 -.391615
vertex .0627285 .0482009 -.39084
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.120021 .0409796 -.36061
vertex -.121753 .040205 -.359061
vertex -.121753 .0425288 -.359835
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.120021 .0409796 -.36061
vertex -.121753 .0425288 -.359835
vertex -.120021 .0417542 -.358286
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.120021 .0409796 -.36061
vertex -.120021 .0417542 -.358286
vertex -.121753 .040205 -.359061
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.121753 .0425288 -.359835
vertex -.121753 .040205 -.359061
vertex -.120021 .0417542 -.358286
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.0123274 .0479146 -.42087
vertex -.0140595 .04714 -.419321
vertex -.0140595 .0494638 -.420096
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.0123274 .0479146 -.42087
vertex -.0140595 .0494638 -.420096
vertex -.0123274 .0486892 -.418547
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.0123274 .0479146 -.42087
vertex -.0123274 .0486892 -.418547
vertex -.0140595 .04714 -.419321
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0140595 .0494638 -.420096
vertex -.0140595 .04714 -.419321
vertex -.0123274 .0486892 -.418547
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0118206 .0469413 -.412413
vertex .0100886 .0461667 -.410864
vertex .0100886 .0484905 -.411638
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0118206 .0469413 -.412413
vertex .0100886 .0484905 -.411638
vertex .0118206 .0477159 -.410089
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0118206 .0469413 -.412413
vertex .0118206 .0477159 -.410089
vertex .0100886 .0461667 -.410864
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0100886 .0484905 -.411638
vertex .0100886 .0461667 -.410864
vertex .0118206 .0477159 -.410089
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0348968 .0458463 -.402898
vertex .0331648 .0450717 -.401349
vertex .0331648 .0473954 -.402123
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0348968 .0458463 -.402898
vertex .0331648 .0473954 -.402123
vertex .0348968 .0466209 -.400574
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0348968 .0458463 -.402898
vertex .0348968 .0466209 -.400574
vertex .0331648 .0450717 -.401349
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0331648 .0473954 -.402123
vertex .0331648 .0450717 -.401349
vertex .0348968 .0466209 -.400574
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0571699 .0449946 -.395498
vertex .0554378 .04422 -.393948
vertex .0554378 .0465438 -.394723
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0571699 .0449946 -.395498
vertex .0554378 .0465438 -.394723
vertex .0571699 .0457692 -.393174
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0571699 .0449946 -.395498
vertex .0571699 .0457692 -.393174
vertex .0554378 .04422 -.393948
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0554378 .0465438 -.394723
vertex .0554378 .04422 -.393948
vertex .0571699 .0457692 -.393174
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.120377 .0387606 -.362838
vertex -.122109 .037986 -.361288
vertex -.122109 .0403098 -.362063
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.120377 .0387606 -.362838
vertex -.122109 .0403098 -.362063
vertex -.120377 .0395352 -.360514
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.120377 .0387606 -.362838
vertex -.120377 .0395352 -.360514
vertex -.122109 .037986 -.361288
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.122109 .0403098 -.362063
vertex -.122109 .037986 -.361288
vertex -.120377 .0395352 -.360514
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00922965 .04496 -.420112
vertex -.0109617 .0441854 -.418563
vertex -.0109617 .0465092 -.419337
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00922965 .04496 -.420112
vertex -.0109617 .0465092 -.419337
vertex -.00922965 .0457346 -.417788
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00922965 .04496 -.420112
vertex -.00922965 .0457346 -.417788
vertex -.0109617 .0441854 -.418563
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0109617 .0465092 -.419337
vertex -.0109617 .0441854 -.418563
vertex -.00922965 .0457346 -.417788
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0147342 .0440415 -.411627
vertex .0130022 .043267 -.410077
vertex .0130022 .0455907 -.410852
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0147342 .0440415 -.411627
vertex .0130022 .0455907 -.410852
vertex .0147342 .0448161 -.409303
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0147342 .0440415 -.411627
vertex .0147342 .0448161 -.409303
vertex .0130022 .043267 -.410077
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0130022 .0455907 -.410852
vertex .0130022 .043267 -.410077
vertex .0147342 .0448161 -.409303
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0376186 .0430083 -.402081
vertex .0358865 .0422337 -.400532
vertex .0358865 .0445575 -.401306
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0376186 .0430083 -.402081
vertex .0358865 .0445575 -.401306
vertex .0376186 .0437829 -.399757
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0376186 .0430083 -.402081
vertex .0376186 .0437829 -.399757
vertex .0358865 .0422337 -.400532
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0358865 .0445575 -.401306
vertex .0358865 .0422337 -.400532
vertex .0376186 .0437829 -.399757
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0598737 .0423195 -.395717
vertex .0581417 .0415449 -.394168
vertex .0581417 .0438687 -.394943
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0598737 .0423195 -.395717
vertex .0581417 .0438687 -.394943
vertex .0598737 .0430941 -.393393
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0598737 .0423195 -.395717
vertex .0598737 .0430941 -.393393
vertex .0581417 .0415449 -.394168
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0581417 .0438687 -.394943
vertex .0581417 .0415449 -.394168
vertex .0598737 .0430941 -.393393
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.109909 .03642 -.364008
vertex -.111641 .0356454 -.362459
vertex -.111641 .0379691 -.363233
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.109909 .03642 -.364008
vertex -.111641 .0379691 -.363233
vertex -.109909 .0371946 -.361684
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.109909 .03642 -.364008
vertex -.109909 .0371946 -.361684
vertex -.111641 .0356454 -.362459
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.111641 .0379691 -.363233
vertex -.111641 .0356454 -.362459
vertex -.109909 .0371946 -.361684
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00616256 .0420328 -.419339
vertex -.00789461 .0412582 -.41779
vertex -.00789461 .043582 -.418565
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00616256 .0420328 -.419339
vertex -.00789461 .043582 -.418565
vertex -.00616256 .0428074 -.417015
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00616256 .0420328 -.419339
vertex -.00616256 .0428074 -.417015
vertex -.00789461 .0412582 -.41779
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00789461 .043582 -.418565
vertex -.00789461 .0412582 -.41779
vertex -.00616256 .0428074 -.417015
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0176171 .0411693 -.410827
vertex .0158851 .0403947 -.409278
vertex .0158851 .0427185 -.410052
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0176171 .0411693 -.410827
vertex .0158851 .0427185 -.410052
vertex .0176171 .0419439 -.408503
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0176171 .0411693 -.410827
vertex .0176171 .0419439 -.408503
vertex .0158851 .0403947 -.409278
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0158851 .0427185 -.410052
vertex .0158851 .0403947 -.409278
vertex .0176171 .0419439 -.408503
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0431951 .0401978 -.40125
vertex .0414631 .0394232 -.399701
vertex .0414631 .041747 -.400476
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0431951 .0401978 -.40125
vertex .0414631 .041747 -.400476
vertex .0431951 .0409724 -.398926
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0431951 .0401978 -.40125
vertex .0431951 .0409724 -.398926
vertex .0414631 .0394232 -.399701
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0414631 .041747 -.400476
vertex .0414631 .0394232 -.399701
vertex .0431951 .0409724 -.398926
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0648875 .0393343 -.392738
vertex .0631554 .0385597 -.391188
vertex .0631554 .0408835 -.391963
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0648875 .0393343 -.392738
vertex .0631554 .0408835 -.391963
vertex .0648875 .0401089 -.390414
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0648875 .0393343 -.392738
vertex .0648875 .0401089 -.390414
vertex .0631554 .0385597 -.391188
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0631554 .0408835 -.391963
vertex .0631554 .0385597 -.391188
vertex .0648875 .0401089 -.390414
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.024578 .0398406 -.426026
vertex -.0263101 .039066 -.424476
vertex -.0263101 .0413898 -.425251
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.024578 .0398406 -.426026
vertex -.0263101 .0413898 -.425251
vertex -.024578 .0406152 -.423702
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.024578 .0398406 -.426026
vertex -.024578 .0406152 -.423702
vertex -.0263101 .039066 -.424476
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0263101 .0413898 -.425251
vertex -.0263101 .039066 -.424476
vertex -.024578 .0406152 -.423702
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.000120472 .0389309 -.416418
vertex -.00185252 .0381563 -.414869
vertex -.00185252 .0404801 -.415643
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.000120472 .0389309 -.416418
vertex -.00185252 .0404801 -.415643
vertex -.000120472 .0397055 -.414094
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.000120472 .0389309 -.416418
vertex -.000120472 .0397055 -.414094
vertex -.00185252 .0381563 -.414869
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.00185252 .0404801 -.415643
vertex -.00185252 .0381563 -.414869
vertex -.000120472 .0397055 -.414094
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0232909 .0381223 -.407878
vertex .0215588 .0373477 -.406329
vertex .0215588 .0396715 -.407103
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0232909 .0381223 -.407878
vertex .0215588 .0396715 -.407103
vertex .0232909 .0388969 -.405554
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0232909 .0381223 -.407878
vertex .0232909 .0388969 -.405554
vertex .0215588 .0373477 -.406329
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0215588 .0396715 -.407103
vertex .0215588 .0373477 -.406329
vertex .0232909 .0388969 -.405554
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0458402 .0374148 -.400406
vertex .0441081 .0366402 -.398856
vertex .0441081 .038964 -.399631
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0458402 .0374148 -.400406
vertex .0441081 .038964 -.399631
vertex .0458402 .0381894 -.398082
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0458402 .0374148 -.400406
vertex .0458402 .0381894 -.398082
vertex .0441081 .0366402 -.398856
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0441081 .038964 -.399631
vertex .0441081 .0366402 -.398856
vertex .0458402 .0381894 -.398082
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0673483 .0366062 -.391866
vertex .0656162 .0358316 -.390316
vertex .0656162 .0381554 -.391091
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0673483 .0366062 -.391866
vertex .0656162 .0381554 -.391091
vertex .0673483 .0373808 -.389542
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0673483 .0366062 -.391866
vertex .0673483 .0373808 -.389542
vertex .0656162 .0358316 -.390316
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0656162 .0381554 -.391091
vertex .0656162 .0358316 -.390316
vertex .0673483 .0373808 -.389542
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex -.00310575 .0362608 -.417753
vertex -.0048378 .0354862 -.416204
vertex -.0048378 .03781 -.416978
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex -.00310575 .0362608 -.417753
vertex -.0048378 .03781 -.416978
vertex -.00310575 .0370354 -.415429
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex -.00310575 .0362608 -.417753
vertex -.00310575 .0370354 -.415429
vertex -.0048378 .0354862 -.416204
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex -.0048378 .03781 -.416978
vertex -.0048378 .0354862 -.416204
vertex -.00310575 .0370354 -.415429
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .020367 .0355071 -.409186
vertex .0186349 .0347325 -.407636
vertex .0186349 .0370563 -.408411
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .020367 .0355071 -.409186
vertex .0186349 .0370563 -.408411
vertex .020367 .0362817 -.406862
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .020367 .0355071 -.409186
vertex .020367 .0362817 -.406862
vertex .0186349 .0347325 -.407636
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0186349 .0370563 -.408411
vertex .0186349 .0347325 -.407636
vertex .020367 .0362817 -.406862
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0428574 .0347534 -.400618
vertex .0411254 .0339789 -.399069
vertex .0411254 .0363026 -.399844
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0428574 .0347534 -.400618
vertex .0411254 .0363026 -.399844
vertex .0428574 .035528 -.398294
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0428574 .0347534 -.400618
vertex .0428574 .035528 -.398294
vertex .0411254 .0339789 -.399069
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0411254 .0363026 -.399844
vertex .0411254 .0339789 -.399069
vertex .0428574 .035528 -.398294
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0645395 .034094 -.393122
vertex .0628075 .0333194 -.391573
vertex .0628075 .0356432 -.392347
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0645395 .034094 -.393122
vertex .0628075 .0356432 -.392347
vertex .0645395 .0348686 -.390798
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0645395 .034094 -.393122
vertex .0645395 .0348686 -.390798
vertex .0628075 .0333194 -.391573
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0628075 .0356432 -.392347
vertex .0628075 .0333194 -.391573
vertex .0645395 .0348686 -.390798
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .00873292 .033154 -.413716
vertex .00700087 .0323794 -.412167
vertex .00700087 .0347032 -.412942
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .00873292 .033154 -.413716
vertex .00700087 .0347032 -.412942
vertex .00873292 .0339286 -.411393
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .00873292 .033154 -.413716
vertex .00873292 .0339286 -.411393
vertex .00700087 .0323794 -.412167
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .00700087 .0347032 -.412942
vertex .00700087 .0323794 -.412167
vertex .00873292 .0339286 -.411393
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0316531 .0324552 -.405121
vertex .0299211 .0316806 -.403572
vertex .0299211 .0340044 -.404347
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0316531 .0324552 -.405121
vertex .0299211 .0340044 -.404347
vertex .0316531 .0332298 -.402798
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0316531 .0324552 -.405121
vertex .0316531 .0332298 -.402798
vertex .0299211 .0316806 -.403572
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0299211 .0340044 -.404347
vertex .0299211 .0316806 -.403572
vertex .0316531 .0332298 -.402798
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0562642 .0316691 -.395452
vertex .0545321 .0308945 -.393903
vertex .0545321 .0332183 -.394678
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0562642 .0316691 -.395452
vertex .0545321 .0332183 -.394678
vertex .0562642 .0324437 -.393128
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0562642 .0316691 -.395452
vertex .0562642 .0324437 -.393128
vertex .0545321 .0308945 -.393903
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0545321 .0332183 -.394678
vertex .0545321 .0308945 -.393903
vertex .0562642 .0324437 -.393128
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0202134 .0300353 -.408567
vertex .0184814 .0292607 -.407018
vertex .0184814 .0315845 -.407793
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0202134 .0300353 -.408567
vertex .0184814 .0315845 -.407793
vertex .0202134 .0308099 -.406244
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0202134 .0300353 -.408567
vertex .0202134 .0308099 -.406244
vertex .0184814 .0292607 -.407018
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0184814 .0315845 -.407793
vertex .0184814 .0292607 -.407018
vertex .0202134 .0308099 -.406244
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0426323 .0294719 -.401023
vertex .0409002 .0286973 -.399474
vertex .0409002 .0310211 -.400248
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0426323 .0294719 -.401023
vertex .0409002 .0310211 -.400248
vertex .0426323 .0302465 -.398699
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0426323 .0294719 -.401023
vertex .0426323 .0302465 -.398699
vertex .0409002 .0286973 -.399474
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0409002 .0310211 -.400248
vertex .0409002 .0286973 -.399474
vertex .0426323 .0302465 -.398699
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0641916 .0289085 -.393478
vertex .0624595 .0281339 -.391929
vertex .0624595 .0304577 -.392704
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0641916 .0289085 -.393478
vertex .0624595 .0304577 -.392704
vertex .0641916 .0296831 -.391155
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0641916 .0289085 -.393478
vertex .0641916 .0296831 -.391155
vertex .0624595 .0281339 -.391929
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0624595 .0304577 -.392704
vertex .0624595 .0281339 -.391929
vertex .0641916 .0296831 -.391155
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0451186 .0267045 -.399053
vertex .0433866 .0259299 -.397503
vertex .0433866 .0282537 -.398278
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0451186 .0267045 -.399053
vertex .0433866 .0282537 -.398278
vertex .0451186 .0274791 -.396729
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0451186 .0267045 -.399053
vertex .0451186 .0274791 -.396729
vertex .0433866 .0259299 -.397503
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0433866 .0282537 -.398278
vertex .0433866 .0259299 -.397503
vertex .0451186 .0274791 -.396729
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0664401 .0261892 -.391484
vertex .064708 .0254146 -.389935
vertex .064708 .0277383 -.390709
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0664401 .0261892 -.391484
vertex .064708 .0277383 -.390709
vertex .0664401 .0269638 -.38916
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0664401 .0261892 -.391484
vertex .0664401 .0269638 -.38916
vertex .064708 .0254146 -.389935
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .064708 .0277383 -.390709
vertex .064708 .0254146 -.389935
vertex .0664401 .0269638 -.38916
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0584461 .0238447 -.394892
vertex .0567141 .0230701 -.393343
vertex .0567141 .0253939 -.394118
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0584461 .0238447 -.394892
vertex .0567141 .0253939 -.394118
vertex .0584461 .0246193 -.392568
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0584461 .0238447 -.394892
vertex .0584461 .0246193 -.392568
vertex .0567141 .0230701 -.393343
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0567141 .0253939 -.394118
vertex .0567141 .0230701 -.393343
vertex .0584461 .0246193 -.392568
endloop
endfacet
facet normal -.57735 -.258199 -.774597
outer loop
vertex .0739326 .0209938 -.38961
vertex .0722005 .0202192 -.38806
vertex .0722005 .0225429 -.388835
endloop
endfacet
facet normal .57735 .774597 -.258199
outer loop
vertex .0739326 .0209938 -.38961
vertex .0722005 .0225429 -.388835
vertex .0739326 .0217683 -.387286
endloop
endfacet
facet normal .57735 -.774597 .258199
outer loop
vertex .0739326 .0209938 -.38961
vertex .0739326 .0217683 -.387286
vertex .0722005 .0202192 -.38806
endloop
endfacet
facet normal -.57735 .258199 .774597
outer loop
vertex .0722005 .0225429 -.388835
vertex .0722005 .0202192 -.38806
vertex .0739326 .0217683 -.387286
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0559787 .312525 -.372926
vertex .0559342 .313171 -.373552
vertex .0560222 .312551 -.374198
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0559787 .312525 -.372926
vertex .0560222 .312551 -.374198
vertex .0560668 .311905 -.373572
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.194586 .286419 -.382018
vertex -.194542 .286444 -.383289
vertex -.19463 .287064 -.382643
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.194586 .286419 -.382018
vertex -.194498 .285799 -.382664
vertex -.194542 .286444 -.383289
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0559787 .312525 -.372926
vertex -.194586 .286419 -.382018
vertex -.19463 .287064 -.382643
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0559787 .312525 -.372926
vertex -.19463 .287064 -.382643
vertex .0559342 .313171 -.373552
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0559342 .313171 -.373552
vertex -.19463 .287064 -.382643
vertex -.194542 .286444 -.383289
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0559342 .313171 -.373552
vertex -.194542 .286444 -.383289
vertex .0560222 .312551 -.374198
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0560222 .312551 -.374198
vertex -.194542 .286444 -.383289
vertex -.194498 .285799 -.382664
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0560222 .312551 -.374198
vertex -.194498 .285799 -.382664
vertex .0560668 .311905 -.373572
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0560668 .311905 -.373572
vertex -.194498 .285799 -.382664
vertex -.194586 .286419 -.382018
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0560668 .311905 -.373572
vertex -.194586 .286419 -.382018
vertex .0559787 .312525 -.372926
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.194139 .286801 -.38295
vertex -.194094 .286156 -.382325
vertex -.194989 .286062 -.382357
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.194139 .286801 -.38295
vertex -.194989 .286062 -.382357
vertex -.195034 .286708 -.382983
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.173942 .144551 -.531119
vertex -.174792 .143812 -.530526
vertex -.173897 .143905 -.530493
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.173942 .144551 -.531119
vertex -.174836 .144457 -.531152
vertex -.174792 .143812 -.530526
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.194139 .286801 -.38295
vertex -.173942 .144551 -.531119
vertex -.173897 .143905 -.530493
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.194139 .286801 -.38295
vertex -.173897 .143905 -.530493
vertex -.194094 .286156 -.382325
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.194094 .286156 -.382325
vertex -.173897 .143905 -.530493
vertex -.174792 .143812 -.530526
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.194094 .286156 -.382325
vertex -.174792 .143812 -.530526
vertex -.194989 .286062 -.382357
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.194989 .286062 -.382357
vertex -.174792 .143812 -.530526
vertex -.174836 .144457 -.531152
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.194989 .286062 -.382357
vertex -.174836 .144457 -.531152
vertex -.195034 .286708 -.382983
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.195034 .286708 -.382983
vertex -.174836 .144457 -.531152
vertex -.173942 .144551 -.531119
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.195034 .286708 -.382983
vertex -.173942 .144551 -.531119
vertex -.194139 .286801 -.38295
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.174433 .144814 -.530812
vertex -.174388 .144169 -.530187
vertex -.1743 .143548 -.530833
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.174433 .144814 -.530812
vertex -.1743 .143548 -.530833
vertex -.174345 .144194 -.531458
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0761316 .17092 -.521721
vertex .0762642 .169655 -.521741
vertex .0761761 .170275 -.521095
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0761316 .17092 -.521721
vertex .0762196 .1703 -.522367
vertex .0762642 .169655 -.521741
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.174433 .144814 -.530812
vertex .0761316 .17092 -.521721
vertex .0761761 .170275 -.521095
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.174433 .144814 -.530812
vertex .0761761 .170275 -.521095
vertex -.174388 .144169 -.530187
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.174388 .144169 -.530187
vertex .0761761 .170275 -.521095
vertex .0762642 .169655 -.521741
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.174388 .144169 -.530187
vertex .0762642 .169655 -.521741
vertex -.1743 .143548 -.530833
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.1743 .143548 -.530833
vertex .0762642 .169655 -.521741
vertex .0762196 .1703 -.522367
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.1743 .143548 -.530833
vertex .0762196 .1703 -.522367
vertex -.174345 .144194 -.531458
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.174345 .144194 -.531458
vertex .0762196 .1703 -.522367
vertex .0761316 .17092 -.521721
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.174345 .144194 -.531458
vertex .0761316 .17092 -.521721
vertex -.174433 .144814 -.530812
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0766674 .170012 -.521402
vertex .0766229 .170657 -.522027
vertex .0757283 .170564 -.52206
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0766674 .170012 -.521402
vertex .0757283 .170564 -.52206
vertex .0757728 .169918 -.521434
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .05647 .312262 -.373233
vertex .0555309 .312814 -.373891
vertex .0564255 .312907 -.373858
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .05647 .312262 -.373233
vertex .0555755 .312169 -.373265
vertex .0555309 .312814 -.373891
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0766674 .170012 -.521402
vertex .05647 .312262 -.373233
vertex .0564255 .312907 -.373858
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0766674 .170012 -.521402
vertex .0564255 .312907 -.373858
vertex .0766229 .170657 -.522027
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0766229 .170657 -.522027
vertex .0564255 .312907 -.373858
vertex .0555309 .312814 -.373891
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0766229 .170657 -.522027
vertex .0555309 .312814 -.373891
vertex .0757283 .170564 -.52206
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex .0757283 .170564 -.52206
vertex .0555309 .312814 -.373891
vertex .0555755 .312169 -.373265
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex .0757283 .170564 -.52206
vertex .0555755 .312169 -.373265
vertex .0757728 .169918 -.521434
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0757728 .169918 -.521434
vertex .0555755 .312169 -.373265
vertex .05647 .312262 -.373233
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0757728 .169918 -.521434
vertex .05647 .312262 -.373233
vertex .0766674 .170012 -.521402
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0666469 .157944 -.223065
vertex .0666023 .15859 -.223691
vertex .0666904 .157969 -.224337
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0666469 .157944 -.223065
vertex .0666904 .157969 -.224337
vertex .0667349 .157324 -.223711
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.183918 .131838 -.232157
vertex -.183874 .131863 -.233429
vertex -.183962 .132483 -.232783
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.183918 .131838 -.232157
vertex -.18383 .131218 -.232803
vertex -.183874 .131863 -.233429
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0666469 .157944 -.223065
vertex -.183918 .131838 -.232157
vertex -.183962 .132483 -.232783
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0666469 .157944 -.223065
vertex -.183962 .132483 -.232783
vertex .0666023 .15859 -.223691
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0666023 .15859 -.223691
vertex -.183962 .132483 -.232783
vertex -.183874 .131863 -.233429
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0666023 .15859 -.223691
vertex -.183874 .131863 -.233429
vertex .0666904 .157969 -.224337
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0666904 .157969 -.224337
vertex -.183874 .131863 -.233429
vertex -.18383 .131218 -.232803
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0666904 .157969 -.224337
vertex -.18383 .131218 -.232803
vertex .0667349 .157324 -.223711
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0667349 .157324 -.223711
vertex -.18383 .131218 -.232803
vertex -.183918 .131838 -.232157
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0667349 .157324 -.223711
vertex -.183918 .131838 -.232157
vertex .0666469 .157944 -.223065
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.183471 .13222 -.233089
vertex -.183426 .131574 -.232464
vertex -.184321 .131481 -.232496
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.183471 .13222 -.233089
vertex -.184321 .131481 -.232496
vertex -.184365 .132127 -.233122
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.163273 -.0100306 -.381258
vertex -.164123 -.0107692 -.380665
vertex -.163229 -.010676 -.380633
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.163273 -.0100306 -.381258
vertex -.164168 -.0101238 -.381291
vertex -.164123 -.0107692 -.380665
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.183471 .13222 -.233089
vertex -.163273 -.0100306 -.381258
vertex -.163229 -.010676 -.380633
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.183471 .13222 -.233089
vertex -.163229 -.010676 -.380633
vertex -.183426 .131574 -.232464
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.183426 .131574 -.232464
vertex -.163229 -.010676 -.380633
vertex -.164123 -.0107692 -.380665
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.183426 .131574 -.232464
vertex -.164123 -.0107692 -.380665
vertex -.184321 .131481 -.232496
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.184321 .131481 -.232496
vertex -.164123 -.0107692 -.380665
vertex -.164168 -.0101238 -.381291
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.184321 .131481 -.232496
vertex -.164168 -.0101238 -.381291
vertex -.184365 .132127 -.233122
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.184365 .132127 -.233122
vertex -.164168 -.0101238 -.381291
vertex -.163273 -.0100306 -.381258
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.184365 .132127 -.233122
vertex -.163273 -.0100306 -.381258
vertex -.183471 .13222 -.233089
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.163765 -.00976707 -.380952
vertex -.16372 -.0104125 -.380326
vertex -.163632 -.0110328 -.380972
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.163765 -.00976707 -.380952
vertex -.163632 -.0110328 -.380972
vertex -.163677 -.0103874 -.381598
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0867997 .0163393 -.37186
vertex .0869323 .0150736 -.37188
vertex .0868443 .0156939 -.371234
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0867997 .0163393 -.37186
vertex .0868878 .015719 -.372506
vertex .0869323 .0150736 -.37188
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.163765 -.00976707 -.380952
vertex .0867997 .0163393 -.37186
vertex .0868443 .0156939 -.371234
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.163765 -.00976707 -.380952
vertex .0868443 .0156939 -.371234
vertex -.16372 -.0104125 -.380326
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.16372 -.0104125 -.380326
vertex .0868443 .0156939 -.371234
vertex .0869323 .0150736 -.37188
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.16372 -.0104125 -.380326
vertex .0869323 .0150736 -.37188
vertex -.163632 -.0110328 -.380972
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.163632 -.0110328 -.380972
vertex .0869323 .0150736 -.37188
vertex .0868878 .015719 -.372506
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.163632 -.0110328 -.380972
vertex .0868878 .015719 -.372506
vertex -.163677 -.0103874 -.381598
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.163677 -.0103874 -.381598
vertex .0868878 .015719 -.372506
vertex .0867997 .0163393 -.37186
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.163677 -.0103874 -.381598
vertex .0867997 .0163393 -.37186
vertex -.163765 -.00976707 -.380952
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0873356 .0154303 -.371541
vertex .087291 .0160757 -.372167
vertex .0863965 .0159825 -.372199
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0873356 .0154303 -.371541
vertex .0863965 .0159825 -.372199
vertex .086441 .0153371 -.371573
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0671382 .157681 -.223372
vertex .0661991 .158233 -.22403
vertex .0670937 .158326 -.223998
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0671382 .157681 -.223372
vertex .0662436 .157588 -.223405
vertex .0661991 .158233 -.22403
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0873356 .0154303 -.371541
vertex .0671382 .157681 -.223372
vertex .0670937 .158326 -.223998
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0873356 .0154303 -.371541
vertex .0670937 .158326 -.223998
vertex .087291 .0160757 -.372167
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .087291 .0160757 -.372167
vertex .0670937 .158326 -.223998
vertex .0661991 .158233 -.22403
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .087291 .0160757 -.372167
vertex .0661991 .158233 -.22403
vertex .0863965 .0159825 -.372199
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex .0863965 .0159825 -.372199
vertex .0661991 .158233 -.22403
vertex .0662436 .157588 -.223405
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex .0863965 .0159825 -.372199
vertex .0662436 .157588 -.223405
vertex .086441 .0153371 -.371573
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .086441 .0153371 -.371573
vertex .0662436 .157588 -.223405
vertex .0671382 .157681 -.223372
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .086441 .0153371 -.371573
vertex .0671382 .157681 -.223372
vertex .0873356 .0154303 -.371541
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0564918 .312274 -.373869
vertex .0564037 .312895 -.373223
vertex .0555092 .312802 -.373255
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0564918 .312274 -.373869
vertex .0555092 .312802 -.373255
vertex .0555972 .312181 -.373901
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .06716 .157693 -.224008
vertex .0661773 .15822 -.223394
vertex .0670719 .158314 -.223362
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .06716 .157693 -.224008
vertex .0662654 .1576 -.22404
vertex .0661773 .15822 -.223394
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0564918 .312274 -.373869
vertex .06716 .157693 -.224008
vertex .0670719 .158314 -.223362
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0564918 .312274 -.373869
vertex .0670719 .158314 -.223362
vertex .0564037 .312895 -.373223
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0564037 .312895 -.373223
vertex .0670719 .158314 -.223362
vertex .0661773 .15822 -.223394
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0564037 .312895 -.373223
vertex .0661773 .15822 -.223394
vertex .0555092 .312802 -.373255
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex .0555092 .312802 -.373255
vertex .0661773 .15822 -.223394
vertex .0662654 .1576 -.22404
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex .0555092 .312802 -.373255
vertex .0662654 .1576 -.22404
vertex .0555972 .312181 -.373901
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0555972 .312181 -.373901
vertex .0662654 .1576 -.22404
vertex .06716 .157693 -.224008
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0555972 .312181 -.373901
vertex .06716 .157693 -.224008
vertex .0564918 .312274 -.373869
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.194073 .286168 -.38296
vertex -.194161 .286788 -.382314
vertex -.195055 .286695 -.382347
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.194073 .286168 -.38296
vertex -.195055 .286695 -.382347
vertex -.194967 .286075 -.382993
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.183405 .131587 -.2331
vertex -.184387 .132114 -.232486
vertex -.183493 .132207 -.232454
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.183405 .131587 -.2331
vertex -.184299 .131494 -.233132
vertex -.184387 .132114 -.232486
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.194073 .286168 -.38296
vertex -.183405 .131587 -.2331
vertex -.183493 .132207 -.232454
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.194073 .286168 -.38296
vertex -.183493 .132207 -.232454
vertex -.194161 .286788 -.382314
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.194161 .286788 -.382314
vertex -.183493 .132207 -.232454
vertex -.184387 .132114 -.232486
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.194161 .286788 -.382314
vertex -.184387 .132114 -.232486
vertex -.195055 .286695 -.382347
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.195055 .286695 -.382347
vertex -.184387 .132114 -.232486
vertex -.184299 .131494 -.233132
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.195055 .286695 -.382347
vertex -.184299 .131494 -.233132
vertex -.194967 .286075 -.382993
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.194967 .286075 -.382993
vertex -.184299 .131494 -.233132
vertex -.183405 .131587 -.2331
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.194967 .286075 -.382993
vertex -.183405 .131587 -.2331
vertex -.194073 .286168 -.38296
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0766892 .170024 -.522038
vertex .0766011 .170644 -.521392
vertex .0757065 .170551 -.521424
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0766892 .170024 -.522038
vertex .0757065 .170551 -.521424
vertex .0757946 .169931 -.52207
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0873573 .0154429 -.372177
vertex .0863747 .01597 -.371563
vertex .0872693 .0160632 -.371531
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0873573 .0154429 -.372177
vertex .0864628 .0153497 -.372209
vertex .0863747 .01597 -.371563
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0766892 .170024 -.522038
vertex .0873573 .0154429 -.372177
vertex .0872693 .0160632 -.371531
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0766892 .170024 -.522038
vertex .0872693 .0160632 -.371531
vertex .0766011 .170644 -.521392
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0766011 .170644 -.521392
vertex .0872693 .0160632 -.371531
vertex .0863747 .01597 -.371563
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0766011 .170644 -.521392
vertex .0863747 .01597 -.371563
vertex .0757065 .170551 -.521424
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex .0757065 .170551 -.521424
vertex .0863747 .01597 -.371563
vertex .0864628 .0153497 -.372209
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex .0757065 .170551 -.521424
vertex .0864628 .0153497 -.372209
vertex .0757946 .169931 -.52207
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0757946 .169931 -.52207
vertex .0864628 .0153497 -.372209
vertex .0873573 .0154429 -.372177
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0757946 .169931 -.52207
vertex .0873573 .0154429 -.372177
vertex .0766892 .170024 -.522038
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.173875 .143918 -.531129
vertex -.173963 .144538 -.530483
vertex -.174858 .144445 -.530516
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.173875 .143918 -.531129
vertex -.174858 .144445 -.530516
vertex -.17477 .143824 -.531162
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.163207 -.0106635 -.381269
vertex -.16419 -.0101364 -.380655
vertex -.163295 -.0100432 -.380623
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.163207 -.0106635 -.381269
vertex -.164102 -.0107567 -.381301
vertex -.16419 -.0101364 -.380655
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.173875 .143918 -.531129
vertex -.163207 -.0106635 -.381269
vertex -.163295 -.0100432 -.380623
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.173875 .143918 -.531129
vertex -.163295 -.0100432 -.380623
vertex -.173963 .144538 -.530483
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.173963 .144538 -.530483
vertex -.163295 -.0100432 -.380623
vertex -.16419 -.0101364 -.380655
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.173963 .144538 -.530483
vertex -.16419 -.0101364 -.380655
vertex -.174858 .144445 -.530516
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.174858 .144445 -.530516
vertex -.16419 -.0101364 -.380655
vertex -.164102 -.0107567 -.381301
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.174858 .144445 -.530516
vertex -.164102 -.0107567 -.381301
vertex -.17477 .143824 -.531162
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.17477 .143824 -.531162
vertex -.164102 -.0107567 -.381301
vertex -.163207 -.0106635 -.381269
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.17477 .143824 -.531162
vertex -.163207 -.0106635 -.381269
vertex -.173875 .143918 -.531129
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0169808 .153105 -.225816
vertex .0170253 .152459 -.22519
vertex .0161307 .152366 -.225223
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0169808 .153105 -.225816
vertex .0161307 .152366 -.225223
vertex .0160862 .153012 -.225849
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0371781 .0108545 -.373985
vertex .0363281 .0101159 -.373392
vertex .0372227 .0102091 -.373359
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0371781 .0108545 -.373985
vertex .0362836 .0107613 -.374017
vertex .0363281 .0101159 -.373392
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0169808 .153105 -.225816
vertex .0371781 .0108545 -.373985
vertex .0372227 .0102091 -.373359
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0169808 .153105 -.225816
vertex .0372227 .0102091 -.373359
vertex .0170253 .152459 -.22519
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0170253 .152459 -.22519
vertex .0372227 .0102091 -.373359
vertex .0363281 .0101159 -.373392
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0170253 .152459 -.22519
vertex .0363281 .0101159 -.373392
vertex .0161307 .152366 -.225223
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex .0161307 .152366 -.225223
vertex .0363281 .0101159 -.373392
vertex .0362836 .0107613 -.374017
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex .0161307 .152366 -.225223
vertex .0362836 .0107613 -.374017
vertex .0160862 .153012 -.225849
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0160862 .153012 -.225849
vertex .0362836 .0107613 -.374017
vertex .0371781 .0108545 -.373985
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0160862 .153012 -.225849
vertex .0371781 .0108545 -.373985
vertex .0169808 .153105 -.225816
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0706863 .129494 -.252699
vertex .0706418 .13014 -.253325
vertex .0707299 .129519 -.253971
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0706863 .129494 -.252699
vertex .0707299 .129519 -.253971
vertex .0707744 .128874 -.253345
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.179878 .103388 -.261791
vertex -.179835 .103413 -.263063
vertex -.179923 .104033 -.262416
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.179878 .103388 -.261791
vertex -.17979 .102768 -.262437
vertex -.179835 .103413 -.263063
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0706863 .129494 -.252699
vertex -.179878 .103388 -.261791
vertex -.179923 .104033 -.262416
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0706863 .129494 -.252699
vertex -.179923 .104033 -.262416
vertex .0706418 .13014 -.253325
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0706418 .13014 -.253325
vertex -.179923 .104033 -.262416
vertex -.179835 .103413 -.263063
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0706418 .13014 -.253325
vertex -.179835 .103413 -.263063
vertex .0707299 .129519 -.253971
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0707299 .129519 -.253971
vertex -.179835 .103413 -.263063
vertex -.17979 .102768 -.262437
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0707299 .129519 -.253971
vertex -.17979 .102768 -.262437
vertex .0707744 .128874 -.253345
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0707744 .128874 -.253345
vertex -.17979 .102768 -.262437
vertex -.179878 .103388 -.261791
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0707744 .128874 -.253345
vertex -.179878 .103388 -.261791
vertex .0706863 .129494 -.252699
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.0331321 .147884 -.227634
vertex -.0330876 .147238 -.227009
vertex -.0339822 .147145 -.227041
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.0331321 .147884 -.227634
vertex -.0339822 .147145 -.227041
vertex -.0340267 .14779 -.227667
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.0129348 .00563319 -.375803
vertex -.0137848 .00489459 -.37521
vertex -.0128902 .0049878 -.375178
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.0129348 .00563319 -.375803
vertex -.0138293 .00553999 -.375836
vertex -.0137848 .00489459 -.37521
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.0331321 .147884 -.227634
vertex -.0129348 .00563319 -.375803
vertex -.0128902 .0049878 -.375178
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.0331321 .147884 -.227634
vertex -.0128902 .0049878 -.375178
vertex -.0330876 .147238 -.227009
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.0330876 .147238 -.227009
vertex -.0128902 .0049878 -.375178
vertex -.0137848 .00489459 -.37521
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.0330876 .147238 -.227009
vertex -.0137848 .00489459 -.37521
vertex -.0339822 .147145 -.227041
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.0339822 .147145 -.227041
vertex -.0137848 .00489459 -.37521
vertex -.0138293 .00553999 -.375836
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.0339822 .147145 -.227041
vertex -.0138293 .00553999 -.375836
vertex -.0340267 .14779 -.227667
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.0340267 .14779 -.227667
vertex -.0138293 .00553999 -.375836
vertex -.0129348 .00563319 -.375803
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.0340267 .14779 -.227667
vertex -.0129348 .00563319 -.375803
vertex -.0331321 .147884 -.227634
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0747258 .101044 -.282333
vertex .0746813 .10169 -.282959
vertex .0747694 .101069 -.283605
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0747258 .101044 -.282333
vertex .0747694 .101069 -.283605
vertex .0748139 .100424 -.282979
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.175839 .0749378 -.291425
vertex -.175795 .0749628 -.292696
vertex -.175883 .0755832 -.29205
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.175839 .0749378 -.291425
vertex -.175751 .0743175 -.292071
vertex -.175795 .0749628 -.292696
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0747258 .101044 -.282333
vertex -.175839 .0749378 -.291425
vertex -.175883 .0755832 -.29205
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0747258 .101044 -.282333
vertex -.175883 .0755832 -.29205
vertex .0746813 .10169 -.282959
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0746813 .10169 -.282959
vertex -.175883 .0755832 -.29205
vertex -.175795 .0749628 -.292696
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0746813 .10169 -.282959
vertex -.175795 .0749628 -.292696
vertex .0747694 .101069 -.283605
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0747694 .101069 -.283605
vertex -.175795 .0749628 -.292696
vertex -.175751 .0743175 -.292071
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0747694 .101069 -.283605
vertex -.175751 .0743175 -.292071
vertex .0748139 .100424 -.282979
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0748139 .100424 -.282979
vertex -.175751 .0743175 -.292071
vertex -.175839 .0749378 -.291425
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0748139 .100424 -.282979
vertex -.175839 .0749378 -.291425
vertex .0747258 .101044 -.282333
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.083245 .142662 -.229453
vertex -.0832005 .142017 -.228827
vertex -.0840951 .141924 -.22886
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.083245 .142662 -.229453
vertex -.0840951 .141924 -.22886
vertex -.0841396 .142569 -.229485
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.0630477 .00041192 -.377622
vertex -.0638977 -.000326679 -.377028
vertex -.0630031 -.000233474 -.376996
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.0630477 .00041192 -.377622
vertex -.0639422 .000318715 -.377654
vertex -.0638977 -.000326679 -.377028
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.083245 .142662 -.229453
vertex -.0630477 .00041192 -.377622
vertex -.0630031 -.000233474 -.376996
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.083245 .142662 -.229453
vertex -.0630031 -.000233474 -.376996
vertex -.0832005 .142017 -.228827
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.0832005 .142017 -.228827
vertex -.0630031 -.000233474 -.376996
vertex -.0638977 -.000326679 -.377028
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.0832005 .142017 -.228827
vertex -.0638977 -.000326679 -.377028
vertex -.0840951 .141924 -.22886
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.0840951 .141924 -.22886
vertex -.0638977 -.000326679 -.377028
vertex -.0639422 .000318715 -.377654
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.0840951 .141924 -.22886
vertex -.0639422 .000318715 -.377654
vertex -.0841396 .142569 -.229485
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.0841396 .142569 -.229485
vertex -.0639422 .000318715 -.377654
vertex -.0630477 .00041192 -.377622
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.0841396 .142569 -.229485
vertex -.0630477 .00041192 -.377622
vertex -.083245 .142662 -.229453
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0787653 .072594 -.311967
vertex .0787208 .0732394 -.312592
vertex .0788088 .0726191 -.313238
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0787653 .072594 -.311967
vertex .0788088 .0726191 -.313238
vertex .0788534 .0719737 -.312613
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.171799 .0464877 -.321058
vertex -.171756 .0465128 -.32233
vertex -.171844 .0471331 -.321684
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.171799 .0464877 -.321058
vertex -.171711 .0458674 -.321704
vertex -.171756 .0465128 -.32233
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0787653 .072594 -.311967
vertex -.171799 .0464877 -.321058
vertex -.171844 .0471331 -.321684
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0787653 .072594 -.311967
vertex -.171844 .0471331 -.321684
vertex .0787208 .0732394 -.312592
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0787208 .0732394 -.312592
vertex -.171844 .0471331 -.321684
vertex -.171756 .0465128 -.32233
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0787208 .0732394 -.312592
vertex -.171756 .0465128 -.32233
vertex .0788088 .0726191 -.313238
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0788088 .0726191 -.313238
vertex -.171756 .0465128 -.32233
vertex -.171711 .0458674 -.321704
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0788088 .0726191 -.313238
vertex -.171711 .0458674 -.321704
vertex .0788534 .0719737 -.312613
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0788534 .0719737 -.312613
vertex -.171711 .0458674 -.321704
vertex -.171799 .0464877 -.321058
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0788534 .0719737 -.312613
vertex -.171799 .0464877 -.321058
vertex .0787653 .072594 -.311967
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.133358 .137441 -.231271
vertex -.133313 .136796 -.230645
vertex -.134208 .136702 -.230678
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex -.133358 .137441 -.231271
vertex -.134208 .136702 -.230678
vertex -.134253 .137348 -.231304
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.113161 -.00480935 -.37944
vertex -.114011 -.00554795 -.378847
vertex -.113116 -.00545474 -.378814
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex -.113161 -.00480935 -.37944
vertex -.114055 -.00490256 -.379473
vertex -.114011 -.00554795 -.378847
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.133358 .137441 -.231271
vertex -.113161 -.00480935 -.37944
vertex -.113116 -.00545474 -.378814
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex -.133358 .137441 -.231271
vertex -.113116 -.00545474 -.378814
vertex -.133313 .136796 -.230645
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.133313 .136796 -.230645
vertex -.113116 -.00545474 -.378814
vertex -.114011 -.00554795 -.378847
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex -.133313 .136796 -.230645
vertex -.114011 -.00554795 -.378847
vertex -.134208 .136702 -.230678
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.134208 .136702 -.230678
vertex -.114011 -.00554795 -.378847
vertex -.114055 -.00490256 -.379473
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.134208 .136702 -.230678
vertex -.114055 -.00490256 -.379473
vertex -.134253 .137348 -.231304
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.134253 .137348 -.231304
vertex -.114055 -.00490256 -.379473
vertex -.113161 -.00480935 -.37944
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex -.134253 .137348 -.231304
vertex -.113161 -.00480935 -.37944
vertex -.133358 .137441 -.231271
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0828048 .044144 -.3416
vertex .0827602 .0447894 -.342226
vertex .0828483 .0441691 -.342872
endloop
endfacet
facet normal -.993969 -.103562 -.036066
outer loop
vertex .0828048 .044144 -.3416
vertex .0828483 .0441691 -.342872
vertex .0828928 .0435237 -.342247
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.16776 .0180376 -.350692
vertex -.167716 .0180627 -.351964
vertex -.167804 .018683 -.351318
endloop
endfacet
facet normal .993969 .103562 .036066
outer loop
vertex -.16776 .0180376 -.350692
vertex -.167672 .0174173 -.351338
vertex -.167716 .0180627 -.351964
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0828048 .044144 -.3416
vertex -.16776 .0180376 -.350692
vertex -.167804 .018683 -.351318
endloop
endfacet
facet normal .0978599 -.689229 -.717905
outer loop
vertex .0828048 .044144 -.3416
vertex -.167804 .018683 -.351318
vertex .0827602 .0447894 -.342226
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0827602 .0447894 -.342226
vertex -.167804 .018683 -.351318
vertex -.167716 .0180627 -.351964
endloop
endfacet
facet normal .0494898 -.717105 .695206
outer loop
vertex .0827602 .0447894 -.342226
vertex -.167716 .0180627 -.351964
vertex .0828483 .0441691 -.342872
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0828483 .0441691 -.342872
vertex -.167716 .0180627 -.351964
vertex -.167672 .0174173 -.351338
endloop
endfacet
facet normal -.0978599 .689229 .717905
outer loop
vertex .0828483 .0441691 -.342872
vertex -.167672 .0174173 -.351338
vertex .0828928 .0435237 -.342247
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0828928 .0435237 -.342247
vertex -.167672 .0174173 -.351338
vertex -.16776 .0180376 -.350692
endloop
endfacet
facet normal -.0494898 .717105 -.695206
outer loop
vertex .0828928 .0435237 -.342247
vertex -.16776 .0180376 -.350692
vertex .0828048 .044144 -.3416
endloop
endfacet
endsolid hccepose_retained_with_depth_samples
```

</td></tr>
</table>

<p align="center"><sub>Drag to rotate, right-drag to pan, scroll to zoom. The native STL preview is monochrome and includes 512 depth samples, CAD and a spatial reference cage. GitHub fits the framing automatically; the website provides calibrated RGB/3D alignment and layer controls.</sub></p>

<details open>
<summary>Scores and original/result crops · 16 instances</summary>

<img src="docs/assets/wapr-region-crops/000003-contact-sheet-en.png" width="100%" alt="Scores and original/result crops · 16 instances">

</details>

## BOP benchmark results

The following screenshots preserve the BOP leaderboard results shown in the original project. They describe the published paper settings and are historical snapshots. See the [BOP evaluation guide](docs/en/bop-evaluation.md) for dataset-specific weights, prediction files and evaluation steps, and the [BOP benchmark](https://bop.felk.cvut.cz/) for current leaderboards.

### 6D object localization

![Historical BOP 6D object localization leaderboard](show_vis/bop-6D-loc.png)

### 2D object segmentation

![Historical BOP 2D object segmentation leaderboard](show_vis/bop-2D-seg.png)

## New objects and the complete training workflow

Public upstream includes mesh centering, rotational symmetry, BOP metadata, PBR rendering, YOLO training, front/back labels, per-object pose training and BOP evaluation scripts. Bilingual docs retain stage-by-stage procedures, public parameters, output layouts and example figures. Stage scripts remain in the repository root; for the original published revision, follow [Installation](docs/en/installation.md#public-training-and-rendering-source); use this project’s `examples/inference/` for application inference.

See [BOP evaluation](docs/en/bop-evaluation.md) for public weights, historical results and leaderboard images. Those historical results describe the original published settings, rather than other data, models or configurations.

## Preview the documentation locally

The published **[website and interactive demos](https://wangyulin-seu.github.io/hccepose/?lang=en)** and **[documentation](https://wangyulin-seu.github.io/hccepose/docs/index.html?lang=en)** are ready to browse. To preview the documentation from this checkout:

```bash
python -m pip install -r docs/requirements.txt
python docs/build.py
python -m http.server 8768 --directory pages
```

Homepage: `http://127.0.0.1:8768/`; docs: `http://127.0.0.1:8768/docs/index.html?lang=en`.

See [project structure](docs/en/project-structure.md), [LICENSE](LICENSE) and [public sources](PUBLIC_SOURCES.md). Third-party components retain their own licenses.

## Citation

```bibtex
@InProceedings{HccePose_BF,
    author    = {Wang, Yulin and Hu, Mengting and Li, Hongli and Luo, Chen},
    title     = {HccePose(BF): Predicting Front \& Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {7166-7175}
}
```
