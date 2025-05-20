@echo on

if not exist results (
  mkdir results
)
@REM Options:
@REM   -m, --models [efficientnet_b0|resnet50|mobilenet_v1|inception_v1|HyCoCLIPPytorchModel|hycoclip|resnet50_trained_on_SIN|resnet50_trained_on_SIN_and_IN|resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN|
@REM bagnet9|bagnet17|bagnet33|simclr_resnet50x1_supervised_baseline|simclr_resnet50x4_supervised_baseline|simclr_resnet50x1|simclr_resnet50x2|simclr_resnet50x4|InsDis|MoCo|MoCoV2|PIRL|InfoMin|
@REM resnet50_l2_eps0|resnet50_l2_eps0_01|resnet50_l2_eps0_03|resnet50_l2_eps0_05|resnet50_l2_eps0_1|resnet50_l2_eps0_25|resnet50_l2_eps0_5|resnet50_l2_eps1|resnet50_l2_eps3|resnet50_l2_eps5|
@REM efficientnet_b0|efficientnet_es|efficientnet_b0_noisy_student|efficientnet_l2_noisy_student_475|transformer_B16_IN21K|transformer_B32_IN21K|transformer_L16_IN21K|transformer_L32_IN21K|
@REM vit_small_patch16_224|vit_base_patch16_224|vit_large_patch16_224|cspresnet50|cspresnext50|cspdarknet53|darknet53|dpn68|dpn68b|dpn92|dpn98|dpn131|dpn107|hrnet_w18_small|hrnet_w18_small|
@REM hrnet_w18_small_v2|hrnet_w18|hrnet_w30|hrnet_w40|hrnet_w44|hrnet_w48|hrnet_w64|selecsls42|selecsls84|selecsls42b|selecsls60|selecsls60b|clip|clipRN50|resnet50_swsl|ResNeXt101_32x16d_swsl|
@REM BiTM_resnetv2_50x1|BiTM_resnetv2_50x3|BiTM_resnetv2_101x1|BiTM_resnetv2_101x3|BiTM_resnetv2_152x2|BiTM_resnetv2_152x4|resnet50_clip_hard_labels|resnet50_clip_soft_labels|swag_regnety_16gf_in1k|
@REM swag_regnety_32gf_in1k|swag_regnety_128gf_in1k|swag_vit_b16_in1k|swag_vit_l16_in1k|swag_vit_h14_in1k]
@REM                                   [required]
@REM   -d, --datasets [imagenet_validation|sketch|stylized|original|greyscale|texture|edge|silhouette|cue-conflict|colour|contrast|high-pass|low-pass|phase-scrambling|power-equalisation|false-colour|rotation|eidolonI|eidolonII|eidolonIII|uniform-noise]
@REM                                   [required]
@REM   -t, --test-run                  If the test-run flag is set, results will
@REM                                   not be saved to csv
@REM   -w, --num-workers INTEGER       Number of cpu workers for data loading
@REM   -b, --batch-size INTEGER        Batch size during evaluation
@REM   -p, --print-predictions BOOLEAN
@REM                                   Print predictions
@REM   --help                          Show this message and exit.


@echo on
if not exist results mkdir results

for %%M in (hycoclip resnet50_trained_on_SIN  resnet50 MoCoV2 resnet50_l2_eps0) do (
  echo -------------------------------------------------
  echo Running %%M on cue-conflict
  echo -------------------------------------------------
  python -m modelvshuman -m %%M -d cue-conflict -b 64 -w 8 -t -p False ^
    > results\%%M_cue-conflict_log.txt 2>&1
  if errorlevel 1 (echo [ERROR] %%M failed) else (echo [DONE] %%M)
)

pause

