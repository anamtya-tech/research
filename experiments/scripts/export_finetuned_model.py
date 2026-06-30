#!/usr/bin/env python3
"""Export the fine-tuned (SpecAugment) YAMNet model to a deployable TFLite + class map, register
it in the yamnet model store, and stage the device files. Separate script — no repo code modified.

Model: experiments/sim/checkpoints/yamnet_finetuned.keras  (96x64 mel -> 7-class softmax)
Output:
  yamnet/model_store/releases/<version>/chatak_yamnet_<version>.tflite   (gitignored binary)
  yamnet/model_store/releases/<version>/custom_class_map.csv             (3-col, firmware-compatible)
  yamnet/model_store/registry.json                                       (TRACKED — the check-in)
  experiments/odas/deploy/{yamnet_core.tflite,yamnet_class_map.csv}      (staged device files)
"""
import os
os.environ["TF_USE_LEGACY_KERAS"]="1"
import sys, csv, json, shutil
import numpy as np, tensorflow as tf

ROOT="/Users/abhinav/research"; EXP=f"{ROOT}/experiments"
KERAS=f"{EXP}/sim/checkpoints/yamnet_finetuned.keras"
VERSION="v2.0.0-specaug"
REL=f"{ROOT}/yamnet/model_store/releases/{VERSION}"
REG=f"{ROOT}/yamnet/model_store/registry.json"
DEPLOY=f"{EXP}/odas/deploy"

# class order MUST match the model's output index order (sorted unique corpus labels)
CLASSES=sorted(set(r["label"] for r in csv.DictReader(open(f"{EXP}/corpus/meta.csv"))))
METRICS=dict(overall_test=0.62, mixed_test=0.61, train_test_gap=0.13,
             per_class={"Lion":0.90,"Frog":0.68,"Elephant":0.54,"Bear":0.53,
                        "drone_bebop":0.54,"background":0.64,"drone_binary":0.19})

def main():
    os.makedirs(REL,exist_ok=True); os.makedirs(DEPLOY,exist_ok=True)
    print(f"classes ({len(CLASSES)}):", CLASSES)
    model=tf.keras.models.load_model(KERAS)
    assert model.output_shape[-1]==len(CLASSES), f"model outputs {model.output_shape[-1]} vs {len(CLASSES)} classes"

    # --- TFLite (float32) ---
    conv=tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations=[]
    tfl=conv.convert()
    tfl_path=f"{REL}/chatak_yamnet_{VERSION}.tflite"
    open(tfl_path,"wb").write(tfl)
    print(f"wrote {tfl_path} ({len(tfl)//1024} KB)")

    # --- verify the TFLite runs and shapes are right ---
    it=tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
    inp=it.get_input_details()[0]; out=it.get_output_details()[0]
    print(f"  TFLite input {inp['shape']} {inp['dtype'].__name__}  output {out['shape']} {out['dtype'].__name__}")
    assert tuple(out['shape'][-1:])==(len(CLASSES),)
    x=np.random.randn(1,96,64).astype(np.float32)
    it.set_tensor(inp['index'],x); it.invoke()
    y=it.get_tensor(out['index'])
    print(f"  sample output sums to {y.sum():.3f} (softmax≈1), argmax={CLASSES[int(y.argmax())]}")
    assert abs(y.sum()-1.0)<1e-3, "output is not a valid softmax"

    # --- class maps ---
    # 3-col (index,mid,display_name) — firmware LoadClassNames reads field[2]
    cm=f"{REL}/custom_class_map.csv"
    with open(cm,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["index","mid","display_name"])
        for i,c in enumerate(CLASSES): w.writerow([i,f"/m/custom_{i}",c])
    print(f"wrote {cm}")

    # --- staged device files (fixed names the firmware loads) ---
    shutil.copy(tfl_path,f"{DEPLOY}/yamnet_core.tflite")
    shutil.copy(cm,f"{DEPLOY}/yamnet_class_map.csv")
    print(f"staged device files -> {DEPLOY}/ (yamnet_core.tflite, yamnet_class_map.csv)")

    # --- export_info ---
    info=dict(version=VERSION, classes=CLASSES, num_classes=len(CLASSES),
              source_checkpoint=KERAS, metrics=METRICS,
              training="backbone fine-tune (unfreeze top 6) + SpecAugment, corpus=7251 post-ODAS samples",
              odas_integration=dict(model_path_dir_contains=["yamnet_core.tflite","yamnet_class_map.csv"],
                                    input="96x64 logmel patch", output=f"{len(CLASSES)}-class softmax"))
    json.dump(info,open(f"{REL}/export_info.json","w"),indent=2)

    # --- register (TRACKED) — experimental, NOT auto-deployed ---
    reg=json.load(open(REG))
    entry=dict(run_name="chatak_yamnet_specaug_ft", nickname="backbone FT + SpecAugment (7-class, ~0.62)",
               timestamp="20260630", classes=CLASSES, num_classes=len(CLASSES),
               val_accuracy=METRICS["overall_test"],
               model_path=KERAS,
               tflite_path=tfl_path, tflite_int8_path=None,
               dataset=f"{EXP}/corpus (7251 post-ODAS samples, 4 ambient envs, 3 SNRs)",
               deployed=False, version=VERSION, exported_at="20260630",
               metrics=METRICS, notes="separate experiments/ track; breaks ~0.50 ceiling via SpecAugment")
    reg["models"]=[m for m in reg["models"] if m.get("run_name")!=entry["run_name"]]+[entry]
    json.dump(reg,open(REG,"w"),indent=2)
    print(f"registered '{entry['run_name']}' {VERSION} in registry.json (deployed=false, active_model unchanged)")

if __name__=="__main__":
    main()
