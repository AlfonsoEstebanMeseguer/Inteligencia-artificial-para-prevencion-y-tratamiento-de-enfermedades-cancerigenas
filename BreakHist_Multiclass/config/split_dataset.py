import sys
from pathlib import Path

ROOT_DIR=Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0,str(ROOT_DIR))

from BreakHist_Binary.src.config.split_dataset import (split_by_patient,split_by_image,plot_split_distributions,parse_arguments,DEFAULT_BASE_PATH_MULTICLASS)
from BreakHist_Multiclass.config.readDataset import read_multiclass_breakhis_data  # noqa: E402

def main():
    args=parse_arguments()
    args.dataset_type="multiclass"
    if args.base_path is None:
        args.base_path=DEFAULT_BASE_PATH_MULTICLASS
    data,all_images,all_labels,label_map,slides=read_multiclass_breakhis_data(base_path=args.base_path,verbose=False)
    if args.split_mode=="patient":
        splits,stats=split_by_patient(all_images,all_labels,slides,args.train_size,args.val_size,args.test_size,args.random_state
                                      ,dataset_type="multiclass",label_map=label_map)
    else:
        splits,stats=split_by_image(all_images,all_labels,slides,args.train_size,args.val_size,args.test_size,args.random_state
                                    ,dataset_type="multiclass",label_map=label_map)
    print("\nESTADÍSTICAS DEL SPLIT:")
    inv_map={}
    for k,v in label_map.items():
        inv_map[v]=k
    for split,s in stats.items():
        print(f"\n{split.upper()}")
        print(f"Imágenes:{s['num_images']}")
        print(f"Pacientes:{s['num_patients']}")
        for cls_idx,count in enumerate(s["class_counts"]):
            cls_name=inv_map.get(cls_idx,f"class_{cls_idx}")
            print(f"{cls_name}({cls_idx}):{count}")
        print("Ratios por clase:")
        for cls_name in s["class_ratios_named"]:
            ratio=s["class_ratios_named"][cls_name]
            print(f"{cls_name}:{ratio:.2f}%")

    for split_name,split_data in splits.items():
        path=Path(args.output_dir)/f"{split_name}.json"
        path.parent.mkdir(parents=True,exist_ok=True)
        with open(path,"w",encoding="utf-8") as f:
            import json

            json.dump(split_data,f,indent=2)

    print(f"\nArchivos guardados en: {args.output_dir}")
    if not args.no_plot:
        num_classes=len(stats[splits[0]]["class_counts"])
        class_names=[]
        for i in range(num_classes):
            class_names.append(inv_map.get(i,f"c{i}"))
        plot_split_distributions(stats,dataset_type="multiclass",class_names=class_names)

if __name__=="__main__":
    main()
