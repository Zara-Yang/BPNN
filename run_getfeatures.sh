# 上次搞到170 LV
for((i=0; i<1500; i++))
do
    date
    cp Produce_features.o /DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Origin_data/Bulk64_Yihao/"$i"
    cd /DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Origin_data/Bulk64_Yihao/"$i"/
    ./Produce_features.o
    cd /DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Code
    echo "$i Finish !"
done
