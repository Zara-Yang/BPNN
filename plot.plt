

set term png
set output "Oxygen_pred_error.png"
set size 1,1

set multiplot layout 2, 1

set origin 0,0
set xrange [-40:40]
set size 1,0.5
plot "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Model_result/Bulk_test_O.dat" u 1:2 w p t "Bulk water"

set origin 0,0.5
set xlabel "Z-axis(Bohr)"
set ylabel "Force loss(82.387 nN)"
set xrange [-40:40]
set size 1,0.5
plot "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Model_result/LV_test_O.dat" u 1:2 w p t "LV water"

unset multiplot
set output
