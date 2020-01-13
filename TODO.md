

~~1. add dropout on mcep aux~~  lower dropout rate
~~2. add 1D conv with p~~
3. make additional dataset
**4. multiple channel p + random roll**
5. negative p value
6. harmonic pulse p
~~7. run no pulse repeat1 evaluate f0 shift semi-tone~~
8. improve evaluate
~~9. not using future p~~
~~10. integrate mask mcep on p (channel_gate)~~
11. better f0 augment
12. use 8 or 9 layer instead of 10
13. input x
14. remove + - in eva set
15. plot more f0 picture
16. 180k step run f0 eva
17. remove dil_x  in some layer

# make time GMT+8:00



2019-11-30@10_20_47 : baseline fn=12
2019-12-01@12_08_04 : remove aux signal in tanh
2019-12-02@01_32_48 : add dropout
2019-12-03@01_27_05 : add conv1d on p
2019-12-03@14_46_57 : multiple p chan
2019-12-15@23_56_15 : remove mcep
2019-12-16@04_37_29 : not using future p
2019-12-17@05_33_30 : p channel gated by mcep
2019-12-17@14_39_43 : 50 p aux_ch, fix vuv and ap_code wrong way
2019-12-18@01_04_02 : try fn=1 random phase
2019-12-18@11_19_44 : try harmonic 4
2019-12-18@16_55_37 : try 2 layer p conv1d <- bad attempt
2019-12-18@17_34_58 : fix pad side <- bad attempt, canceled
2019-12-18@16_55_37 : try 2 layer p conv1d with residual skip <- bad attempt
2019-12-18@20_13_04 : remove p conv input in sigmoid <- good attempt (bad, removed)
2019-12-18@21_42_51 : use transpose conv1 as upsampling instead of expand <- good attempt
2019-12-19@00_06_10 : use dilate conv at p <- good attempt
2019-12-19@00_38_25 : add leakReLU at mcep before upsample <- bad attempt
2020-01-13@05_20_12 : remove batch_norm and relu on mcep


try: modify mcep : is batch norm good?

## TODO: decode input 對齊問題? -> 多review幾次code
## TODO: overfitting 問題? -> try run train set