

~~1. add dropout on mcep aux~~  lower dropout rate
~~2. add 1D conv with p~~
3. make additional dataset
**4. multiple channel p + random roll**
5. negative p value
6. harmonic pulse p
7. run no pulse repeat1 evaluate f0 shift semi-tone
8. improve evaluate
~~9. not using future p~~
10. integrate mask mcep on p (channel_gate)


# make time GMT+8:00



2019-11-30@10_20_47 : baseline fn=12
2019-12-01@12_08_04 : remove aux signal in tanh
2019-12-02@01_32_48 : add dropout
2019-12-03@01_27_05 : add conv1d on p
2019-12-03@14_46_57 : multiple p chan
2019-12-15@23_56_15 : remove mcep
2019-12-16@04_37_29 : not using future p