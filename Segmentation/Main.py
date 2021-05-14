from ProcessImage import ProcessImage
pimg = ProcessImage()
pimg.main()
# pimg.load(np.arange(1,144)) #80, 101, 125
idList = []
idRange = "1, 2, 5-10, 68"
rangeList = idRange.split(",")
for rg in rangeList:
    if '-' in rg:
        range2 = rg.split('-')
        start = int(range2[0])
        end = int(range2[1]) + 1
        idList.extend(range(start, end))
    else:
        idList.append(int(rg))


print(idList)
pimg.display7(idList)
# # pimg.runSaveGTStats(np.arange(1, 144))
# # pimg.segList.saveROIStats()