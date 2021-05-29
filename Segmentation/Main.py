from ProcessImage import ProcessImage
pimg = ProcessImage()
pimg.main()
# pimg.load(np.arange(1,144)) #80, 101, 125
idList = []
idRange = "1, 2"
rangeList = idRange.split(",")
for rg in rangeList:
    if '-' in rg:
        range2 = rg.split('-')
        start = int(range2[0])
        end = int(range2[1]) + 1
        idList.extend(range(start, end))
    else:
        idList.append(int(rg))

import cProfile
# cProfile.run('pimg.runSaveGTStats(idList)')

# prof = cProfile.Profile()
# prof.enable()
pimg.runSaveGTStats(idList)
# prof.disable()
# prof.print_stats()
# prof.dump_stats("main_func.prof")


# print(idList)
# pimg.display7(idList)
# # pimg.runSaveGTStats(np.arange(1, 144))
# # pimg.segList.saveROIStats()
import pstats
# p = pstats.Stats("main_func.prof")

# p.print_stats()

# p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(50)

# p.sort_stats(pstats.SortKey.TIME, pstats.SortKey.CUMULATIVE).print_stats(5)

# p.print_stats("BUSSegmentorList.py")