from trem_researcher import TremResearcher, sep

#researcher = TremResearcher()
#researcher.calibrate(r'D:\Projects\Tremor\Calibration\Calibration photo\Accurate attempt 5')
#researcher.save_calib_settings(r'D:\Projects\Tremor\BIN FILES\Calibration for Accurate attempt 5 (all) (corrected).pkl')
#print(researcher.mtx, researcher.dist)

researcher1 = TremResearcher()
researcher2 = TremResearcher()
researcher1.load_calib_settings(r'D:\Projects\Tremor\BIN FILES\Calibration for Accurate attempt 4.pkl')
researcher2.load_calib_settings(r'D:\Projects\Tremor\BIN FILES\Calibration for Accurate attempt 5 (all) (corrected).pkl')

print("MTX:", researcher1.mtx - researcher2.mtx)
print("DIST:", researcher1.dist - researcher2.dist)
print(researcher2.mtx.dtype)