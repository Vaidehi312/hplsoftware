import openslide

p="/Users/vaidehipandya/mnt/tcga_wsi/TCGA-55-7574-01A-01-TS1.13d42a4b-074e-45a7-8a31-dbe505408711.svs"

s = openslide.OpenSlide(p)
print(s.level_dimensions)