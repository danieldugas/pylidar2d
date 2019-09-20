import numpy as np
from timeit import default_timer as timer

import lidar_clustering

scan = np.array([ 1.53      ,  1.5339956 ,  1.5409843 ,  1.5479403 ,  1.5459074 , 1.5698668 ,  1.5678197 ,  1.5777647 ,  1.5836343 ,  1.6015576 , 1.6124736 ,  1.6083832 ,  1.6742799 ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  2.3073106 , 2.324145  ,  2.3309727 ,  2.361603  ,  2.3704076 ,  2.3972018 , 2.4449847 ,  2.4577649 ,  2.480535  ,  2.4923    ,  2.4970572 , 2.5218034 ,  2.5495417 ,  2.537551  ,  2.6057124 ,  2.611423  , 2.5831351 ,  2.559838  ,  2.5745234 ,  2.5812037 ,  2.6208665 , 2.681515  ,  2.6791742 ,  2.726809  ,  2.7694383 ,  2.7804413 , 2.7920718 ,  2.8116927 ,  2.8605022 ,  2.8740938 ,  2.909668  , 2.9262414 ,  2.9528039 ,  2.984356  ,  3.006904  ,  3.035441  , 3.0459774 ,  3.0854938 ,  3.1180053 ,  3.135516  ,  3.075054  , 3.1035438 ,  3.1580143 ,  3.171495  ,  3.1989617 ,  3.2334175 , 3.2588694 ,  3.2923098 ,  3.3267415 ,  3.356168  ,  3.3925831 , 3.4259918 ,  3.5673387 ,  3.5977335 ,  3.6501095 ,  3.69748   , 3.7398455 ,  3.7182345 ,  3.6876218 ,  3.658001  ,  3.6667018 , 3.7360067 ,  0.        ,  4.0278606 ,  4.3360415 ,  4.3813467 , 4.1310973 ,  4.395272  ,  4.430551  ,  4.5008054 ,  4.5640564 , 4.6372952 ,  4.729519  ,  4.7837524 ,  4.87796   ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 5.3878627 ,  5.1700807 ,  5.152204  ,  4.995386  ,  5.129426  , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  7.918638  ,  7.7662506 ,  7.6648364 , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  4.9105034 ,  4.883085  , 4.8007374 ,  4.810275  ,  4.8078275 ,  4.8163686 ,  4.844885  , 4.8534284 ,  4.885942  ,  0.        ,  4.907473  ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  7.8126216 ,  7.782214  ,  7.751807  , 7.8433456 ,  0.        ,  0.        ,  0.        ,  0.        , 3.5681338 ,  3.566499  ,  0.        ,  0.        ,  3.7295783 , 3.7458959 ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 4.224759  ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.5819443 , 0.        ,  0.        ,  1.2379493 ,  0.        ,  0.6144757 , 1.2291027 ,  0.        ,  0.        ,  0.        ,  0.32962722, 0.6205187 ,  0.88632613,  1.1426613 ,  0.3737923 ,  0.39845943, 0.39075875,  1.3710179 ,  0.4366448 ,  0.6300158 ,  1.2581147 , 1.3430172 ,  1.336992  ,  1.3279878 ,  1.3298775 ,  1.3237293 , 1.3236194 ,  0.        ,  1.3046712 ,  1.3133392 ,  1.3013098 , 1.3090984 ,  1.2752466 ,  1.2691432 ,  1.2759216 ,  1.2707973 , 1.2775649 ,  0.        ,  0.        ,  0.        ,  0.        , 1.2754115 ,  1.2722199 ,  1.2613071 ,  1.2814658 ,  1.287185  , 1.2899204 ,  1.399473  ,  0.        ,  5.519064  ,  5.521788  , 5.5382113 ,  5.536915  ,  5.550607  ,  5.5512977 ,  5.5539804 , 5.549657  ,  5.5513253 ,  5.5769815 ,  5.5776362 ,  5.592281  , 5.589922  ,  5.5955553 ,  5.604179  ,  5.617796  ,  5.6274066 , 5.64001   ,  5.642607  ,  5.6501975 ,  5.657779  ,  5.6713543 , 5.6689234 ,  5.6744843 ,  5.701583  ,  5.704123  ,  5.720654  , 5.7321777 ,  5.7446947 ,  5.7512054 ,  5.7707057 ,  5.7742014 , 5.782689  ,  5.7931695 ,  5.8066425 ,  5.8171067 ,  0.        , 0.        ,  0.        ,  5.7899003 ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  7.369657  ,  7.2360415 , 6.9864225 ,  6.9967856 ,  6.942142  ,  6.8224945 ,  6.8568325 , 6.949162  ,  7.0324826 ,  7.066798  ,  7.0861063 ,  7.0104084 , 6.9177046 ,  6.9489894 ,  7.056263  ,  7.074533  ,  7.109794  , 7.116048  ,  7.1362944 ,  7.1705313 ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  3.2658887 ,  3.2380652 , 3.1502368 ,  2.85941   ,  2.88656   ,  2.9697006 ,  2.9808369 , 2.9869657 ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 3.3939993 ,  3.4609919 ,  3.463977  ,  3.479954  ,  3.5029233 , 3.5198848 ,  3.5428388 ,  3.5557854 ,  3.5717235 ,  3.5886548 , 3.611578  ,  3.6264932 ,  3.6494005 ,  3.670301  ,  3.7060769 , 3.7239532 ,  3.733822  ,  3.7636828 ,  3.7785366 ,  3.7993817 , 0.        ,  0.        ,  0.        ,  0.        ,  2.8333592 , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 2.6538084 ,  2.6095626 ,  2.4903235 ,  2.58004   ,  2.5957615 , 2.610476  ,  2.6111856 ,  2.5898936 ,  2.566595  ,  2.559967  , 2.559641  ,  2.5563092 ,  2.5519695 ,  2.5646176 ,  2.5672612 , 2.5629003 ,  2.5495358 ,  2.5671518 ,  2.5643775 ,  2.5579813 , 2.5565758 ,  2.5611606 ,  2.573734  ,  2.5763056 ,  2.5808682 , 2.5824254 ,  2.5649853 ,  2.5535367 ,  2.5595996 ,  2.57911   , 2.5636353 ,  2.5691402 ,  2.5806327 ,  2.6041105 ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  3.3619492 ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  2.8870792 ,  2.8686569 ,  2.8249753 ,  2.7942753 , 2.7905374 ,  2.769814  ,  2.8010201 ,  2.8052523 ,  2.7994914 , 2.673111  ,  2.6833112 ,  2.7094808 ,  2.7426322 ,  2.7917511 , 2.7659783 ,  2.7132418 ,  2.766333  ,  2.815422  ,  2.798612  , 2.8197331 ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        , 10.811511  ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 1.4521989 ,  1.4358414 ,  1.4473921 ,  1.4125721 ,  1.4221421 , 1.4507413 ,  1.4504405 ,  1.4540771 ,  1.4448968 ,  1.4643077 , 1.4531543 ,  1.4922802 ,  1.5166218 ,  1.4548444 ,  1.6845176 , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  1.3203733 ,  0.        , 1.3088386 ,  1.3300719 ,  1.3385354 ,  1.3411111 ,  1.3112781 , 1.3148263 ,  1.3046306 ,  1.3219255 ,  1.3234985 ,  1.31918   , 1.3466775 ,  1.2954655 ,  0.        ,  1.38425   ,  0.        , 0.        ,  1.4018825 ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  2.9830334 ,  2.830434  ,  2.7992485 , 2.7790227 ,  2.8528118 ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  1.8437372 ,  1.8333961 ,  1.8180996 ,  1.8033683 , 1.7549411 ,  1.7366811 ,  1.7406499 ,  1.7313095 ,  1.7229593 , 1.7245166 ,  1.6747117 ,  1.7234389 ,  1.6778271 ,  1.6675047 , 1.659163  ,  1.6409222 ,  1.6438984 ,  1.6424863 ,  1.6252416 , 1.6268015 ,  1.628362  ,  1.6012248 ,  1.57722   ,  1.5669129 , 1.5645231 ,  1.577385  ,  1.5670815 ,  1.5548025 ,  1.5365903 , 1.5440924 ,  1.5397305 ,  1.5294363 ,  1.5161778 ,  1.4922286 , 1.4997395 ,  1.4706738 ,  1.4771984 ,  1.4732491 ,  1.4698889 , 1.4566479 ,  1.4641638 ,  1.4671361 ,  1.4535043 ,  1.4373077 , 1.4309969 ,  1.4485799 ,  1.4183785 ,  1.4217792 ,  1.4261425 , 0.        ,  1.4143324 ,  1.4141836 ,  1.3840895 ,  1.3884393 , 1.3721219 ,  1.3725312 ,  1.3660511 ,  0.        ,  1.3670405 , 1.3605566 ,  1.3511218 ,  1.3578047 ,  1.349343  ,  1.3330159 , 1.3225869 ,  1.3111739 ,  1.3105673 ,  1.3197842 ,  1.309343  , 1.3106912 ,  0.        ,  1.3015847 ,  0.        ,  1.2927505 , 1.2980155 ,  1.2865788 ,  1.2921759 ,  1.2787646 ,  1.281063  , 1.2755022 ,  1.2669916 ,  1.2673148 ,  1.2686164 ,  1.265004  , 1.2554952 ,  1.251876  ,  1.2561061 ,  1.2406961 ,  1.2478635 , 1.2412816 ,  1.2405846 ,  1.2379214 ,  1.2362334 ,  1.2247267 , 1.230884  ,  1.2272214 ,  1.2343527 ,  1.2179172 ,  1.2260214 , 1.2193937 ,  1.2245437 ,  1.2257622 ,  1.2220663 ,  1.2124741 , 1.2028768 ,  1.2021127 ,  1.2091978 ,  1.2044951 ,  1.1988051 , 1.199985  ,  1.2001777 ,  1.199383  ,  1.2015305 ,  1.1928681 , 1.1938547 ,  1.1910809 ,  1.1895988 ,  1.1917164 ,  1.1889158 , 1.1802133 ,  1.1872299 ,  1.1834303 ,  1.182574  ,  1.1856446 , 1.1828116 ,  1.1730901 ,  1.1750616 ,  1.1800841 ,  1.168561  , 1.1755306 ,  1.1736417 ,  1.1678114 ,  1.1688626 ,  1.1728616 , 1.1679975 ,  1.1650957 ,  1.1701039 ,  1.1662    ,  1.1721382 , 1.1642832 ,  1.1692264 ,  1.1672686 ,  1.1672757 ,  1.1672773 , 1.1682492 ,  1.1652766 ,  1.1662412 ,  1.1632563 ,  1.1681541 , 1.1671301 ,  1.1630903 ,  1.1650081 ,  1.1659333 ,  1.1638921 , 1.168754  ,  1.1676888 ,  1.1675143 ,  1.167418  ,  1.1692921 , 1.1691843 ,  1.170059  ,  1.1696874 ,  1.17166   ,  1.1764663 , 1.1644582 ,  1.1702417 ,  1.1717412 ,  1.1658    ,  1.1725523 , 1.1723734 ,  1.1632798 ,  1.175559  ,  1.1735753 ,  1.1793077 , 1.1771116 ,  1.1719369 ,  1.1714736 ,  1.18709   ,  1.177924  , 1.1875855 ,  1.1860727 ,  1.1848106 ,  1.1924698 ,  1.1822665 , 1.1893233 ,  1.1922961 ,  1.1850401 ,  1.1946532 ,  1.1986445 , 1.1946678 ,  1.1983024 ,  1.2049117 ,  1.2018578 ,  1.2048349 , 1.2004893 ,  1.2021006 ,  1.2069043 ,  1.2178382 ,  1.218432  , 1.2100672 ,  1.2142061 ,  1.2227347 ,  1.2163306 ,  1.2367934 , 1.2289283 ,  1.225483  ,  1.2309924 ,  1.2385273 ,  1.2405155 , 1.2390289 ,  1.2365389 ,  1.2519538 ,  1.2479631 ,  1.2544177 , 1.2528919 ,  1.2518133 ,  1.2582438 ,  1.2656662 ,  1.2690939 , 1.2689486 ,  1.2723576 ,  1.2757599 ,  1.2851433 ,  1.2859311 , 1.2883111 ,  1.2856941 ,  1.2957749 ,  1.2957698 ,  1.2991158 , 1.3044527 ,  1.3071191 ,  1.3104398 ,  1.3207446 ,  1.322055  , 1.3226583 ,  1.3289435 ,  1.3312262 ,  1.3345019 ,  1.341033  , 1.352282  ,  1.3495361 ,  1.3487816 ,  1.3592443 ,  1.3694628 , 1.3696809 ,  1.3688924 ,  1.3802891 ,  1.389476  ,  1.3816628 , 1.3888375 ,  1.4021667 ,  1.403323  ,  1.4114708 ,  1.4106137 , 1.429875  ,  1.4379958 ,  1.4451102 ,  1.441219  ,  1.4525074 , 1.454501  ,  1.4645807 ,  1.4616543 ,  1.4667201 ,  1.4718319 , 1.4878769 ,  1.4989148 ,  1.494946  ,  1.5069704 ,  1.5139973 ],
      dtype=np.float32)

# PROCESSING ----------------------------------------

# clustering
EUCLIDEAN_CLUSTERING_THRESH_M = 0.2
angles = np.linspace(0, 2*np.pi, scan.shape[0]+1, dtype=np.float32)[:-1]
xx = np.cos(angles) * scan
yy = np.sin(angles) * scan
tic = timer()
clusters, _, _ = lidar_clustering.euclidean_clustering(scan, angles,
                                                 EUCLIDEAN_CLUSTERING_THRESH_M)
cluster_sizes = lidar_clustering.cluster_sizes(len(scan), clusters)
toc = timer()
print("Clustering : {} ms".format((toc-tic)*1000.))

# center of gravity
tic = timer()
cogs = [[np.mean(xx[c]), np.mean(yy[c])] for c in clusters]
radii = [np.max(np.sqrt((xx[c]-cog[0])**2 + (yy[c]-cog[1])**2)) for cog, c in zip(cogs, clusters)]
toc = timer()
print("C.O.Gs, Radii : {} ms".format((toc-tic)*1000.))

# VISUALS -------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.figure()
plt.scatter(xx, yy, zorder=1, facecolor=(1,1,1,1), edgecolor=(0,0,0,1), color='k', marker='.')
for x, y in zip(xx, yy):
    plt.plot([0, x], [0, y], linewidth=0.01, color='red' , zorder=2)

for c in clusters:
    plt.plot(xx[c], yy[c], zorder=2)

for cog, r in zip(cogs, radii):
    patch = patches.Circle(cog, r, facecolor=(0,0,0,0), edgecolor=(0,0,0,1), linestyle='--')
    plt.gca().add_artist(patch)

plt.axis('equal')
plt.show()
