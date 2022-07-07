import sys
import time
import numpy as np
import mmwave.dsp as dsp
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mmwave.dsp.utils import Window
from mmwave.dataloader.adc import DCA1000
import gc
from oscfar import os_cyth
from music import music_cyth


gc.disable()


plt.close('all')

# QOL settings
loadData = True
datapath = r"D:\\03_Inbetriebnahme DCA\\DUMP\\leicht_schraeg\\"

numFrames = 900
numADCSamples = 256
numTxAntennas = 2
numRxAntennas = 4
numLoopsPerFrame = 128
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 128

range_resolution, bandwidth, max_range = dsp.range_resolution(numADCSamples)
doppler_resolution, max_doppler = dsp.doppler_resolution(bandwidth)

print("range resolution: " + str(range_resolution))
print("doppler resolution: " + str(doppler_resolution))

max_range = 18
max_doppler = 7

plotRangeDopp = False
plot2DXYdoppler = False
plot2DXYClusters = True
plotmusic = False
saveVideo = True


def main():
    ims = []
    max_size = 0

    # (1) Reading in adc data
    if loadData:
        adc_data = np.fromfile(
            datapath + "adc_data.bin",
            dtype=np.uint16)

        adc_data = adc_data.reshape(numFrames, -1)
        adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                       num_rx=numRxAntennas, num_samples=numADCSamples)
        print("Data Loaded!")

    # (1.1) Required Plot Declarations
    if plot2DXYdoppler:
        fig, axes = plt.subplots(1, 2)
    elif plot2DXYClusters:
        fig, axes = plt.subplots(1, 2)
    elif plotmusic:
        fig, axes = plt.subplots(1, 2)

    for i, frame in enumerate(adc_data[50:400]):  # leicht_schraeg:50, gerade_vor:120, 90grad:50
        #        print(i,end=',') # Frame tracker
        # (2) Range Processing
        start = time.time()

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BARTLETT)
        assert radar_cube.shape == (
            numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=2, clutter_removal_enabled=False,
                                                       window_type_2d=Window.HAMMING)

        # (4) Detection of Reflections
        # --- CFAR, SNR is calculated as well.
        fft2d_sum = det_matrix.astype(np.int64)

        startOS = time.time()
        thresholdRange, noiseFloorRange = os_cyth(fft2d_sum, guard_len=4, noise_len=10, scale=1.05, axis=1)
        #thresholdDoppler, noiseFloorDoppler = os_cyth(fft2d_sum, guard_len=4, noise_len=20, scale=1.05, axis=0)
        print("OS time: " + str(time.time()-startOS))
        det_doppler_mask = (fft2d_sum > 0)
        det_range_mask = (fft2d_sum > thresholdRange)

        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # evaluate doppler variance for Micro-Doppler-recognition of wheels
        doppler_var = dsp.get_micro_doppler(aoa_input, numLoopsPerFrame, det_peaks_indices)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        #snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        dtype_location = '(' + str(numTxAntennas) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'azimuth', 'doppler_var'],
                                   'formats': ['<i4', '<i4', '<i4', '<f4', dtype_location, '<f4']})
        detRefl = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detRefl['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detRefl['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()

        azimuthInput = aoa_input[detRefl['rangeIdx'], :, detRefl['dopplerIdx']]


        Theta, Ranges, dopplerIdx, xydoppVec = dsp.beamforming_naive_mixed_xy(azimuthInput, detRefl['rangeIdx'],
                                                                              detRefl['dopplerIdx'],
                                                                              range_resolution, doppler_resolution,
                                                                              numLoopsPerFrame, method='Bartlett')

        # (5) 3D-Clustering
        # detObj2D must be fully populated and completely accurate right here
        numDetObjs = Ranges.shape[0]
        dtf = np.dtype({'names': ['location_x','location_y','azimuth', 'doppler', 'cluster', 'variance_D'],
                        'formats': ['<f4', '<f4', '<f4', '<f4', '<i4', '<f4']})
        detObj2D = np.zeros((numDetObjs,), dtype=dtf)
        detObj2D['location_x'] = xydoppVec[0]
        detObj2D['location_y'] = xydoppVec[1]
        detObj2D['azimuth'] = Theta
        detObj2D['doppler'] = xydoppVec[2]
        #detObj2D['variance_D'] = doppler_var_sum_angle[Ranges, dopplerIdx]

        if len(detObj2D) > 0:
            cluster = clu.radar_dbscan_ms(detObj2D, 4, doppler_resolution)
            detObj2D_nf = detObj2D      #noise_free
            for j in reversed(range(numDetObjs)):
                delete = False
                if detObj2D_nf['cluster'][j] == -1:
                    delete = True
                elif np.abs(cluster['avgVelocity'][detObj2D_nf['cluster'][j]]*0.2) < np.abs(detObj2D_nf['doppler'][j]) < np.abs(cluster['avgVelocity'][detObj2D_nf['cluster'][j]]*1.8):
                    delete = True
                if delete == True:
                    detObj2D_nf = np.delete(detObj2D_nf, j, axis=0)

        end = time.time()
        print("execution time " + str(i) + ": " + str(end - start))



        # (6) Visualization
        if plotRangeDopp:
            continue

        elif plot2DXYdoppler:
            ann0 = axes[0].annotate(f"XY scatter " + str(i), (0.5, 1.03), xycoords="axes fraction", ha="center")
            ann1 = axes[1].annotate(f"velocity profile " + str(i), (0.5, 1.03), xycoords="axes fraction", ha="center")

            axes[0].set_ylim(bottom=0, top=max_range)
            axes[0].set_ylabel('x-range [m]')
            axes[0].set_xlim(left=-10, right=10)
            axes[0].set_xlabel('y-range [m]')
            axes[0].grid(visible=True)

            axes[1].set_xlabel('azimuth [°]')
            axes[1].set_ylabel('velocity [m/s]')
            axes[1].grid(visible=True)
            axes[1].set_xlim(left=-40, right=40)
            axes[1].set_ylim(bottom=-3, top=3)

            if not saveVideo:
                p0 = axes[0].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['doppler'],
                                     vmin=-max_doppler, vmax=max_doppler, marker='o', s=20, cmap=cm.jet)
                p1 = axes[1].scatter(detObj2D['azimuth'], detObj2D['doppler'], marker='o', s=20)
                cb = plt.colorbar(p0, ax=axes[0])
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
                cb.remove()
            else:
                p0 = axes[0].scatter(detObj2D_nf['location_x'], detObj2D_nf['location_y'], c=detObj2D['doppler'],
                                     vmin=-max_doppler, vmax=max_doppler, marker='o', s=5, cmap=cm.jet)
                p1 = axes[1].scatter(detObj2D_nf['azimuth'], detObj2D_nf['doppler'], marker='o', s=5, c='#1f77b4')
                p = [p0, p1, ann0, ann1]
                ims.append(p)

        elif plot2DXYClusters:
            ann0 = axes[0].annotate(f"XY scatter " + str(i), (0.5, 1.03), xycoords="axes fraction", ha="center")
            ann1 = axes[1].annotate(f"Clusters " + str(i), (0.5, 1.03), xycoords="axes fraction", ha="center")

            axes[0].set_ylim(bottom=0, top=max_range)
            axes[0].set_ylabel('x-range [m]')
            axes[0].set_xlim(left=-10, right=10)
            axes[0].set_xlabel('y-range [m]')
            axes[0].grid(visible=True)

            axes[1].set_ylim(bottom=0, top=max_range)
            axes[1].set_ylabel('x-range [m]')
            axes[1].set_xlim(left=-10, right=10)
            axes[1].set_xlabel('y-range [m]')
            axes[1].grid(visible=True)

            if not saveVideo:
                p0 = axes[0].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['doppler'],
                                     vmin=-max_doppler, vmax=max_doppler, marker='o', s=20, cmap=cm.jet)
                p1 = axes[1].scatter(detObj2D_nf['location_x'], detObj2D_nf['location_y'], c=detObj2D_nf['cluster'], marker='o', s=20, cmap=cm.jet)
                cb0 = plt.colorbar(p0, ax=axes[0])
                plt.pause(0.01)
                axes[0].clear()
                axes[1].clear()
                cb0.remove()
            else:
                p0 = axes[0].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['doppler'],
                                     vmin=-max_doppler, vmax=max_doppler, marker='o', s=5, cmap=cm.jet)
                p1 = axes[1].scatter(detObj2D_nf['location_x'], detObj2D_nf['location_y'], c=detObj2D_nf['cluster'],
                                     marker='o', s=5, cmap=cm.jet)
                p = [p0, p1, ann0, ann1]
                ims.append(p)

        elif plot2DXYdopplervar:
            ann0 = axes[0].annotate(f"XY scatter " + str(i), (0.5, 1.03), xycoords="axes fraction", ha="center")
            ann1 = axes[1].annotate(f"Clusters " + str(i), (0.5, 1.03), xycoords="axes fraction", ha="center")

            axes[0].set_ylim(bottom=0, top=max_range)
            axes[0].set_ylabel('x-range [m]')
            axes[0].set_xlim(left=-10, right=10)
            axes[0].set_xlabel('y-range [m]')
            axes[0].grid(visible=True)

            axes[1].set_ylim(bottom=0, top=max_range)
            axes[1].set_ylabel('x-range [m]')
            axes[1].set_xlim(left=-10, right=10)
            axes[1].set_xlabel('y-range [m]')
            axes[1].grid(visible=True)

            if not saveVideo:
                p0 = axes[0].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['doppler'],
                                     vmin=-max_doppler, vmax=max_doppler, marker='o', s=20, cmap=cm.jet)
                p1 = axes[1].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['variance_D'],
                                     marker='o', s=20, cmap=cm.jet)
                cb0 = plt.colorbar(p0, ax=axes[0])
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
                cb0.remove()
            else:
                p0 = axes[0].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['doppler'],
                                     vmin=-max_doppler, vmax=max_doppler, marker='o', s=5, cmap=cm.jet)
                p1 = axes[1].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['variance_D'],
                                     marker='o', s=5, cmap=cm.jet)
                p = [p0, p1, ann0, ann1]
                ims.append(p)

        else:
            sys.exit("Unknown plot options.")

    if saveVideo:
        cb = fig.colorbar(p0, ax=axes[0])
        ani = animation.ArtistAnimation(fig, ims, repeat=False, blit=True)
        ani.save(datapath + r'konzept2\\auswertung_scale_1_05Zwischenpräsentation.mp4', dpi=200, fps=1 / 0.035)


if __name__ == '__main__':
    main()
