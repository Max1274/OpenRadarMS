import time
import numpy as np
import mmwave.dsp as dsp
import matplotlib.pyplot as plt
import mmwave.clustering as clu
from matplotlib import cm
from matplotlib import animation
from mmwave.dsp.utils import Window
from mmwave.dataloader.adc import DCA1000
from oscfar import os_cyth
from music import music_cyth
import gc
import math
import cv2

start_whole = time.time()

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

max_range = 18
max_doppler = 7

print("range resolution: " + str(range_resolution))
print("doppler resolution: " + str(doppler_resolution))

plot2DXYdoppler = False
plotmusic = False
plotFelgen = True
saveVideo = False

useMUSIC = True


def main():
    ims = []

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
    elif plotmusic:
        fig, axes = plt.subplots(1, 2)
    elif plotFelgen:
        fig, axes = plt.subplots(1, 2)

    num_vec, steering_vec = dsp.gen_steering_vec(90, 0.5, 8)

    for i, frame in enumerate(adc_data[50:500]):  # leicht_schraeg:50, gerade_vor:120, 90grad:50
        #        print(i,end=',') # Frame tracker
        # (2) Range Processing
        start_all = time.time()

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BARTLETT)
        assert radar_cube.shape == (
            numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=2, clutter_removal_enabled=False,
                                                       window_type_2d=Window.HAMMING)

        '''music_cube = dsp.separate_tx(radar_cube, 2, vx_axis=1, axis=0)
        music_cube = np.transpose(music_cube, axes=(2, 1, 0))'''

        # (4) Detection of Reflections
        # --- CFAR, SNR is calculated as well.
        fft2d_sum = det_matrix.astype(np.int16)

        start = time.time()
        thresholdRange, noiseFloorRange = os_cyth(fft2d_sum, guard_len=4, noise_len=10, scale=1.05, axis=1)
        thresholdDoppler, noiseFloorDoppler = os_cyth(fft2d_sum, guard_len=4, noise_len=20, scale=1.05, axis=0)
        print("CFAR time: " + str(time.time() - start))

        det_doppler_mask = (fft2d_sum > thresholdDoppler)
        det_range_mask = (fft2d_sum > thresholdRange)

        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # evaluate doppler variance for Micro-Doppler-recognition of wheels
        doppler_var = dsp.get_micro_doppler(aoa_input, numLoopsPerFrame, det_peaks_indices)
        doppler_var_sum_angle = np.sum(doppler_var, axis=1)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        # snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        dtype_location = '(' + str(numTxAntennas) + ',)<f4'
        dtype_detObj2D = np.dtype(
            {'names': ['rangeIdx', 'dopplerIdx', 'azimuth', 'peakVal', 'location', 'SNR', 'variance_D'],
             'formats': ['<i4', '<i4', '<i4', '<f4', dtype_location, '<f4', '<f4']})
        detRefl = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detRefl['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detRefl['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detRefl['peakVal'] = peakVals.flatten()
        detRefl['variance_D'] = doppler_var_sum_angle[detRefl['rangeIdx'], detRefl['dopplerIdx']]

        #min. 0.5m Abstand
        detRefl = detRefl[detRefl['rangeIdx']>(0.5/range_resolution)]


        Refl = detRefl
        start = time.time()
        n_insertions = 0
        numnDets = detRefl.shape[0]

        for refs in range(numnDets):
            # MUSIC
            aoa_vector = aoa_input[detRefl[refs]['rangeIdx'], :, detRefl[refs]['dopplerIdx']]
            if useMUSIC:
                aoa_spectrum = music_cyth(aoa_vector, steering_vec, 1)
            else:
                aoa_spectrum = dsp.aoa_bartlett_ms(steering_vec, aoa_vector, 0)
            peak_bin = aoa_spectrum.argmax()
            Refl[refs + n_insertions]['azimuth'] = -(
                    peak_bin * 0.5 - 90)  # 0.5: Auflösung steering_vector, 90 Grad, um Bereich -90° ... 90° abzudecken

            # adjust dopplerIdx to take negative values into account
            if Refl[refs]['dopplerIdx'] >= numLoopsPerFrame / 2:
                Refl[refs]['dopplerIdx'] = Refl[refs]['dopplerIdx'] - numLoopsPerFrame

        print("MUSIC time: " + str(time.time() - start))
        numDetObjs = Refl.shape[0]
        dtf = np.dtype({'names': ['location_x', 'location_y', 'azimuth', 'doppler', 'cluster', 'variance_D'],
                        'formats': ['<f4', '<f4', '<f4', '<f4', '<i4', '<f4']})
        detObj2D = np.zeros((numDetObjs,), dtype=dtf)

        detObj2D['location_x'] = Refl['rangeIdx'] * range_resolution * np.sin(Refl['azimuth'] / 180 * np.pi)
        detObj2D['location_y'] = Refl['rangeIdx'] * range_resolution * np.cos(Refl['azimuth'] / 180 * np.pi)
        detObj2D['azimuth'] = Refl['azimuth']
        detObj2D['doppler'] = Refl['dopplerIdx'] * doppler_resolution
        detObj2D['variance_D'] = Refl['variance_D']

        if len(detObj2D) > 0:
            cluster = clu.radar_dbscan_ms(detObj2D, 4, eps=4,
                                          min_samples=7)  # cluster: [numPoints, center, size, avg_vel]
            Felgenpunkte = detObj2D
            for j in reversed(range(numDetObjs)):
                delete = False
                if Felgenpunkte['cluster'][j] == -1:
                    delete = True
                # elif cluster['size'][Felgenpunkte['cluster'][j]][1] < 1:
                #   delete = True
                if Felgenpunkte['variance_D'][j] < 0.25:  # 0.25; 0.4
                    delete = True
                if delete == True:
                    Felgenpunkte = np.delete(Felgenpunkte, j, axis=0)

            """if len(Felgenpunkte) > 1:
                cluster_rad = clu.radar_dbscan_ms(Felgenpunkte, 0, eps=0.5, min_samples=1)
                if len(cluster_rad) == 2:
                    dist = math.sqrt(math.pow(cluster_rad[0]['center'][0] - cluster_rad[1]['center'][0], 2) + \
                                     math.pow(cluster_rad[0]['center'][1] - cluster_rad[1]['center'][1], 2))
                    if dist < 1.8:  # kurze Seite #TODO: implement angle detection
                        alpha = 1
                    else:
                        alpha = 2

                if len(cluster_rad) == 3 or len(cluster_rad) == 4:
                    rect = cv2.minAreaRect(cluster_rad['center'])"""

        if plotmusic:
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
                p0 = axes[0].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['doppler'],
                                     vmin=-max_doppler, vmax=max_doppler, marker='o', s=2, cmap=cm.jet)
                p1 = axes[1].scatter(detObj2D['azimuth'], detObj2D['doppler'], marker='o', s=2, c='#1f77b4')
                p = [p0, p1, ann0, ann1]
                ims.append(p)

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
                p0 = axes[0].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['variance_D'],
                                     vmin=0, vmax=1, marker='o', s=20, cmap=cm.jet)
                p1 = axes[1].scatter(detObj2D['azimuth'], detObj2D['doppler'], marker='o', s=20)
                cb = plt.colorbar(p0, ax=axes[0])
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
                cb.remove()
            else:
                p0 = axes[0].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['variance_D'],
                                     vmin=0, vmax=1, marker='o', s=2, cmap=cm.jet)
                p1 = axes[1].scatter(detObj2D['azimuth'], detObj2D['doppler'], marker='o', s=2, c='#1f77b4')
                p = [p0, p1, ann0, ann1]
                ims.append(p)

        elif plotFelgen:
            ann0 = axes[0].annotate(f"XY scatter " + str(i), (0.5, 1.03), xycoords="axes fraction", ha="center")
            ann1 = axes[1].annotate(f"extracted wheels " + str(i), (0.5, 1.03), xycoords="axes fraction", ha="center")

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
                p1 = axes[1].scatter(Felgenpunkte['location_x'], Felgenpunkte['location_y'], marker='o', s=20)
                cb = plt.colorbar(p0, ax=axes[0])
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
                cb.remove()
            else:
                p0 = axes[0].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['doppler'],
                                     vmin=-max_doppler, vmax=max_doppler, marker='o', s=2, cmap=cm.jet)
                p1 = axes[1].scatter(Felgenpunkte['location_x'], Felgenpunkte['location_y'], marker='o', s=2,
                                     c='#1f77b4')
                p = [p0, p1, ann0, ann1]
                ims.append(p)

        end = time.time()
        print("execution time " + str(i) + ": " + str(end - start_all))

    if saveVideo:
        cb = fig.colorbar(p0, ax=axes[0])
        ani = animation.ArtistAnimation(fig, ims, repeat=False, blit=True)
        ani.save(datapath + 'auswertung_MUSIC_07_15_2Dfft.mp4', dpi=200, fps=1 / 0.035)

    print("TOTAL time " + str(i) + ": " + str(time.time() - start_whole))


if __name__ == '__main__':
    main()
