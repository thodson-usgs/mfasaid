
File ---------------------- SPOON001.arg
File Size (bytes) --------- 808923
Number of Samples --------- 5866
Time of first sample ------ 20/03/2013 14:30:14
Time of last  sample ------ 17/05/2013 09:15:14

ArgonautDP Hardware Configuration
-----------------------------------
ArgType ------------------- SL
SerialNumber -------------- E2418
Frequency ------- (kHz) --- 1500
Nbeams -------------------- 2
BeamGeometry -------------- 2_BEAMS
VerticalBeam -------------- YES
SlantAngle ------ (deg) --- 25.0
CPUSoftwareVerNum --------- 11.8
DSPSoftwareVerNum --------- 1.0
BoardRev ------------------ F
SensorOrientation---------- SIDE
CompassInstalled ---------- YES
RecorderInstalled --------- YES
TempInstalled ------------- YES
PressInstalled ------------ YES
CtdSensorInstalled -------- NO
YsiSensorInstalled -------- NO
Ext Press Sensor ---------- NONE
TempOffset (deg C) -------- 0.00
TempScale  (deg C/deg C) -- 1.0000
PressOffset (dbar) -------- -1.265820
PressScale  (dbar/count) -- 0.000205
PressScale_2 (pdbar/c^2) -- 57.000000
Transformation Matrix -----    1.183  -1.183
                               0.552   0.552

Argonaut User Setup
---------------------
DefaultTemp ----- (deg C) -- 20.00
DefaultSal ------ (ppt) ---- 0.00
TempMode ------------------- MEASURED
DefaultSoundSpeed (m/s) ---- 1482.30
CellBegin ------- (m) ------ 1.00
CellEnd --------- (m) ------ 18.50
BlankDistance---- (m) ------ 1.00
CellSize -------- (m) ------ 1.75
Number of Cells ------------ 10
ProfilingMode -------------- YES
DynBoundaryAdj ------------- NO
WaveSpectra ---------------- NO
WaterDepth ------ (m) ------ 0.00
AvgInterval ----- (s) ------ 90
SampleInterval -- (s) ------ 900
YsiBufferSize --- (bytes) -- 0
BurstMode ------------------ DISABLED
BurstInterval --- (s) ------ 1200
SamplesPerBurst ------------ 1
CoordSystem ---------------- XYZ
AutoSleep ------------------ YES
Voltage Protection---------- YES
One Beam Solution----------- NO
Check for Ice Coverage------ NO
OutMode -------------------- AUTO
OutFormat ------------------ ENGLISH
DataFormat ----------------- LONG FORMAT
RecorderEnabled ------------ ENABLED
RecorderMode --------------- NORMAL
DeploymentName ------------- SPOON
DeploymentStart Date/Time -- 17/04/2012 14:38:29
Comments:




---------------------------------------------------
Argonaut ASCII data file Long format is as follows:
---------------------------------------------------
Column  1- 6: Year Month Day Hour Minute Second;
Column  7- 8: WaterVel1/X/E WaterVel2/Y/N (cm/s)
Column  9   : WaterLevel (m)
Column 10-12: VelStDev1/X/E VelStDev2/Y/N VelStDev3/Z/U (cm/s)
Column 13-15: SNR1          SNR2          SNR3          (dB);
Column 16-18: SignalAmp1    SignalAmp2    SignalAmp3    (counts);
Column 19-21: Noise1        Noise2        Noise3        (counts);
Column    22: Ice Detection
Column 23-25: Heading Pitch Roll (deg);
Column 26-28: Standard deviation of the Heading Pitch Roll (deg);
Column 29-30: Mean Tempr (degC) MeanPress (dBar);
Column    31: StDevPress (dBar);
Column    32: Power level (battery voltage) (Volts);
Column 33-34: CellBegin CellEnd (m);
Column    35: Speed (cm/s);
Column    36: Direction (deg);
Column    37: Internal Flow - Area (m3);
Column    38: Internal Flow - Flow (m3/s);


------------------------------------
Flow data file format is as follows:
------------------------------------
Column  1- 6: Year Month Day Hour Minute Second;
Column     7: Depth (m)
Column     8: Area (m2)
Column     9: Vx (m/s);
Column    10: V Mean (m/s);
Column    11: Flow (m3/s);


----------------------------------------------------
Multi-Cell (Profiling) Data file formats as follows:
----------------------------------------------------
Velocity File (*.vel)       : Sample #, For each individual cell - Velocity X and Y, Speed and Direction;
Standard Error File (*.std) : Sample #, For each individual cell - Standard Error X and Y;
SNR File (*.snr)            : Sample #, For each individual cell - SNR and Amplitude for each beam;
