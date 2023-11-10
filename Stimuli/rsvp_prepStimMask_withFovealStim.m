function rsvp_prepStimMask_withFovealStim()

% This function creates the stimulus sequence (and sequence timing) and the
% image mask set required for the pRF model.

%%% Created by MasR - 11/16/2020

%% set parameters
p.trDur = 1.3;

p.sweepNum = 8;
p.stepsPerSweep = 12;

p.trs_initDummy = 1;
p.trs_preSweep = 2;
p.trs_stayAtEachLoc = 2;
p.trs_sweepLen = p.stepsPerSweep * p.trs_stayAtEachLoc;
p.trs_endDummy = 0;
p.runLen = p.trs_initDummy + p.sweepNum *(p.trs_preSweep + p.trs_sweepLen) + p.trs_endDummy;

p.sweepDirSeq = [1 2 3 4 1 2 3 4];% 1:LR , 2:TB , 3:RL , 4:BT
p.imSize = 120;

%% stimulus (sequence and sequence timings)
stimulus.seq = [1:p.runLen];
stimulus.seqtiming = p.trDur*(stimulus.seq-1);
save('bar_stimulus_masks_1300ms_params','stimulus');

%% images 
% Here, we create a set 3D matrix corresponding to each sweep direction
% This image set contains all image(what the subject sees on the display
% during the experiment) binary masks. The image set  contains one image
% per TR (trs go along the 3rd dimension).  
p.barWidth = p.imSize / p.stepsPerSweep;
p.barCenters = [p.barWidth/2:p.barWidth:p.imSize-p.barWidth/2];

%%%% Left to right sweeps
for stepInd = 1:p.trs_stayAtEachLoc:p.trs_stayAtEachLoc*p.stepsPerSweep
    thisIm = zeros(p.imSize);
    tmpInd = ceil(stepInd/p.trs_stayAtEachLoc);
    OnPixInds = (p.barCenters(tmpInd)-p.barWidth/2)+1:(p.barCenters(tmpInd)+p.barWidth/2);
    thisIm(:,OnPixInds) = 1;  
    imSet.lr(:,:,stepInd:stepInd+p.trs_stayAtEachLoc-1) = repmat(thisIm,[1 1 p.trs_stayAtEachLoc]);
end

%%%% Right to left sweeps
for stepInd = 1:size(imSet.lr,3)
    imSet.rl(:,:,stepInd) = fliplr(imSet.lr(:,:,stepInd));
end

%%%% Top to bottom sweeps
for stepInd = 1:p.trs_stayAtEachLoc:p.trs_stayAtEachLoc*p.stepsPerSweep
    thisIm = zeros(p.imSize);
    tmpInd = ceil(stepInd/p.trs_stayAtEachLoc);
    OnPixInds = (p.barCenters(tmpInd)-p.barWidth/2)+1:(p.barCenters(tmpInd)+p.barWidth/2);
    thisIm(OnPixInds,:) = 1;  
    imSet.tb(:,:,stepInd:stepInd+p.trs_stayAtEachLoc-1) = repmat(thisIm,[1 1 p.trs_stayAtEachLoc]);
end

%%%% Bottom to top sweeps
for stepInd = 1:size(imSet.tb,3)
    imSet.bt(:,:,stepInd) = flipud(imSet.tb(:,:,stepInd));
end

imSet_all{1} = imSet.lr;
imSet_all{2} = imSet.tb;
imSet_all{3} = imSet.rl;
imSet_all{4} = imSet.bt;

%% make the whole array of images
% Here we concatenate all sweeps and non-sweep(dummy) images to make the
% image mask corresponding to the whole run.
images = zeros(p.imSize,p.imSize,p.runLen);
images(:,:,1:p.trs_initDummy) = zeros(p.imSize);
image_foveal = zeros(p.imSize);
maskInds_foveal = (p.imSize-p.imSize/6)/2:(p.imSize+p.imSize/6)/2;
image_foveal(maskInds_foveal,maskInds_foveal) = 1;
for sweepInd = 1:p.sweepNum
    thisSweepInd0 = p.trs_initDummy + (sweepInd-1)*(p.trs_preSweep + p.trs_sweepLen);
    thisSweepInd1 = thisSweepInd0 + p.trs_preSweep;
    thisSweepInd2 = thisSweepInd1 + p.trs_sweepLen;
    images(:,:,thisSweepInd0+1:thisSweepInd1-1) = zeros(p.imSize,p.imSize,p.trs_preSweep-1);
    images(:,:,thisSweepInd1) = image_foveal;
    images(:,:,thisSweepInd1+1:thisSweepInd2) = imSet_all{p.sweepDirSeq(sweepInd)};
end
images = cat(3,images,zeros(p.imSize,p.imSize,p.trs_endDummy));
save('bar_stimulus_masks_1300ms_images','images');