%% This file has been included in lc_createImages.m
%% Experiments to show some possible sensor parameters
%
% Interesting features
%   1.  CLassify directly from the pixel responses, no image processing
%   2.  Analyze the tradeoff between pixel size and light level for a given
%   classification performance
%
%   
%
ieInit

%%  Diffraction limited optics
s = sceneCreate('rings rays');
s = sceneSet(s,'fov',1);
s = sceneAdjustLuminance(s,100);
ieAddObject(s); sceneWindow;

oi = oiCreate;
oif2 = oiSet(oi,'optics fnumber',4);
oif2 = oiSet(oif2,'optics off axis method','skip');

oif2 = oiCompute(oif2,s);
ieAddObject(oif2); oiWindow;


%% Here is how we might change the f-number of the optics

% oif16 = oiSet(oi,'optics fnumber',16);
% oif16 = oiCompute(oif16,s);
% ieAddObject(oif16); oiWindow;


%%

sensor = sensorCreate;
sensor = sensorSet(sensor,'noise flag',2);

sensor = sensorSet(sensor,'exposure duration',1/60);  % 15 ms

sensor = sensorSet(sensor,'size',[144 176]);
sensor = sensorCompute(sensor,oif2);
ieAddObject(sensor); sensorWindow('scale',true);

%% Here is how to set the pixel size to be smaller

sensor = sensorSet(sensor,'pixel size constant fill factor',1.4e-6);
sensor = sensorCompute(sensor,oif2);

ieAddObject(sensor); sensorWindow('scale',true);

%%

sensor = sensorSet(sensor,'pixel size constant fill factor',1.1e-6);
sensor = sensorCompute(sensor,oif2);

ieAddObject(sensor); sensorWindow('scale',true);

%% Not important for you, but you might classify on this if you prefer

% ip = ipCreate;
% ip = ipCompute(ip,sensor);
% ieAddObject(ip); ipWindow;

%% Little timing check


%%
% clear I
% ieSessionSet('wait bar','off');
% 
% img = rand(28,28); 
% for ii=1:3, I(:,:,ii) = img; end
% scene = sceneFromFile(I,'rgb');
% 
% tic
% for ii=1:16
%     oi = oiCompute(oi,scene);
%     sensor = sensorCompute(sensor,oi);
% end
% toc
% 
% %%
% clear I
% img = rand(4*28,4*28); for ii=1:3, I(:,:,ii) = img; end
% scene = sceneFromFile(I,'rgb');
% 
% tic
% oi = oiCompute(oi,scene);
% sensor = sensorCompute(sensor,oi);
% toc
% 
% %% Suppose your
% 
% for ii=1:256
%     I = imread('your image');
%     bigI(theseRows,theseCols) = I;
% end
% 
% 
