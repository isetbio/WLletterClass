%% Experiments to show some possible sensor parameters

ieInit

%%  Diffraction limited optics
s = sceneCreate('rings rays');
s = sceneSet(s,'fov',1);
s = sceneAdjustLuminance(s,100);
ieAddObject(s); sceneWindow;

oi = oiCreate;
oif2 = oiSet(oi,'optics fnumber',4);

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

ip = ipCreate;
ip = ipCompute(ip,sensor);
ieAddObject(ip); ipWindow;


