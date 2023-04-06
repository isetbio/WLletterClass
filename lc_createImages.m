%% Create images that will go into any classification tool
% For Caffe: creates a lmdb database that can be feed directly
% For tensorflow and nupic: it will store the data in png images:
% -- nupic has its own built-in functions to read and make sense of them
% -- for tensorflow I created some python scripts to read the data and
%    create the corresponding numpy arrays that go into tensorflow 
% TODO:
% 1.- Select some variables and create some demo png images to decide which
%     ones to use
% 1.1.- Use sensors defined in lc_sensorChoices as they are to check how
%       they look
% 1.2.- Use lighting values of [10, 50, 100]
% 1.3.- Change diferent pixel sizes [1.1e-06, 1.25e-06, 1.4e-06];
% 
% 2.- Refactor de code to remove commented blocks and create functions
% 
% 3.- Generate the images and check them to select best parameters for
%     simulation
% 
% 4.- 


%% INITIALIZE
ieInit;

% Where are the files to convert
projDir = '~/soft/WLletterClass';
mnistDirTest  = [projDir filesep 'data/origMnistSmall/test'];
mnistDirTrain = [projDir filesep 'data/origMnistSmall/train'];

mnistDir = mnistDirTrain;
testTrain = 'train';


%% EXP. VARIABLES
% We are going to change lighting and pixel size
CharNamesList = 3:6;
pixelSizes = [1.1e-06, 1.25e-06, 1.4e-06]; pSize = 2;
sceneLights = [45];
% fovs = [0.6, 0.8, 1];
dists_ft = [25]; 
dists = ft2m(dists_ft);

% mnistISETBIOdir = [mnistDir filesep 'OUT'];
% if ~isdir(mnistISETBIOdir)
%     mkdir(mnistISETBIOdir);
% end

%% EXP. OPTIONS
ViewResults = 0;  % Make it 1 to visualize results
if ViewResults
    CharNamesList = CharNamesList(6); % Visualize only one category for test
end

% For the SCIEN poster Brian created this LED display from Barco
% It can be out of the loop
ledBarcoDisplay = displayCreate('LED-BarcoC8');

%% LOOP TO PROCESS ALL IMAGES
for dist = 1:length(dists)
    for sLight = 1:length(sceneLights)
        for nc = 1:length(CharNamesList)
            cd([mnistDir filesep num2str(CharNamesList(nc))])
%             WriteFolder = [projDir filesep 'data' filesep 'isetMnist' ...
%                             num2str(fov)  filesep testTrain filesep ...
%                            'Light_' num2str(sceneLights(sLight)) ...
%                            '_pSize_' num2str(pixelSizes(pSize)) ...
%                            filesep num2str(CharNamesList(nc))];
            WriteFolder = [projDir filesep 'data'  ...
                           filesep 'Light_' num2str(sceneLights(sLight)) ...
                                   '_DistFt_' num2str(dists_ft(dist)) ...
                            filesep testTrain  ...
                           filesep num2str(CharNamesList(nc))];
            if ~isdir(WriteFolder)
                mkdir(WriteFolder);
            end
            % Read all the file names to convert
            sampleNames = dir('*.png');
            if ViewResults
                sampleNames = sampleNames(5:5); % Visualize one or two for testing
            end

            for npng = 1:length(sampleNames)

                % SCENE CREATION PER EVERY IMAGE
                % ------------------------------
                % CREATE access to the file
                fullFileName = [mnistDir filesep num2str(CharNamesList(nc)) ...
                                filesep sampleNames(npng).name];
                % fullFileName = vcSelectImage; % If you want to select manually
                % imgType  = 'rgb';
                imgType  = 'monochrome';

                % CREATE a Display: iPhone Retina Display 2012, dpi = 326
%                 retinaDisplay = displayCreate('LCD-AppleRetina2012');
                % I created LCD-AppleRetina2012 as a copy of LCD-Apple, so now I am
                % going to change the values that interest me. 
                % Steve Jobs said 10-12 inches, and 11 is the lenght of letter
%                 distin = 11;
%                 retinadpi = 326;

%                 retinaDisplay = displaySet(retinaDisplay, 'name', 'AppleRetina2012');
%                 retinaDisplay = displaySet(retinaDisplay, 'dpi', 326);
%                 retinaDisplay = displaySet(retinaDisplay, 'viewingdistance', in2m(distin));
                  ledBarcoDisplay = displaySet(ledBarcoDisplay, 'viewingdistance', dists(dist));

                % CREATE a scene for my files that will simulate
                % watching the mnist character at 11 inches in an iPhone
%                 [scene, I] = sceneFromFile(fullFileName, imgType, [], ...
%                              retinaDisplay, [], [], [], []);

%                 CREATE a scene for my files that will simulate watching the 
%                 characters at .6, .8, 1 fov, that translates to: 
                  s = sceneFromFile(fullFileName, imgType, [], ...
                             ledBarcoDisplay, [], [], [], []);
%                   distin = m2in(displayGet(ledBarcoDisplay, 'viewing distance'));



%                 % CONFIRM that it makes sense
%                 himgpix = 28; % Mnist image is 28x28 pixels
%                 vimgpix = 28;
% 
%                 % 28 dot / 326 dot per inch = dim in inch
% %                 himgin = himgpix / retinadpi;
% %                 vimgin = vimgpix / retinadpi;
%                 himgin = himgpix / displayGet(ledBarcoDisplay, 'dpi');
%                 vimgin = vimgpix / displayGet(ledBarcoDisplay, 'dpi');
%                 % dim in inch / dist in inch = tan(alpha)
%                 halpharad = atan(himgin / distin);
%                 valpharad = atan(vimgin / distin);
%                 % convert to degrees
%                 hwangulardeg = rad2deg(halpharad);
%                 vwangulardeg = rad2deg(valpharad);
% 
%                 degperdot = displayGet(ledBarcoDisplay, 'deg per dot');
%                 secperdot = degperdot * 60 * 60;
% 
%                 % Calc dpi with homework1 example
%                 if ~isequal(sprintf('%.2f',displayGet(ledBarcoDisplay, 'dpi')), ...
%                         sprintf('%.2f',space2dpi(secperdot, distin)))
%                     error('dpi is not the same, revise calculations')
%                 end
% 
%                 % Obtain horiz fov and compare with hwangulardeg
%                 if ~isequal(sprintf('%.2f',hwangulardeg), ...
%                             sprintf('%.2f',sceneGet(scene, 'wangular')))
%                     error('fov is not the same, revise calculations')
%                 end
% 
%                 % Obtain horiz width and compare with himgin
%                 if ~isequal(sprintf('%.2f',in2m(himgin)), ...
%                             sprintf('%.2f',sceneGet(scene, 'width')))
%                     error('width is not the same, revise calculations')
%                 end
                % 
%                 % View results when debuging
%                 if ViewResults
%                     vcNewGraphWin;
%                     sceneShowImage(scene);
%                 end

                %%  Diffraction limited optics
%                 s = sceneCreate('rings rays'); % BW used it as a demo
                % s = sceneFromFile(fullFileName, imgType);
                % s = sceneSet(s,'fov',fov);
                s = sceneAdjustLuminance(s,sceneLights(sLight));
                ieAddObject(s); 
                
                % View results when debuging
                if ViewResults
                    sceneWindow;
                end
                
                oi = oiCreate;
                oif2 = oiSet(oi,'optics fnumber', 4);
                oif2 = oiSet(oif2,'optics off axis method','skip');

                oif2 = oiCompute(oif2,s);
                ieAddObject(oif2); 
                % View results when debuging
                if ViewResults
                    oiWindow;
                end

                %% Here is how we might change the f-number of the optics
                %  Here is how to set the pixel size to be smaller
                %
                % (We are not doing it right now, first check other variables)
                % oif16 = oiSet(oi,'optics fnumber',16);
                % oif16 = oiCompute(oif16,s);
                % ieAddObject(oif16); oiWindow;

                sensor = sensorCreate;
                sensor = sensorSet(sensor,'noise flag',2);

                sensor = sensorSet(sensor,'exposure duration',1/60);  % 15 ms
                sensor = sensorSet(sensor,'pixel size constant fill factor',...
                                   pixelSizes(pSize));
                sensor = sensorSet(sensor,'size',[64 64]);
                sensor = sensorCompute(sensor,oif2);
                ieAddObject(sensor); 
                
                
                % View results when debuging
                if ViewResults
                    sensorWindow('scale',true);
                end



                
                
%                 % HUMAN OPTICS
%                 % ------------
%                 oi = oiCreate;
%                 oi = oiCompute(oi, scene);
%                 % 
%                 % Scene Options
%                 % 
%                 % 
%                 % 
%                 % View results when debuging
%                 if ViewResults
%                     vcNewGraphWin;
%                     oiShowImage(oi);
%                 end
% 
% 
%                 sensor = sensorCreate();
%                 sensor = sensorSet
% 
%                 % CONE ABSORPTION HUMAN RETINA
%                 % ----------------------------
%                 cones = sensorCreate('human');
%                 cones = sensorCompute(cones,oi);
%                 img = coneImageActivity(cones);
%                 % 
%                 % Scene Options
%                 % 
%                 % 
%                 % 
%                 % View results when debuging
%                 if ViewResults
%                     vcNewGraphWin;
%                     imagesc(img); axis off; axis image
%                     coneimg = cones.data.volts;
%                     figure(); imagesc(coneimg); colormap(gray);
%                 end


                % SAVE THE IMAGE IN OUT DIRECTORY
                % -------------------------------
                % imwrite(img, [WriteFolder filesep sampleNames(npng).name]);  
                % I think that if we write the image directly, we will 
                % imfinfo([WriteFolder filesep sampleNames(npng).name])
                % If I save the image like that, it has non meaningful information,
                % as every cone instead of one value takes 8x8 values. I am
                % interested only in taking the grey value of the central cone (I
                % am going to ignore the color as our image is b&w).

                
                
                
                
                
%                 I = mat2gray(cones.data.volts);
                I = mat2gray(sensor.data.volts);
                imwrite(I, [WriteFolder filesep sampleNames(npng).name], ...
                        'png', 'BitDepth',8, 'ColorType', 'grayscale');
                % imwrite(I, 'test.png', ...
                %       'png', 'BitDepth',8, 'ColorType', 'grayscale');
                %        imfinfo('test.png')    
                % A = imread('test.png');
                % imtool(A) >> Class = uint8, Type:intensity, Min-max: 0-255
                % size(A) = 72 x 88

            end
        end
    end
end
