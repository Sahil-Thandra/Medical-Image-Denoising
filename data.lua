local file = require('pl.file')
require 'torch'
require 'image'
require 'nnx'

imagesAll = torch.Tensor(722,1,64,64)
labelsAll = torch.Tensor(722,1,64,64)

--Loading images from the DX Dataset
for f=1, 400 do
	imagesAll[f] = image.load('images/DX/Noisy/'..f..'.pgm')
	labelsAll[f] = image.load('images/DX/Original/'..f..'.pgm')
end

--Loading images from the MMM Dataset
for f=1, 322 do
	imagesAll[400+f] = image.load('images/MMM/Noisy/mdb'..f..'.pgm')
	labelsAll[400+f] = image.load('images/MMM/Original/mdb'..f..'.pgm')
end

local labelsShuffle = torch.randperm((#labelsAll)[1])

local trsize = 650
local tesize = labelsShuffle:size(1) - trsize

--Function to load training data
local function traindataset()
	local x = torch.Tensor(trsize, 1, 64, 64)
	local y = torch.Tensor(trsize, 1, 64, 64)
	for i=1,trsize do
   	x[i] = imagesAll[labelsShuffle[i]]:clone()
   	y[i] = labelsAll[labelsShuffle[i]]:clone()
	end
	return x,y
end

--Function to load testing data
local function testdataset()
	local x = torch.Tensor(tesize, 1, 64, 64)
	local y = torch.Tensor(tesize, 1, 64, 64)
	for i=trsize+1,tesize+trsize do
   	x[i-trsize] = imagesAll[labelsShuffle[i]]:clone()
   	y[i-trsize] = labelsAll[labelsShuffle[i]]:clone()
	end
	return x,y
end

return {traindataset = traindataset,
        testdataset=testdataset}
