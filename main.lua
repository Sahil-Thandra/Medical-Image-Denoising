require('nngraph')
require('dpnn')
require('optim')
data = require('data')

gpu=1
if gpu>0 then
  print("CUDA ON")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(gpu)
end

--Parameters
nIters = 100
nPredict = 72
batchSize = 10   
lr = 0.001

--Denoising Convolutional Auto-Encoder Architecture
model = nn.Sequential()
	:add(nn.SpatialConvolution(1,64,5,5,1,1,2,2))
	:add(nn.SpatialMaxPooling(2,2,2,2))	
	:add(nn.SpatialConvolution(64,64,5,5,1,1,2,2))
	:add(nn.SpatialMaxPooling(2,2,2,2))
	:add(nn.SpatialConvolution(64,64,5,5,1,1,2,2))
	:add(nn.SpatialUpSamplingNearest(2))
	:add(nn.SpatialConvolution(64,64,5,5,1,1,2,2))
	:add(nn.SpatialUpSamplingNearest(2))
	:add(nn.SpatialConvolution(64,1,5,5,1,1,2,2))

print(model)


if gpu>0 then
	model = model:cuda()
end

--Loss Function
criterion = nn.CrossEntropyCriterion()

if gpu>0 then
  criterion=criterion:cuda()
end

paramx,paramdx = model:getParameters()

--Loading the training and testing data
train_data,train_labels = data.traindataset()
test_data,test_labels = data.testdataset()

if gpu>0 then
  train_data = train_data:cuda()
  train_labels = train_labels:cuda()
  test_data = test_data:cuda()
  test_labels = test_labels:cuda()
end

--Function to convert tensor to a table
function tensor2Table(inputTensor)
   local outputTable = {}
   for t = 1, inputTensor:size(1) do outputTable[t] = inputTensor[t] end
   return outputTable
end

--Function to convert table to a tensor
function table2tensor(tx)
  merge = nn.Sequential()
    :add(nn.JoinTable(1))
    :add(nn.View(-1,64,64))

  return merge:forward(tx)
end

--Function to evaluate loss and gradients
function feval()

		model:training()
		model:zeroGradParameters()
    outputs = model:forward(inputs)
    	if gpu>0 then
    	  outputs = outputs:cuda()
    	  targets = targets:cuda()
    	end
    err = criterion:forward(outputs, targets)
    gradOutputs = criterion:backward(outputs, targets)    
    model:backward(inputs, gradOutputs)
    
  return err, paramdx
end 

--Training the model
trainError = 0
iteration = 1
while iteration <= nIters do
   
  offsets = {}

  for i=1,batchSize do
    s = ((10*(iteration-1))%650)+i     
	table.insert(offsets, s)
  end


  offsets = torch.Tensor(offsets)
  if gpu>0 then
      offsets=offsets:cuda()
  end

   inputs={}
   targets={}
   inputs = train_data:index(1, offsets):view(batchSize,1,64,64)
   targets = train_labels:index(1, offsets):view(batchSize,1,64,64)
   for j=1,batchSize do
      if offsets[j] > train_data:size(1) then
         offsets[j] = 1
      end
   end
  

      
  paramx, Error = optim.adam(feval,paramx)
  trainError = trainError + Error[1]

  print(string.format("Iteration %d ; AdamOpt err = %f ", iteration, Error[1]))

  iteration = iteration + 1
end
trainError = trainError/nIters
print(trainError)

--Saving the trained model
paths = require 'paths'
filename  = paths.concat(paths.cwd(),'Frame.net')
torch.save(filename, model)

--Evaluating the trained model and saving the output images
model:evaluate()
iters = 1

while iters <=nPredict do
	test_table={}
	table.insert(test_table,iters)	
	test_table = torch.Tensor(test_table)
  	if gpu>0 then
      test_table=test_table:cuda()
  	end		
  	inputs_test, targets_test = {}, {}
    inputs_test = test_data:index(1, test_table):view(1,64,64)
    targets_test = test_labels:index(1, test_table):view(1,64,64)
    
    model:zeroGradParameters()
    outputs_test = model:forward(inputs_test)

    image.save('Result/Input'..iters..'.pgm',targets_test)		
    image.save('Result/Output'..iters..'.pgm',outputs_test)
end    





