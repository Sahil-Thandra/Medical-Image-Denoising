require('nngraph')
require('dpnn')
require('optim')
data = require('vdata')

gpu=1
if gpu>0 then
  print("CUDA ON")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(gpu)
end

--Parameters
nIters = 100
batchSize = 10   
lr = 0.001

--Loading the training data
train_data,train_labels = data.traindataset()

if gpu>0 then
  train_data = train_data:cuda()
  train_labels = train_labels:cuda()
end

--Loading the pre-trained model
paths = require 'paths'
filename  = paths.concat(paths.cwd(),'Frame.net')
model = torch.load(filename)

print(model)

if gpu>0 then
  model=model:cuda()
end

--Acquiring the model parameters
paramx,paramdx = model:getParameters()

--Loss Function
criterion =nn.CrossEntropyCriterion()

if gpu>0 then
  criterion=criterion:cuda()
end

--Function to evaluate loss and gradients
function feval()

		model:training()
		model:zeroGradParameters()
		outputs_table = model:forward(inputs)
    	outputs = table2Tensor(outputs_table)
    	targets_tensor = table2Tensor(targets)
    	if gpu>0 then
    	  outputs = outputs:cuda()
    	  targets_tensor = targets_tensor:cuda()
    	end
    	err = criterion:forward(outputs, targets_tensor)
    	gradOutputs = criterion:backward(outputs, targets_tensor)    
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
   inputs = train_data:index(1, offsets):view(batchSize,64,64)
   targets = train_labels:index(1, offsets):view(batchSize,64,64)
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
filename  = paths.concat(paths.cwd(),'vTrain.net')
torch.save(filename, model)

