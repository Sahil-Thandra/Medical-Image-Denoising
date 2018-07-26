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
nPredict = 72
batchSize = 10   
lr = 0.001

--Loading the testing data
test_data,test_labels = data.test_dataset()

if gpu>0 then
  test_data = test_data:cuda()
  test_labels = test_labels:cuda()
end

--Loading the trained model
paths = require 'paths'
filename  = paths.concat(paths.cwd(),'vTrain.net')
model = torch.load(filename)

print(model)

if gpu>0 then
  model=model:cuda()
end

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

    image.save('Result/Input1'..iters..'.png',targets_test)		
    image.save('Result/Output1'..iters..'.png',outputs_test)
end    