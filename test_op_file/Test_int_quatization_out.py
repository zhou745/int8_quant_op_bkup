import mxnet as mx
import time
import random
def quantization_int8_test_module():
    data = mx.symbol.Variable('data')
    
    internal = mx.symbol.Quantization_int8(data,is_weight=False,ema_decay=0.99,delay_quant=10)
    #internal = mx.symbol.Moving_average(data=data)
    #mu = mx.sym.mean(data=internal,axis=(1,),exclude=True)
   
  
    #mu = mx.sym.reshape(mu,shape=(1,3,1))
    #internal = mx.sym.broadcast_minus(internal,mu)

    #internal = mx.symbol.relu(internal)
    #internal = -internal+6
    #internal = mx.symbol.relu(internal)
    #internal = -internal+6
    output = mx.symbol.make_loss(data=internal)
    return(output)


net = quantization_int8_test_module()

data = mx.nd.array([[[[random.uniform(0.,1.) for i in range(3)] for j in range(3)] for k in range(256)] for l in range(256)])


net_run = net.simple_bind(ctx=mx.gpu(),data=data.shape)
net_run_cpu = net.simple_bind(ctx=mx.cpu(),data=data.shape)
t1=time.time()
for repeat in range(10):
    net_run.forward(is_train=True,data=data)
    mx.ndarray.waitall()

t2=time.time()

print("time used is %f"%(t2-t1))
#net_run_cpu.forward(is_train=True,data=data,mu=mu)
#net_run_cpu.backward()

#result = net_run.outputs[0].asnumpy()
#for i in range(128):
#    print(result[i*32:i*32+32])
#print(data.asnumpy())
#print(net_run.outputs[0].asnumpy())

#print(net_run.grad_arrays)

#print(net_run_cpu.outputs[0].asnumpy())
#print(net_run_cpu.grad_arrays)
