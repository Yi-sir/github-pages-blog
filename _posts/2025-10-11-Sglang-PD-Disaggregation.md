 https://docs.sglang.ai/advanced_features/pd_disaggregation.html
## 0. Overview
### 0.1 Route
![](https://youke1.picui.cn/s1/2025/10/14/68edc85132dcd.png)
### 0.2 P&D Instance
![](https://youke1.picui.cn/s1/2025/10/14/68edc85137bc5.png)
## 1. 参数和设计
### 1.1 参考命令
- prefill node
	- disaggregation-ib-device \${device_name}
	- disaggregation-mode prefill
	- dist-init-addr \${prefill_master_ip}:\${prefill_master_port}
	- nnodes/node-rank/tp-size/dp-size...
- decode node
	- disaggregation-ib-device \${device_name}
	- disaggregation-mode decode
	- dist-init-addr \${decode_master_ip}:\${decode_master_port}
	- nnodes/node-rank/tp-size/dp-size...
### 1.1.1 Advanced Features
https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/router.md
参考这里，可以使用如下命令来启动PD分离任务
```shell
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://prefill1:8000 9000 \
    --prefill http://prefill2:8001 9001 \
    --decode http://decode1:8002 \
    --decode http://decode2:8003 \
    --prefill-policy cache_aware \
    --decode-policy round_robin
```
这个方法也会启动http_server，router默认是走rust
router具体的实现好像在```sgl-router/src/lib.rs```
### 1.2 核心设计
- 原本的scheduling event loop基础上，增加了non-blocking sender & receiver operations
	- Prefill Server
		- Bootstrap Queue ```python/sglang/srt/managers/scheduler.py +909```
			- 给每个请求初始化一个sender
			- 用队列存储未完成bootstrap（握手和预分配内存）的请求
			- poll senders来检查bootstrap状态
			- 一旦某个bootstrap完成了，把请求移动到waiting queue
		- Waiting Queue
			- 用PrefillAdder来pop请求
			- 跑forward
			- 把请求添加到Infight Queue
		- Infight Queue ```python/sglang/srt/managers/scheduler.py +932```
			- 非阻塞地poll sender
			- 一旦某个transfer完成了，返回请求
	- Decode Server
		- PreallocQueue ```python/sglang/srt/managers/scheduler.py +871```
			- 给每个请求初始化一个receiver
			- 请求先握手，然后如果有available kv的话，会pre-allocate
			- 请求移动到TransferQueue
		- TransferQueue ```python/sglang/srt/managers/scheduler.py +861```
			- poll receiver，来检查transfer状态
			- 如果transfer完成了，请求移动到waiting queue
		- WaitingQueue
			- 用队列里的请求构造PrebuiltExtendBatch
			- 跳过prefill forward，只填充metadata
		- RunningBatch
			- 把解析出来的PrebuiltExtendBatch合并进running batch，跑decode
- 所以总体上，操作顺序是：
	- 首先sender和receiver握手，然后decode server调用pre-allocation，完成后通知prefill server做forward，然后KV transfer，decode server再做decode
## 2. Engine
下面小节顺序就是初始化的顺序
### 2.1 Tokenizer Manager
```python
# python/sglang/srt/entrypoints/http_server.py +1254
```
- **只有node rank 0上有tokenzier manager，因此全局也只有一个bootstrap server** python/sglang/srt/entrypoints/engine.py +833
- 在tokenizer manager初始化过程中构造bootstrap server ```python/sglang/srt/managers/tokenizer_manager.py +310```
	- 只在prefill节点上构造
	- 实际是一个```MooncakeKVBootstrapServer```对象```python/sglang/srt/disaggregation/mooncake/conn.py```，构造传入了host(127.0.0.1 by default)和disaggregation_bootstrap_port(8998 by default)
	- 注册health请求和用于注册prefill节点的put请求、用于decode节点连接prefill节点的get请求
	- 构造完成时，bootstrap server即开启工作线程，运行其event loop
### 2.2 DP Controller
```python
# python/sglang/srt/entrypoints/engine.py +827
# python/sglang/srt/managers/data_parallel_controller.py
```
- **每个node上只有一个dp controller**
	- **只有node 0持有request dispatcher** ```python/sglang/srt/managers/data_parallel_controller.py +150```
	- 先根据输入的dp size参数初始化dp size个zmq，key是scheduler_input_ipc_name，用来从tokenizer向scheduler发请求
- 以nnodes2，tpsize16，dpsize8为例
	- 先考虑每个tp组有几个节点，这里其实是优先划分pp的逻辑。由于没有pp，所以nnodes_per_tp_group = 2 // 1 = 2
	- 考虑每个node上的tp size，tp_size_per_node = 16 // 2 = 8
	- 计算当前node上的tp rank range，假设当前为node0，那么tp_rank_range = range(8 * (0%2)， 8 * (0%2+1)) = range(0, 8)
	-
	- 然后考虑pp。这里不涉及pp，size是1，rank是0。
	-
	- 对**每个local tp rank**做操作  ```python/sglang/srt/managers/dp_parallel_controller.py +280```
		- 计算dp attention相关参数
			- attn_tp_size = tp_size // dp_size = 2
			- attn_dp_rank = tp_rank(local) // attn_tp_size = \[0, 3]
			- attn_tp_rank = tp_rank % attn_tp_size = 0 or 1
		- 初始化一组pipe，分别是reader和writer
			- reader在controller里用，writer送给了scheduler进程。用来做父子进程间通信，只通信一次
		- 初始化当前dp rank使用的zmq ports。其中**scheduler_input_ipc_name**是**tcp://{dist_init_host}:{port_base+4+1+dp_rank}**，nccl_port和tp group保持一致
			- 这里的scheduler_input_ipc_name和node 0上request dispatcher的zmq key一致，二者是push和pull的关系
		- 计算当前rank使用的gpu id（这里可以有起始gpu id和gpu step参数）
		- 计算当前rank对应的moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
			- 注意，这里启动server时可以不指定ep size。只要moe_a2a_backend是deepep，就会设置ep_size = tp_size（python/sglang/srt/server_args.py +991）
			- 所以ep rank就等于tp rank
		- **启动进程**，跑```run_scheduler_process```
	- 主进程等待scheduler进程发过来一个消息，表示各个子进程模型load完毕
	- 记录两个参数，返回
- 所以，依照上面给定的参数，通信组划分结果是：
```shell
attn_tp_size = 2
attn_dp_size = 8
---------------------------------------------------------------------------------------
NODE_ID | 0                                     | 1
---------------------------------------------------------------------------------------
GPU_ID  | 0    1    2    3    4    5    6    7  | 0    1    2    3    4    5    6    7
---------------------------------------------------------------------------------------
TP      | 0    1    2    3    4    5    6    7  | 8    9    10   11   12   13   14   15
ATTN_DP | 0    0    1    1    2    2    3    3  | 4    4    5    5    6    6    7    7
ATTN_TP | 0    1    0    1    0    1    0    1  | 0    1    0    1    0    1    0    1
---------------------------------------------------------------------------------------
```
如果设想中两个P节点是DP2的关系，那么通信组划分为
```shell
attn_tp_size = 8
attn_dp_size = 2
---------------------------------------------------------------------------------------
NODE_ID | 0                                     | 1
---------------------------------------------------------------------------------------
GPU_ID  | 0    1    2    3    4    5    6    7  | 0    1    2    3    4    5    6    7
---------------------------------------------------------------------------------------
TP      | 0    1    2    3    4    5    6    7  | 8    9    10   11   12   13   14   15
ATTN_DP | 0    0    0    0    0    0    0    0  | 1    1    1    1    1    1    1    1
ATTN_TP | 0    1    2    3    4    5    6    7  | 0    1    2    3    4    5    6    7
---------------------------------------------------------------------------------------
```
- 相同ATTN_DP的GPU共用一个zmq，从node 0的request dispatcher接收数据
	- 例如：node 0的gpu 0和gpu 1共用一个zmq；全局总共8个zmq
### 2.3 Router

**新版的router可能是用rust写的，这里用python版本举例说明**

generate请求发送到router，先随机分配一个合适的PD pair
```python
# sgl-router/py_src/sglang_router/mini_lb.py +290
@app.post("/generate")
async def handle_generage_request(request_data: dict):
	prefill_server, bootstrap_port, decode_server = lb.select_pair()  # lb: load_balancer
```
这里的P/D server其实就是一对URL
然后给每个请求生成bootstrap_room，这里是个64位随机数
```python
modified_request.update(
	{
		"bootstrap_host": [hostname] * batch_size,
		"bootstrap_port": [bootstrap_port]* batch_size,
		"bootstrap_room": [_generate_bootstrap_room() for _ in range(batch_size)],
	}
)
```
调用generate_stream或者generate方法，把请求发给prefill & decode server
```python
# sgl-router/py_src/sglang_router/mini_lb.py +95
tasks = [
	session.post(f"{prefill_server}/{endpoint}, json=modified_request"),
	session.post(f"{decode_server}/{endpoint}, json=modified_request").
]
```
然后走到 [[#3.2.3 process]]，根据bootstrap_room分配decode节点和prefill节点的连接关系
## 3. Scheduler
承接DP Controller的初始化，scheduler会在```run_scheduler_process```方法里决定运行哪个event loop。
**每个rank都有自己的scheduler**
对于PD Disaggregation而且不禁用overlap的情况：
- P Nodes：event_loop_overlap_disagg_prefill
- D Nodes：event_loop_overlap_disagg_decode
### 3.1 Prefill
### 3.1.1 init
```python
# python/sglang/srt/managers/scheduler.py +841
```
先看下PrefillBootstrapQueue初始化传参
```python
tp_size = server_args.tp_size
dp_size = server_args.dp_size
tp_rank = self.tp_rank(local)
gloo_group = self.attn_tp_cpu_group
bootstrap_port = server_args.disaggregation_bootstrap_port, 8998 by default   # 这个和kv_sender也有关联，可能是负担之一
transfer_backend = server_args.disaggregation_transfer_backend, "mooncake" by default
kv_args.engine_rank = self.tp_rank
```
PrefillBootstrapQueue初始化过程的最后一步，需要初始化**KV Manager**
```python
self.kv_manager = self._init_kv_manager()
```
其中主要是获取一些通信组信息、获取cache buffer信息、ib device和gpu信息，构造实际的MooncakeKVManager对象
- 初始化server_socket
	- 此server会绑定到host上一个空闲的随机端口。实际绑定行为发生在prefill/decode开始时
- 调用```_register_to_bootstrap()```方法，**向上文在tokenizer manager里初始化的bootstrap server发送PUT请求，注册自身。 注意这里是每个rank都会注册，所以同一个node上的ip相同，但port有差异```python/sglang/srt/disaggregation/common/conn.py +79```
	- 此时，会向bootstrap server注册如下信息
		- self.prefill_port_table\[system_dp_rank]\[attn_tp_rank]\[pp_rank] = {"rank_ip": rank_ip, "rank_port": rank_port}
		- 这些信息的具体含义要看PUT请求的内容
			- ```shell
				"attn_tp_size": 2,
				"attn_tp_rank": range(2),
				"attn_dp_size": 8,
				"attn_dp_rank": range(8),
				"system_dp_size": 1 if enable dp_attn else dp_size,
				"system_dp_rank": attn_dp_rank # 这个参数从dp_controller传给scheduler，然后传给PrefillBootstrapQueue，再传给KVManager，
				"rank_ip": self.local_ip,
				"rank_port": self.rank_port # 随机的free port
			  ```
		- 所以，bootstrap server里 **每个prefill节点都有自己的唯一标识**
		- 两层字典（忽略pp）嵌套的分级逻辑是：相同dp rank的节点们共享属于同一组request的cache信息
- **初始化MooncakeTransferEngine对象**，传入local_ip、gou_id、ib_device
- 向engine注册buffer
- 启动prefill线程
	- server_socket绑定端口
	- 启动线程跑运行bootstrap_thread。**此线程实现握手的逻辑** ```python/sglang/srt/disaggregation/mooncake/conn.py +747```
		- 第一阶段，decode节点发送room为None的请求
			- prefill线程注册kv cache地址信息，和session id绑定，用于后续计算kv cache目标地址
		- 第二阶段，decode节点发送包含合法bootstrap room id（这个参数来自request）的请求
			- 如果是新的room（不在self.transfer_infos里面）
				- **self.transfer_infos\[room]** = {}
			- self.transfer_infos\[room]\[mooncake_session_id] = TransferInfo.from_zmq(waiting_req_bytes)
			- 此时，**建立了decode节点和prefill节点的联系**，一个room可以对应一组mooncake会话的id
			- 如果收集到了足够多的decode端信息，把room的状态修改为KVPoll.WaitingForInput（？）==这部分还得结合decode看看==
- 初始化transfer_queues和executors，默认各4个；两两一组分配给4个线程运行transfer_worker
	- 应该是用来处理kv transfer任务，```python/sglang/srt/disaggregation/mooncake/conn.py +609```
	- 这部分为了保证逻辑连贯，放到prefill的send章节讲解
#### 3.1.2 recv_request
通过上面的scheduler_input_ipc_name从tokenizer获取request
只有**pp rank0 && attn_tp_rank0**的进程需要接收request（这里结合上面的表格可以看出是合理的）
如果开了dp attn，那么还会根据req的类型划分为work_reqs和control_reqs
然后在attn_tp组内广播work_reqs
在整个tp组内广播control_reqs
#### 3.1.3 bootstrap
参数校验和准备工作不看
先把req塞到bootstrap队列里
```python
# python/sglang/srt/managers/scheduler.py +1463
def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
	if self.disaggregation_mode == DisaggregationMode.PREFILL:
		self._prefetch_kvcache(req)
		self.disagg_prefill_bootstrap_queue.add(req, self.model_config.num_key_value_heads)
```
调用add方法时，会为当前请求构造一个新的MooncakeKVSender，记录在req.disagg_kv_sender里
然后把这个req放到PrefillBootstrapQueue里面的队列里
#### 3.1.4 send
分别在```event_loop_overlap_disagg_prefill```的```process_prefill_chunk```和```process_batch_result_disagg_prefill```两个方法里调用
先调用```send_kv_chunk```，然后调用kv_sender的```send```方法
```python
# python/sglang/srt/disaggregation/prefill.py +607
def send_kv_chunk(self, req, last_chunk: bool = False, end_idx: Optional[int] = None)
```
- 一次发送整页的cache，例如如果page size是4，当前start-end有10个，那么只发8个
- self.req_to_token_pool，好像是ReqToTokenPool类型 ```python/sglang/srt/mem_cache/memory_pool.py +61```
	- 其核心变量self.req_to_token是一个shape为(size, max_context_len)的tensor，记录request --> token locations的映射，token locations代码里也称之为kv indices
- 获取kv indices之后，计算出page indices（简单说就是kv indices除以page size）
- 调用kv sender的send接口
需要注意，这里是按照chunk发送的。所以为了标识发送完成，在发最后一个chunk时会额外带一些辅助信息。
```python
# python/sglang/srt/disaggregation/mooncake/conn.py +989
def send(self, kv_indices: npt.NDArray[np.int32])
```
调用kv_mgr的add_transfer_request方法，这里也区分了是否是最后一个chunk
```python
# python/sglang/srt/disaggregation/mooncake/conn.py +872
def add_transfer_request(self,
						bootstrap_room,
						kv_indices,
						indice_slice,
						is_last,
						aux_index)
```
先构造transfer dst的信息，从前文来看，应该是一组mooncake会话id
```python
dst_infos = self.transfer_infos[bootstrap_room].keys()
```
根据dst_infos选择transfer_queue，把transfer request塞到队列里
这里说，如此选择队列的目的是保证dst session相同的request都放到同一个队列里，从而允许针对failed sessions做early abort
```python
session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
shard_idx = session_port_sum % len(self.transfer_queues)

self.transfer_queues[shard_idx].put(
	TransferKVChunk(
		room=bootstrap_room,
		prefill_kv_indices=kv_indices,
		index_slice=index_slice,
		is_last=is_last,
		prefill_aux_index=aux_index,
	)
)
```
回到上面讲KVManager的部分，放到了transfer_queue里的任务会在```transfer_worker```线程上做处理
这个线程具体的行为如下
```python
# python/sglang/srt/disaggregation/mooncake/conn.py +609
reqs_to_be_processed = self.transfer_infos[kv_chunk.room].values()
local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank
```
取出来当前room对应的**所有kv transfer requests**，对每个请求做操作
从decode节点向prefill节点注册的kv args信息里获取attn_tp_size
```python
# python/sglang/srt/disaggregation/mooncake/conn.py
target_rank_registration_info = self.decode_kv_args_table[req.mooncake_session_id]
```
如果当前是**MLA后端**，或者self.attn_tp_size == target_rank_registration_info.dst_attn_tp_size，则会走self.send_kvcache方法
```python
ret = self.send_kvcache(
	req.mooncake_session_id,                    # mooncake会话id
	kv_chunk.prefill_kv_indices,                # 当前req的当前chunk对应的kv page id
	target_rank_registration_info.dst_kv_ptrs,  #
	chunked_dst_kv_indices,
	executor
)
```
先把连续的页合并到一起
```python
# python/sglang/srt/disaggregation/mooncake/conn.py +251
prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(prefill_kv_indices, dst_kv_indices)
```
获得模型层数、每一层的kv ptr起始和末尾指针
```python
layers_params = [(src_kv_ptrs[layer_id], dst_kv_ptrs[layer_id], kv_item_len) for layer_id in range(layers_current_pp_stage)]
```
每一层都作为一个单独的任务提交给executor，实现并行调用
不过具体传输也限制于executor的线程数
每一次调用传输都是**同步的**
也就是说，假如61层，executor有3个线程，那么总的传输时间是$\text{ceil}(61/3) * t$，t是传输一次的时间
具体的传输任务实现是
```python
# python/sglang/srt/disaggregation/mooncake/conn.py +242
def _transfer_data(self, mooncake_session_id, transfer_blocks):
	src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
	return self.engine.batch_transfer_sync(
		mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
	)
```
这里的self.engine是```MooncakeTransferEngine```，在前面PrefillBootstrapQueue队列构造kv manager时初始化
```python
# python/sglang/srt/disaggregation/mooncake/transfer_engine.py +133
def batch_transfer_sync(self,
						session_id,
						buffers,                  # src_addrs
						peer_buffer_addresses,    # dst_addrs
						lengths                   # lengths
):
	ret = self.engine.batch_transfer_sync_write(session, buffers, peer_buffer_address, lengths)
```
这里的self.engine就是mooncake.TransferEngine了，在mooncake库里面。
### 3.2 Decode
#### 3.2.1 init
decode端初始化过程中，关键是初始化DecodePreallocQueue
```python
# python/sglang/srt/managers/scheduler.py +871
self.disagg_decode_prealloc_queue = DecodePreallocQueue(...)
```
这个对象构造的最后一步也是初始化kv_manager
```python
# python/sglang/srt/disaggregation/decode.py +225
kv_args.engine_rank = self.tp_rank % (attn_tp_size)
kv_manager = kv_manager_class(DisaggregationMode.DECODE)
```
与prefill的区别是：decode端的kv_manager不需要连接到bootstrap server，会多一个connection_pool及lock。这里的连接池在握手时用到（开头提到每个请求都会有一对sender和receiver，就是这个地方）
- **初始化MooncakeTransferEngine对象**，传入local_ip、gou_id、ib_device
- 向engine注册buffer
- kv manager启动工作线程
	- server socket绑定端口，用于接收来自prefill的状态反馈
	- 启动两个线程
		- decode线程，接收并处理prefill的状态通知
			- 给bootstrap_room更新状态
		- heartbeat_checker，定期检查prefill节点的健康状态
#### 3.2.2 recv_requests
#### 3.2.3 process
调用DecodePreallocQueue的add方法
构造一个新的MooncakeKVReceiver，**第一次握手**
```python
# python/sglang/srt/disaggregation/common/conn.py +237
if self.bootstrap_addr not in self.kv_mgr.prefill_dp_size_table: # 后者初始化是个空表
	(self.prefill_attn_tp_size, self.prefill_dp_size, self.prefill_pp_size) = self._get_prefill_parallel_info_from_server()
```
这里是发送了一个特殊的请求（各项参数为-1），从bootstrap server获取prefill info，也就是**prefill节点的并行方案**。
对应的handler是：
```python
# python/sglang/srt/disaggregation/common/conn.py +550
if (int(engine_rank) == -1 ...):
	# P节点还是以TP16 DP8为例
	prefill_parallel_info = {
		"prefill_attn_tp_size": self.attn_tp_size, # 2
        "prefill_dp_size": self.dp_size,           # 就是attn_dp_size，8
        "prefill_pp_size": self.pp_size,           # 1
	}
```
这之后，KVReceiver把上面三个参数分别记录在三个表然后向Prefill节点的握手线程注册kv args
```python

```
```python
self.kv_mgr.prefill_attn_tp_size_table[self.bootstrap_addr] = self.prefill_attn_tp_size
self.kv_mgr.prefill_dp_size_table[self.bootstrap_addr] = self.prefill_dp_size
```
然后处理一些参数，进行**第二次握手**，获取特定一组prefill节点的ip和端口信息
```python
# python/sglang/srt/disaggregation/common/conn.py +342
bootstrap_key = (f"{self.bootstrap_addr}_{self.target_dp_group}_{self.target_tp_rank}")
if bootstrap_key not in self.kv_mgr.connection_pool:
	# 首次连接
	bootstrap_infos = []
	for target_tp_rank in self.target_tp_ranks:
		for target_pp_rank in range(self.prefill_pp_size):
			# 从bootstrap server获取bootstrap info
			# 这个接口实际上是发送了一个GET请求，返回prefill节点注册进去的信息
			bootstrap_info = self._get_bootstrap_info_from_server(target_tp_rank, self.target_dp_group, target_pp_rank)

			bootstrap_infos.append(bootstrap_info)
	self.bootstrap_infos = bootstrap_infos
	self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos
	# 向prefill端注册kv cache
	# 这里的room信息是None  python/sglang/srt/disaggregation/mooncake/conn.py +1124
	self._register_kv_args()
else:
	# 非首次连接
	self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]
```
这里看一下GET请求的参数以及返回值：
- 参数
	- self.target_tp_ranks(对应形参是engine_rank)
		- 初始化应该走的是 python/sglang/srt/disaggregation/common/conn.py +305
		- ```python
		  self.target_tp_ranks = [
                rank
                for rank in range(
                    (self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size)
                    * (self.prefill_attn_tp_size // self.kv_mgr.attn_tp_size),
                    (self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size + 1)
                    * (self.prefill_attn_tp_size // self.kv_mgr.attn_tp_size),
                )
            ]
		  ```
			- 对于decode节点，kv_args.engine_rank = self.tp_rank % (attn_tp_size)  ```python/sglang/srt/disaggregation/decode.py +192```
				- 考虑D节点DP32+EP32的情况，attn_tp_size = 1，所以engine_rank只能是0
			- 所以target_tp_ranks
				- 若P节点TP16 DP8：range((0 % 1) * (2 // 1), (0 % 1 + 1) * (2 // 1)) = range(0, 2)，也就是\[0, 1]
				- 若P节点TP16 DP2：range((0 % 1) * (8 // 1), (0 % 1 + 1) * (8 // 1)) = range(0, 8)，也就是\[0, 7]
	- self.target_dp_group
		- 这个参数来自```req.data_parallel_rank```，在kv receiver初始化时传递。如果不是None，就采用这个值。否则是个prefill_dp_size范围内的随机数（随机的根源变量是bootstrap_room。这个随机来自sgl-router，```sgl-router/py_src/sglang_router/mini_lb.py +369```，每个请求都会有不同的room id）
- 返回值
	- ```python
	  # python/sglang/srt/disaggregation/common/conn.py +564
	  bootstrap_info = self.prefill_port_table[int(target_dp_group)][int(engine_rank)][int(target_pp_rank)]
	  ```
	  - 联系上面算出来的值，查询的是bootstrap server里**随机dp，一整组attn tp，pp0**，共ATTN_TP_SIZE个节点的ip和port信息
	  - **这些信息存储到KVReceiver.kv_mgr.connection_pool里**
从bootstrap server获得了一组Prefill节点信息之后，还要向这组Prefill节点的握手线程注册kv args
```python
# python/sglang/disaggregation/common/conn.py +379
self._register_kv_args()
```
构造一个DecodeRequest，入队
```python
self.queue.append(
                DecodeRequest(req=req, kv_receiver=kv_receiver, waiting_for_input=False)
            )
```