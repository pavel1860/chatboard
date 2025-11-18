





from typing import Any





# class GeneratorFrame:
#     def __init__(self, controller: StreamController, agen: AsyncGenerator, accumulator: SupportsExtend):
#         self.controller = controller
#         self.agen = agen
#         self.accumulator = accumulator



        
        
        
        
        
        
        
        
        
        
        
        
        
        

# yield StreamEvent(type="span_start", name=self.current.name)
# value = await self.current.asend(None)
# try:      
#     while True:
#         if isinstance(value, StreamController):
#             stream = value
#             async for chunk in stream:
#                 yield chunk
#                 # print(chunk)
#             value = await self.current.asend(stream.acc)
#             # yield StreamEvent(type="span_value", name=self._gen_func.__name__, payload=value)
#             continue
#         elif isinstance(value, PipeController):
#             pipe = value
#             res = None
#             async for res in pipe:
#                 yield res
#             value = await gen.asend(res.payload if res else None)
#             continue
#         else:
#             yield StreamEvent(type="span_value", name=self._gen_func.__name__, payload=value)
#             value = await gen.asend(value)                    
# except StopAsyncIteration:
#     yield StreamEvent(type="span_end", name=self._gen_func.__name__, payload=value)
