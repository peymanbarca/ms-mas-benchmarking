# Scenario 3:

## Order Service --> Inventory Services via gRPC for reservation
    gRPC messages are serialized using Protocol Buffers (protobuf)
    
    Flow (client call):
        - Python code creates a ReserveRequest object.

        - Stub serializes it to protobuf binary.

        - Binary payload is wrapped in an HTTP/2 frame and sent to the server.

        - Server receives, parses into ReserveRequest object.

        - Your handler (Reserve method) runs and returns a ReserveResponse.

        - Response is serialized to binary protobuf, streamed back over HTTP/2.

        - Stub deserializes into ReserveResponse object in the client.

### Performance vs REST/JSON

    Serialization cost: Protobuf binary → ~10–50x faster than JSON parse/dump.
    
    Message size: Usually ~3–10x smaller than JSON.
    
    Transport: HTTP/2 persistent connections mean you avoid the HTTP/1.1 TCP connection overhead REST calls often suffer from.
    
    Concurrency: HTTP/2 allows multiplexing many requests over one connection → lower latency under load.

## Only a fixed delay injected in response of reservation from inventory service

## Run trial Experiments

### 1. Parallel Orders
### 2. Sequential Orders
