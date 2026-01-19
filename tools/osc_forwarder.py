import argparse
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client

def print_and_forward(address, *args):
    """
    Handler that prints the incoming message and forwards it via the client.
    """
    # 1. Print to console (optional, good for debugging)
    print(f"Received: {address}: {args}")
    
    # 2. Forward to the remote target
    if client:
        client.send_message(address, args)
        # Only print forwarding confirmation if you want verbose logs
        # print(f"Forwarded to {args_namespace.dest_ip}:{args_namespace.dest_port}")

if __name__ == "__main__":
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Forward OSC messages from localhost to a remote IP.")
    
    # Listening settings (Localhost)
    parser.add_argument("--listen-ip", default="127.0.0.1", help="The IP to listen on (default: 127.0.0.1)")
    parser.add_argument("--listen-port", type=int, default=5005, help="The port to listen on (default: 5005)")
    
    # Forwarding settings (Remote)
    parser.add_argument("--dest-ip", default="192.168.1.50", help="The destination IP to forward to")
    parser.add_argument("--dest-port", type=int, default=9000, help="The destination port to forward to")
    
    args_namespace = parser.parse_args()

    # 2. Set up the UDP Client (The Sender)
    print(f"--- Configuration ---")
    print(f"Listening on: {args_namespace.listen_ip}:{args_namespace.listen_port}")
    print(f"Forwarding to: {args_namespace.dest_ip}:{args_namespace.dest_port}")
    print(f"---------------------")
    
    try:
        client = udp_client.SimpleUDPClient(args_namespace.dest_ip, args_namespace.dest_port)
    except Exception as e:
        print(f"Error creating client: {e}")
        exit()

    # 3. Set up the Dispatcher (The Router)
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("*", print_and_forward) 

    # 4. Start the Server (The Listener)
    server = osc_server.ThreadingOSCUDPServer(
        (args_namespace.listen_ip, args_namespace.listen_port), 
        dispatcher
    )
    
    print("Forwarder running... Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
