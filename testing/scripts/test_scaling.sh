#!/bin/bash

# Configuration
SERVICE_URL="http://vox-raga.default.74.224.102.71.nip.io"
REQUESTS_PER_BATCH=10
REQUESTS_PER_SECOND=2
MONITOR_INTERVAL=5
DESIRED_PODS=2
MAX_WAIT_TIME=120  # Maximum time to wait for pods to be ready (seconds)

# Function to check pod count and details
check_pods() {
    export KUBECONFIG=/Users/bnr/work/openstack/kubeconfigs/s-azure-standalone.kubeconfig
    echo "=== Current Pods ==="
    kubectl get pods -l serving.kserve.io/inferenceservice=vox-raga -o wide
    echo "=== Pod Count ==="
    kubectl get pods -l serving.kserve.io/inferenceservice=vox-raga --no-headers | wc -l | tr -d ' '
}

# Function to send a request
send_request() {
    local id=$1
    local text="This is test request ${id}. Testing TTS service scaling."
    
    echo "Sending request $id..."
    curl -s -X POST "$SERVICE_URL/synthesize" \
         -H "Content-Type: application/json" \
         -d "{\"text\": \"$text\", \"language\": \"en\", \"voice\": \"p225\"}" \
         > /dev/null
    
    if [ $? -eq 0 ]; then
        echo "Request $id completed successfully"
        return 0
    else
        echo "Request $id failed"
        return 1
    fi
}

# Function to send a batch of requests
send_batch() {
    local batch_num=$1
    local start_id=$2
    local count=$3
    
    echo "DEBUG: Starting batch $batch_num with $count requests starting from ID $start_id"
    
    for i in $(seq $start_id $((start_id + count - 1))); do
        send_request $i &
        sleep $(echo "scale=3; 1/$REQUESTS_PER_SECOND" | bc)
    done
    
    # Wait for all requests in this batch to complete
    wait
    echo "DEBUG: Batch $batch_num complete"
}

echo "=== Starting Scaling Test ==="
echo "Initial pod count: $(check_pods)"
echo

# Phase 1: Send first batch to trigger scaling
echo "DEBUG: Starting Phase 1 - Sending first batch to trigger scaling"
send_batch 1 1 $REQUESTS_PER_BATCH

# Wait for the second pod to be ready
echo "DEBUG: Waiting for $DESIRED_PODS pods to be ready..."
start_time=$(date +%s)
pods_ready=false

while [ "$pods_ready" = false ]; do
    # Check if we've exceeded the maximum wait time
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -gt $MAX_WAIT_TIME ]; then
        echo "DEBUG: Timeout waiting for pods to be ready after $MAX_WAIT_TIME seconds"
        exit 1
    fi
    
    # Get detailed pod information
    export KUBECONFIG=/Users/bnr/work/openstack/kubeconfigs/s-azure-standalone.kubeconfig
    echo "DEBUG: Checking pod status..."
    kubectl get pods -l serving.kserve.io/inferenceservice=vox-raga -o wide
    
    # Get the total number of pods
    total_pods=$(kubectl get pods -l serving.kserve.io/inferenceservice=vox-raga --no-headers | wc -l | tr -d ' ')
    
    # Get container readiness for all pods in service
    pod_status=$(kubectl get pods -l serving.kserve.io/inferenceservice=vox-raga -o jsonpath='{range .items[*]}{range .status.containerStatuses[*]}{.ready}{" "}{end}{"\n"}{end}')
    
    # Count number of 'true' statuses
    ready_count=$(echo "$pod_status" | grep -o 'true' | wc -l)
    container_count=$(echo "$pod_status" | wc -w)
    
    echo "DEBUG: Pod ready status: $pod_status"
    echo "DEBUG: Ready count: $ready_count / $container_count (elapsed: ${elapsed}s)"
    
    if [ "$ready_count" -eq "$container_count" ] && [ "$total_pods" -eq "$DESIRED_PODS" ]; then
        pods_ready=true
        echo "DEBUG: All $DESIRED_PODS pods are now ready!"
    else
        echo "DEBUG: Pods exist but not all are ready yet. Sleeping..."
        sleep 5
    fi
done

# Phase 2: Send second batch which should be distributed across pods
echo "DEBUG: Starting Phase 2 - Sending second batch for load distribution"
send_batch 2 $((REQUESTS_PER_BATCH + 1)) $REQUESTS_PER_BATCH

echo -e "\n=== Test Complete ==="
echo "Final pod count: $(check_pods)"
echo "DEBUG: Waiting for scale-down to occur..."
sleep 30
echo "Final pod count after scale-down: $(check_pods)" 