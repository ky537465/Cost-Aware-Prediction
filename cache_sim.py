import sys
import math

class Cache:
    def __init__(self, size, assoc, blocksize, policy):
        self.size = size
        self.assoc = assoc
        self.blocksize = blocksize
        self.policy = policy # 0:LRU, 1:FIFO, 2:Optimal
        
        # Geometry
        self.num_sets = size // (assoc * blocksize)
        self.index_bits = int(math.log2(self.num_sets))
        self.offset_bits = int(math.log2(blocksize))
        
        # Data storage: List of sets, each set is a list of blocks (dicts)
        self.sets = []
        for _ in range(self.num_sets):
            # Each block: {'tag': int, 'valid': bool, 'dirty': bool, 'counter': int}
            self.sets.append([{
                'tag': None, 
                'valid': False, 
                'dirty': False, 
                'counter': 0
            } for _ in range(assoc)])

        # Statistics
        self.reads = 0
        self.read_misses = 0
        self.writes = 0
        self.write_misses = 0
        self.writebacks = 0

    def get_addr_params(self, addr):
        index = (addr >> self.offset_bits) & (self.num_sets - 1)
        tag = addr >> (self.offset_bits + self.index_bits)
        return index, tag

    def access(self, op, addr, timer, next_level=None, opt_data=None):
        index, tag = self.get_addr_params(addr)
        target_set = self.sets[index]
        
        if op == 'r': self.reads += 1
        else: self.writes += 1

        # 1. Check for Hit
        for block in target_set:
            if block['valid'] and block['tag'] == tag:
                if op == 'w': block['dirty'] = True
                if self.policy == 0: # LRU
                    block['counter'] = timer
                return "HIT", None

        # 2. On Miss
        if op == 'r': self.read_misses += 1
        else: self.write_misses += 1
        
        # 3. Find Victim
        victim_idx = self._find_victim(index, timer, opt_data)
        victim = target_set[victim_idx]
        evicted_addr = None

        # 4. Handle Writeback if victim is dirty
        if victim['valid']:
            if victim['dirty']:
                self.writebacks += 1
                # Reconstruct address for next level
                evicted_addr = (victim['tag'] << (self.index_bits + self.offset_bits)) | (index << self.offset_bits)
                if next_level:
                    next_level.access('w', evicted_addr, timer)

        # 5. Bring in new block (Read from next level)
        if next_level:
            next_level.access('r', addr, timer)

        # 6. Update victim slot with new data
        victim['valid'] = True
        victim['tag'] = tag
        victim['dirty'] = (op == 'w')
        victim['counter'] = timer
        
        return "MISS", evicted_addr

    def _find_victim(self, index, timer, opt_data):
        target_set = self.sets[index]
        
        # Check for empty slot
        for i, block in enumerate(target_set):
            if not block['valid']: return i
            
        # Replacement Policies
        if self.policy == 0 or self.policy == 1: # LRU or FIFO
            # Both use the lowest counter value (oldest/least recent)
            return min(range(len(target_set)), key=lambda i: target_set[i]['counter'])
        
        elif self.policy == 2: # Optimal
            # Logic would involve looking ahead in opt_data trace
            return 0 # Placeholder for simplicity

# Example Setup for Hierarchy
def main():
    # Parameters from command line or config
    # L1: 8KB, 4-way, 32B blocks, LRU
    L1 = Cache(8192, 4, 32, 0)
    # L2: 32KB, 8-way, 32B blocks, LRU
    L2 = Cache(32768, 8, 32, 0)

    # Simulated trace processing
    trace = [('r', 0xffe04540), ('w', 0xffe04544)]
    timer = 0
    
    for op, addr in trace:
        timer += 1
        # L1 access triggers L2 internally via 'next_level' logic
        L1.access(op, addr, timer, next_level=L2)

    print(f"L1 Miss Rate: {(L1.read_misses + L1.write_misses) / (L1.reads + L1.writes):.4f}")

if __name__ == "__main__":
    main()