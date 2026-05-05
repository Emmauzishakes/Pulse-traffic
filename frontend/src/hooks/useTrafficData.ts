"use client";

import { useQuery } from '@tanstack/react-query';
import { BackendTrafficReading, EnrichedTrafficData, NODE_DIRECTORY } from '../lib/types';

const API_URL = 'http://localhost:8000';

export const useLiveNetwork = () => {
  return useQuery({
    queryKey: ['liveTraffic'],
    queryFn: async (): Promise<EnrichedTrafficData[]> => {
      const res = await fetch(`${API_URL}/traffic-data?limit=100`);
      if (!res.ok) throw new Error('Failed to fetch network data');
      
      const rawData: BackendTrafficReading[] = await res.json();
      
      // Enrich backend data with UI metadata and calculate status
      return rawData.map(reading => {
        const meta = NODE_DIRECTORY[reading.node_id] || { name: 'Unknown Node', corridor: 'Unknown', lat: 0, lng: 0 };
        
        let status: 'Free' | 'Slow' | 'Jammed' = 'Free';
        if (reading.speed < 20 || reading.density > 0.8) status = 'Jammed';
        else if (reading.speed < 40 || reading.density > 0.5) status = 'Slow';

        return { ...reading, meta, status };
      });
    },
    refetchInterval: 4000, // Sync with your simulator's tick rate
  });
};