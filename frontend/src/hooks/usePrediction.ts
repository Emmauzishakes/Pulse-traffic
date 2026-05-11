"use client";

import { useQuery } from '@tanstack/react-query';
import { NODE_DIRECTORY } from '../lib/types';

const API_URL = 'http://localhost:8000';

export interface RouteOption {
  display_path: string[];
  estimated_time: number;
  congestion_segments: string[];
  summary: string;
}

export interface PredictionResult {
  node_id: string;
  predicted_level: 'Low' | 'Medium' | 'High';
  confidence_score: number;
  used_ml: boolean;
  destination: string;
  alternative_routes: RouteOption[];
  recommendation: string;
}

// Resolves a node_id to its human-readable name e.g. "ST-LANG-001" → "T-Mall Roundabout"
export const resolveNodeName = (nodeId: string): string => {
  return NODE_DIRECTORY[nodeId]?.name || nodeId;
};

export const usePrediction = (nodeId: string | null, destinationId: string | null) => {
  return useQuery({
    queryKey: ['prediction', nodeId, destinationId],
    queryFn: async (): Promise<PredictionResult> => {
      const res = await fetch(
        `${API_URL}/predict/${nodeId}/route?destination=${destinationId}`,
        { method: 'POST' }
      );
      if (!res.ok) throw new Error('Prediction failed');
      return res.json();
    },
    // Only run when both nodeId and destinationId are set
    enabled: !!nodeId && !!destinationId,
    // Don't refetch automatically — only when the user picks a new node/destination
    refetchOnWindowFocus: false,
    retry: 1,
  });
};