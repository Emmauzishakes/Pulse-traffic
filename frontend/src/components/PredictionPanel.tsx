"use client";

import React from 'react';
import { X, Navigation, AlertTriangle, CheckCircle, Clock, ChevronRight, Cpu } from 'lucide-react';
import { PredictionResult, resolveNodeName } from '@/hooks/usePrediction';
import { NODE_DIRECTORY } from '@/lib/types';

interface PredictionPanelProps {
  nodeId: string;
  destinationId: string;
  result: PredictionResult | undefined;
  isLoading: boolean;
  isError: boolean;
  onClose: () => void;
  onDestinationChange: (destinationId: string) => void;
}

const levelConfig = {
  Low:    { color: 'text-pulse-green', border: 'border-pulse-green/30', bg: 'bg-pulse-green/10',  icon: CheckCircle,     label: 'Free Flow' },
  Medium: { color: 'text-pulse-yellow', border: 'border-pulse-yellow/30', bg: 'bg-pulse-yellow/10', icon: AlertTriangle,  label: 'Moderate' },
  High:   { color: 'text-pulse-red',   border: 'border-pulse-red/30',   bg: 'bg-pulse-red/10',    icon: AlertTriangle,   label: 'Congested' },
};

// All 15 nodes as destination options
const ALL_NODES = Object.values(NODE_DIRECTORY);

export const PredictionPanel: React.FC<PredictionPanelProps> = ({
  nodeId,
  destinationId,
  result,
  isLoading,
  isError,
  onClose,
  onDestinationChange,
}) => {
  const nodeName = resolveNodeName(nodeId);
  const destName = resolveNodeName(destinationId);
  const level = result?.predicted_level ?? 'Low';
  const config = levelConfig[level];
  const LevelIcon = config.icon;

  return (
    <div className="fixed bottom-6 right-6 w-[400px] bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl z-50 overflow-hidden">
      
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-700 bg-slate-800/60">
        <div className="flex items-center gap-2">
          <Navigation size={16} className="text-pulse-cyan" />
          <span className="text-white font-semibold text-sm">Route Predictor</span>
          {result?.used_ml && (
            <span className="flex items-center gap-1 text-[10px] bg-pulse-cyan/10 text-pulse-cyan border border-pulse-cyan/20 px-2 py-0.5 rounded-full">
              <Cpu size={10} /> ML
            </span>
          )}
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
          <X size={16} />
        </button>
      </div>

      <div className="p-5 space-y-4">

        {/* Origin → Destination */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-slate-400 w-16 text-xs uppercase tracking-wider">From</span>
            <span className="text-white font-medium">{nodeName}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-400 w-16 text-xs uppercase tracking-wider">To</span>
            <select
              value={destinationId}
              onChange={(e) => onDestinationChange(e.target.value)}
              className="flex-1 bg-slate-800 border border-slate-600 text-white text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-pulse-cyan"
            >
              {ALL_NODES
                .filter(n => n.id !== nodeId)
                .map(n => (
                  <option key={n.id} value={n.id}>{n.name}</option>
                ))
              }
            </select>
          </div>
        </div>

        {/* Loading state */}
        {isLoading && (
          <div className="flex items-center justify-center py-8 text-slate-400 text-sm gap-2">
            <span className="w-2 h-2 bg-pulse-cyan rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-2 h-2 bg-pulse-cyan rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-2 h-2 bg-pulse-cyan rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
        )}

        {/* Error state */}
        {isError && !isLoading && (
          <div className="text-pulse-red text-sm text-center py-4 bg-pulse-red/10 rounded-lg border border-pulse-red/20">
            Could not fetch prediction. Check backend is running.
          </div>
        )}

        {/* Result */}
        {result && !isLoading && (
          <>
            {/* Congestion badge */}
            <div className={`flex items-center gap-3 p-3 rounded-xl border ${config.border} ${config.bg}`}>
              <LevelIcon size={20} className={config.color} />
              <div>
                <p className={`font-semibold text-sm ${config.color}`}>{config.label} at {nodeName}</p>
                <p className="text-slate-400 text-xs mt-0.5">
                  Confidence: {result.confidence_score}%
                </p>
              </div>
            </div>

            {/* Alternative routes */}
            {result.alternative_routes.length > 0 && (
              <div className="space-y-2">
                <p className="text-slate-400 text-xs uppercase tracking-wider">Suggested Routes</p>
                {result.alternative_routes.map((route, i) => {
                  const hasCongestion = route.congestion_segments.length > 0;
                  return (
                    <div
                      key={i}
                      className={`p-3 rounded-xl border text-sm ${
                        i === 0
                          ? 'border-pulse-cyan/30 bg-pulse-cyan/5'
                          : 'border-slate-700 bg-slate-800/40'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className={`text-xs font-bold uppercase tracking-wider ${i === 0 ? 'text-pulse-cyan' : 'text-slate-400'}`}>
                          {i === 0 ? '★ Best Route' : `Option ${i + 1}`}
                        </span>
                        <span className="flex items-center gap-1 text-xs text-slate-300">
                          <Clock size={11} /> {route.estimated_time} min
                        </span>
                      </div>

                      {/* Path as chips */}
                      <div className="flex flex-wrap items-center gap-1">
                        {route.display_path.map((nodeId, idx) => {
                          const isCongested = route.congestion_segments.includes(nodeId);
                          return (
                            <React.Fragment key={nodeId}>
                              <span className={`text-[11px] px-2 py-0.5 rounded-full border ${
                                isCongested
                                  ? 'text-pulse-red border-pulse-red/30 bg-pulse-red/10'
                                  : 'text-slate-300 border-slate-600 bg-slate-800'
                              }`}>
                                {resolveNodeName(nodeId)}
                              </span>
                              {idx < route.display_path.length - 1 && (
                                <ChevronRight size={10} className="text-slate-500 flex-shrink-0" />
                              )}
                            </React.Fragment>
                          );
                        })}
                      </div>

                      {hasCongestion && (
                        <p className="text-[10px] text-pulse-yellow mt-2">
                          ⚠ Congestion on {route.congestion_segments.map(resolveNodeName).join(', ')}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};