"use client";
import React, { useMemo } from 'react';
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip, Pane } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { EnrichedTrafficData } from '@/lib/types';

// Coordinates
const JUNCTION_COORDINATES: Record<string, [number, number]> = {
  "Globe Roundabout (CBD)": [-1.2785, 36.8200],
  "Museum Hill Interchange": [-1.2755, 36.8145],
  "Pangani": [-1.2680, 36.8350],
  "Muthaiga Roundabout": [-1.2600, 36.8400],
  "Allsopps / Roasters": [-1.2400, 36.8650],
  "Kasarani Interchange": [-1.2230, 36.8800],
  "Kenyatta Ave / Uhuru Hwy": [-1.2850, 36.8180],
  "Landhies / Bus Station": [-1.2880, 36.8300],
  "Bunyala Roundabout": [-1.2980, 36.8200],
  "City Stadium": [-1.2950, 36.8400],
  "Burma Market": [-1.2920, 36.8450],
};

// Corridors
const CORRIDOR_PATHS = {
  "Thika Superhighway": ["Muthaiga Roundabout", "Pangani", "Globe Roundabout (CBD)"],
  "Uhuru Highway": ["Museum Hill Interchange", "Kenyatta Ave / Uhuru Hwy", "Bunyala Roundabout"],
  "Jogoo Road": ["Landhies / Bus Station", "City Stadium", "Burma Market"]
};

// Status color helper
const getStatusColor = (status?: string) => {
  switch (status?.toLowerCase()) {
    case 'free': return '#22c55e';
    case 'slow': return '#eab308';
    case 'jammed': return '#ef4444';
    default: return '#4b5563';
  }
};

export default function LiveMap({ data }: { data: EnrichedTrafficData[] | null }) {
  const center: [number, number] = [-1.286389, 36.817223];

  const nodeLookup = useMemo(() => {
    const lookup: Record<string, EnrichedTrafficData> = {};
    data?.forEach(node => {
      lookup[node.meta.name] = node;
    });
    return lookup;
  }, [data]);

  return (
    <div className="w-full h-full min-h-[600px] bg-[#0a0a0a] relative overflow-hidden rounded-lg border border-slate-800">
      
      <MapContainer 
        center={center} 
        zoom={13} 
        className="h-full w-full"
        zoomControl={false}
        scrollWheelZoom={true}
      >
         <Pane name="trafficPane" style={{ zIndex: 650 }} />   {/* 👈 ADD THIS */}
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; CartoDB'
        />

        {/* ===== ROADS WITH GLOW EFFECT ===== */}
        {Object.entries(CORRIDOR_PATHS).map(([corridorName, junctionNames]) => {
          const points = junctionNames
            .map(name => JUNCTION_COORDINATES[name])
            .filter(Boolean) as [number, number][];

          // Normalize status casing (important)
          const statuses = junctionNames.map(
    name => nodeLookup[name]?.status?.toLowerCase()
   );

          const isJammed = statuses.includes('jammed');
          const isSlow = statuses.includes('slow');

          const pathColor = isJammed
            ? '#ef4444'
            : isSlow
            ? '#eab308'
            : '#22c55e';

          return (
            <React.Fragment key={corridorName}>
              
              {/* Glow layer */}
              <Polyline
                positions={points}
                pathOptions={{
                  color: pathColor,
                  weight: 14,
                  opacity: 0.25,
                }}
              />

              {/* Main road */}
              return (
  <React.Fragment key={corridorName}>
    
    {/* Glow line */}
    <Polyline
      pane="trafficPane"
      positions={points}
      pathOptions={{
        color: pathColor,
        weight: 20,
        opacity: 0.35,
      }}
    />

    {/* Main visible road */}
    <Polyline
      pane="trafficPane"
      positions={points}
      pathOptions={{
        color: pathColor,
        weight: 10,
        opacity: 1,
        lineCap: 'round',
      }}
    />

  </React.Fragment>
);
        
                {/* ===== NODES ===== */}
        {Object.entries(JUNCTION_COORDINATES).map(([name, coords]) => {
          const liveNode = nodeLookup[name];
          const color = getStatusColor(liveNode?.status);

          return (
            <CircleMarker
              key={name}
              center={coords}
              radius={liveNode ? 12 : 8}
              pathOptions={{
                fillColor: color,
                fillOpacity: 1,
                color: '#000',
                weight: 2,
              }}
            >
              <Tooltip direction="top" offset={[0, -10]}>
                <div className="bg-slate-900 text-white p-2 rounded shadow-xl border border-slate-700">
                  <p className="font-bold text-sm border-b border-slate-700 pb-1 mb-1">{name}</p>
                  {liveNode ? (
                    <div className="text-xs space-y-1">
                      <p className="flex justify-between gap-4">
                        <span>Status:</span>
                        <span style={{ color }} className="font-bold uppercase">
                          {liveNode.status}
                        </span>
                      </p>
                      <p className="flex justify-between gap-4">
                        <span>Speed:</span>
                        <span>{liveNode.speed} km/h</span>
                      </p>
                    </div>
                  ) : (
                    <p className="text-[10px] text-slate-400 italic">
                      Awaiting sensor data...
                    </p>
                  )}
                </div>
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>

      {/* ===== OPTIONAL DARK OVERLAY (improves contrast) ===== */}
      <div className="absolute inset-0 bg-black/30 pointer-events-none" />
    </div>
  );
}