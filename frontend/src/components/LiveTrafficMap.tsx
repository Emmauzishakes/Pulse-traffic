// src/components/LiveTrafficMap.tsx
"use client";
import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip } from 'react-leaflet';

// CRITICAL: This import must be here for the map to be visible
import 'leaflet/dist/leaflet.css';

const getStatusColor = (status: string) => {
  // Matching the exact colors from your screenshot buttons
  switch (status?.toLowerCase()) {
    case 'free': return '#22c55e';   // pulse-green
    case 'slow': return '#eab308';   // pulse-yellow
    case 'jammed': return '#ef4444'; // pulse-red
    default: return '#6b7280';
  }
};

const trafficData = {
  thikaHighway: {
    status: 'free',
    nodes: [
      { id: 't1', name: 'Globe Roundabout', lat: -1.2785, lng: 36.8200 },
      { id: 't2', name: 'Museum Hill Interchange', lat: -1.2755, lng: 36.8145 },
      { id: 't3', name: 'Pangani', lat: -1.2680, lng: 36.8350 },
      { id: 't4', name: 'Muthaiga Roundabout', lat: -1.2600, lng: 36.8400 },
      { id: 't5', name: 'Allsopps / Roasters', lat: -1.2400, lng: 36.8650 },
      { id: 't6', name: 'Kasarani Interchange', lat: -1.2230, lng: 36.8800 }
    ]
  },
  waiyakiWay: {
    status: 'slow',
    nodes: [
      { id: 'w1', name: 'Westlands', lat: -1.2660, lng: 36.8000 },
      { id: 'w2', name: 'Kangemi', lat: -1.2620, lng: 36.7500 }
    ]
  },
  langataRoad: {
    status: 'jammed',
    nodes: [
      { id: 'l1', name: 'Nyayo Stadium', lat: -1.3050, lng: 36.8250 },
      { id: 'l2', name: 'T-Mall Roundabout', lat: -1.3120, lng: 36.8150 }
    ]
  },
  ngongRoad: {
    status: 'free',
    nodes: [
      { id: 'n1', name: 'Prestige Plaza', lat: -1.2980, lng: 36.7850 },
      { id: 'n2', name: 'Junction Mall', lat: -1.2990, lng: 36.7600 }
    ]
  }
};

export default function LiveTrafficMap() {
  const nairobiCenter: [number, number] = [-1.286389, 36.817223];

  return (
    /* We use h-full to fill the parent container, but also add a min-height 
       to prevent the map from collapsing to 0px */
    <div className="w-full h-full min-h-[500px] rounded-xl overflow-hidden border border-gray-800 bg-[#0a0a0a]">
      <MapContainer 
        center={nairobiCenter} 
        zoom={13} 
        scrollWheelZoom={true}
        style={{ height: '100%', width: '100%' }}
        zoomControl={false}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; CartoDB'
        />

        {Object.entries(trafficData).map(([key, corridor]) => {
          const color = getStatusColor(corridor.status);
          const positions = corridor.nodes.map(n => [n.lat, n.lng] as [number, number]);

          return (
            <React.Fragment key={key}>
              <Polyline 
                positions={positions} 
                pathOptions={{ 
                  color: color, 
                  dashArray: '10, 10', 
                  weight: 3, 
                  opacity: 0.6 
                }} 
              />
              
              {corridor.nodes.map((node) => (
                <CircleMarker
                  key={node.id}
                  center={[node.lat, node.lng]}
                  radius={9}
                  pathOptions={{ 
                    color: color, 
                    fillColor: color, 
                    fillOpacity: 0.9, 
                    weight: 2 
                  }}
                >
                  <Tooltip direction="top" offset={[0, -10]} opacity={1}>
                    <div className="p-1 font-sans">
                      <span className="font-bold block text-gray-900">{node.name}</span>
                      <span className="text-xs text-gray-700 capitalize">{corridor.status}</span>
                    </div>
                  </Tooltip>
                </CircleMarker>
              ))}
            </React.Fragment>
          );
        })}
      </MapContainer>
    </div>
  );
}