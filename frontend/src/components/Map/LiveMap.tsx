import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { EnrichedTrafficData } from '@/lib/types';

interface LiveMapProps {
  data: EnrichedTrafficData[] | null;
  focusedNode?: { lat: number, lng: number } | null;
}

// Custom component to handle the "Flying" animation when a sidebar item is clicked
const MapController = ({ focusedNode }: { focusedNode?: {lat: number, lng: number} | null }) => {
  const map = useMap();
  
  useEffect(() => {
    if (focusedNode) {
      // Flied to the node at zoom level 15 with a smooth 1.5 second animation
      map.flyTo([focusedNode.lat, focusedNode.lng], 15, {
        duration: 1.5,
      });
    }
  }, [focusedNode, map]);

  return null; // This component doesn't render anything visually
};

const createCustomIcon = (status: 'Free' | 'Slow' | 'Jammed', name: string) => {
  const colors = { Free: '#22c55e', Slow: '#eab308', Jammed: '#ef4444' };
  const color = colors[status];

  return L.divIcon({
    className: 'custom-node-icon',
    html: `
      <div style="display: flex; flex-direction: column; align-items: center; gap: 4px; margin-top: -24px;">
        <div style="
          background: #0f172a; 
          border: 1px solid #334155; 
          padding: 2px 8px; 
          border-radius: 4px; 
          font-size: 10px; 
          font-weight: 600;
          color: #f1f5f9; 
          white-space: nowrap;
          box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        ">
          ${name}
        </div>
        <div style="
          width: 14px; 
          height: 14px; 
          background-color: ${color}; 
          border-radius: 50%; 
          border: 2px solid #0f172a;
          box-shadow: 0 0 12px ${color};
        "></div>
      </div>
    `,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });
};

const LiveMap: React.FC<LiveMapProps> = ({ data, focusedNode }) => {
  // Center slightly offset to account for the wide spread from Juja to Karen
  const nairobiCenter: [number, number] = [-1.285, 36.850];

  // The 4 major highway paths drawn strictly point-to-point based on your nodes
  const corridors = {
    thikaHwy: [
      [-1.2780, 36.8180], // CBD
      [-1.2650, 36.8350], // Pangani
      [-1.2580, 36.8400], // Muthaiga
      [-1.2200, 36.8800], // Kasarani
      [-1.1030, 37.0144], // Juja
    ] as [number, number][],
    expressway: [
      [-1.2660, 36.8000], // Westlands
      [-1.2730, 36.8130], // Museum Hill
      [-1.2780, 36.8180], // CBD 
      [-1.3050, 36.8350], // Capital Centre
      [-1.3340, 36.9060], // JKIA
      [-1.3850, 36.9380], // Mlolongo
    ] as [number, number][],
    ngongRd: [
      [-1.2780, 36.8180], // CBD
      [-1.2990, 36.7860], // Prestige
      [-1.2980, 36.7620], // Junction
      [-1.3190, 36.7050], // Karen
    ] as [number, number][],
    langataRd: [
      [-1.2780, 36.8180], // CBD
      [-1.3130, 36.8140], // T-Mall
      [-1.3400, 36.7660], // Bomas
    ] as [number, number][],
  };

  return (
    <MapContainer 
      center={nairobiCenter} 
      zoom={11} // Set to 11 to fit both Juja and Karen on initial load
      scrollWheelZoom={true} 
      className="w-full h-full z-0"
    >
      <MapController focusedNode={focusedNode} />

      <TileLayer
        attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />

      {/* Draw the high-tech glowing corridors */}
      <Polyline positions={corridors.thikaHwy} pathOptions={{ color: '#06b6d4', weight: 3, opacity: 0.6 }} />
      <Polyline positions={corridors.expressway} pathOptions={{ color: '#8b5cf6', weight: 3, opacity: 0.6 }} />
      <Polyline positions={corridors.ngongRd} pathOptions={{ color: '#f59e0b', weight: 3, opacity: 0.6 }} />
      <Polyline positions={corridors.langataRd} pathOptions={{ color: '#10b981', weight: 3, opacity: 0.6 }} />
      
      {data?.map((node) => {
        if (!node.meta || node.meta.lat === 0) return null;

        return (
          <Marker 
            key={node.id} 
            position={[node.meta.lat, node.meta.lng]} 
            icon={createCustomIcon(node.status, node.meta.name)}
          />
        );
      })}
    </MapContainer>
  );
};

export default LiveMap;