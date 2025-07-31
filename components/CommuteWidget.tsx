"use client";

import { useEffect, useRef } from "react";

export default function CommutesWidget() {
  const mapContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!window.google || !mapContainerRef.current) return;

    const configuration = {
      mapOptions: {
        center: { lat: 37.7749, lng: -122.4194 },
        zoom: 13,
        mapTypeId: "roadmap",
      },
      defaultTravelMode: "DRIVING",
      initialDestinations: [],
      distanceMeasurementType: "METRIC",
    };

    // Make sure Commutes is globally available (injected by script)
    if ((window as any).Commutes) {
      new (window as any).Commutes(configuration);
    } else {
      console.error("Commutes function not found");
    }
  }, []);

  return (
    <div className="w-full h-[600px] relative mt-4 border rounded-lg shadow">
      <div className="map-view w-full h-full" ref={mapContainerRef} />
      <div className="commutes-initial-state hidden"></div>
      <div className="commutes-destinations hidden">
        <div className="left-control" data-direction="-1">←</div>
        <div className="right-control" data-direction="1">→</div>
        <div className="destinations-container">
          <div className="destination-list"></div>
          <button className="add-button">Add</button>
        </div>
      </div>
      <div className="commutes-modal-container hidden">
        <form name="destination-form">
          <input name="destination-address" type="text" />
          <input type="radio" name="travel-mode" value="DRIVING" id="driving-mode" />
          <input type="radio" name="travel-mode" value="WALKING" id="walking-mode" />
          <input type="radio" name="travel-mode" value="BICYCLING" id="bicycling-mode" />
          <input type="radio" name="travel-mode" value="TRANSIT" id="transit-mode" />
        </form>
        <div className="error-message"></div>
        <button className="add-destination-button">Add</button>
        <button className="edit-destination-button">Edit</button>
        <button className="delete-destination-button">Delete</button>
        <button className="cancel-button">Cancel</button>
      </div>
    </div>
  );
}