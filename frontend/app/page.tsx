'use client';

import { useState, useRef, useCallback, useEffect } from "react";
import { GoogleMap, Marker, InfoWindow, useJsApiLoader, Polyline, Autocomplete, TransitLayer } from "@react-google-maps/api";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Alert } from "@/components/ui/alert";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Info, MapPin, Car, User, Upload, Route } from "lucide-react";

// Re-introduce the Step interface
interface Step {
  text: string;
  photo_url: string;
}

const Maps_API_KEY = process.env.NEXT_PUBLIC_Maps_API_KEY?.trim();
const FLASK_API_URL = process.env.NEXT_PUBLIC_FLASK_API_URL?.trim();
console.log("Maps API Key:", Maps_API_KEY);
console.log("Test var:", process.env.NEXT_PUBLIC_TEST_VAR);


const containerStyle = {
  width: "100%",
  height: "100vh",
};

const defaultCenter = {
  lat: 37.7749,
  lng: -122.4194,
};
// Types
interface MarkerData {
  position: { lat: number; lng: number };
  label: string;
  info: string;
}

interface Landmark {
  type: "text" | "structure";
  confidence: number;
  box: [number, number, number, number];
  text?: string;
  category?: string;
  class_id?: number;
  structure_name?: string;
}

interface PickupPoint {
  lat: number;
  lng: number;
  ar_guidance?: string;
}

export default function Home() {


  const [map, setMap] = useState<google.maps.Map | null>(null);
  const [center, setCenter] = useState<{ lat: number; lng: number }>(defaultCenter);
  const [markers, setMarkers] = useState<MarkerData[]>([]);
  const [selectedMarker, setSelectedMarker] = useState<MarkerData | null>(null);
  const [pickupPoint, setPickupPoint] = useState<PickupPoint | null>(null); // Recommended pickup

  const [driverToPickupPolylinePath, setDriverToPickupPolylinePath] = useState<google.maps.LatLngLiteral[]>([]);
  const [passengerToPickupPolylinePath, setPassengerToPickupPolylinePath] = useState<google.maps.LatLngLiteral[]>([]);
  // New state to control which polyline is shown
  const [activePolyline, setActivePolyline] = useState<'driver' | 'passenger' | null>(null);

  const [processing, setProcessing] = useState<boolean>(false);
  const [processedImageUrl, setProcessedImageUrl] = useState<string>("");
  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [scene, setScene] = useState<string>("");
  const [buildingType, setBuildingType] = useState<string>("");
  const [searchBox, setSearchBox] = useState<google.maps.places.Autocomplete | null>(null);
  const [searchInput, setSearchInput] = useState<string>("");

  const [driverLat, setDriverLat] = useState<string>("");
  const [driverLng, setDriverLng] = useState<string>("");
  const [passengerLat, setPassengerLat] = useState<string>("");
  const [passengerLng, setPassengerLng] = useState<string>("");
  const [driverMarker, setDriverMarker] = useState<MarkerData | null>(null);
  const [passengerMarker, setPassengerMarker] = useState<MarkerData | null>(null);

  const [pickupLoading, setPickupLoading] = useState<boolean>(false);

  const [steps, setSteps] = useState<Step[]>([]); // State now holds an array of Step objects
  const [showDirectionsModal, setShowDirectionsModal] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<number>(0);

  const [pickupMessage, setPickupMessage] = useState<string>("");
  const [pickupError, setPickupError] = useState<string>("");
  const [safetyValidation, setSafetyValidation] = useState<{
    is_safe: boolean;
    safety_score: number;
    issues_found: string[];
    recommendations: string[];
    lighting_quality?: string;
    is_accessible_by_car?: boolean;
    safety_features?: string[];
  } | null>(null);
  const [pickupPreferences, setPickupPreferences] = useState<string>("");
  const [relevantPlaces, setRelevantPlaces] = useState<{
    name: string;
    vicinity: string;
    distance: number;
    lat: number;
    lng: number;
    type: string;
  }[]>([]);

  const [directionsToDriver, setDirectionsToDriver] = useState<Step[]>([]);
  const [directionsToPickup, setDirectionsToPickup] = useState<Step[]>([]);
  const [selectedRoute, setSelectedRoute] = useState<'driver' | 'pickup' | null>(null);

  const [activeTab, setActiveTab] = useState<string>("locations");

  const [showIntro, setShowIntro] = useState(true);
  const [showPickupInfo, setShowPickupInfo] = useState(false);
  const [showSafetyModal, setShowSafetyModal] = useState(false);
  const [showPlacesModal, setShowPlacesModal] = useState(false);

  const getRoutePath = useCallback(async (origin: google.maps.LatLngLiteral, destination: google.maps.LatLngLiteral) => {
    if (!map || !window.google) return [];
    const directionsService = new window.google.maps.DirectionsService();
    try {
      const result = await directionsService.route({
        origin: origin,
        destination: destination,
        travelMode: google.maps.TravelMode.DRIVING, // Or dynamically chosen
      });
      if (result.routes && result.routes.length > 0) {
        return result.routes[0].overview_path.map(p => ({ lat: p.lat(), lng: p.lng() }));
      }
    } catch (error) {
      console.error("Error fetching directions for polyline:", error);
    }
    return [];
  }, [map]);


  const handleSetDriver = () => {
    const lat = parseFloat(driverLat);
    const lng = parseFloat(driverLng);
    if (!isNaN(lat) && !isNaN(lng)) {
      const marker = { position: { lat, lng }, label: "Driver", info: `Driver: (${lat}, ${lng})` };
      setDriverMarker(marker);
      setMarkers((prev) => [
        ...prev.filter((m) => m.label !== "Driver"),
        marker,
      ]);
      setCenter({ lat, lng });
      // Clear polylines on new driver input
      setDriverToPickupPolylinePath([]);
      setPassengerToPickupPolylinePath([]);
      setActivePolyline(null);
    }
  };
  const handleSetPassenger = () => {
    const lat = parseFloat(passengerLat);
    const lng = parseFloat(passengerLng);
    if (!isNaN(lat) && !isNaN(lng)) {
      const marker = { position: { lat, lng }, label: "Passenger", info: `Passenger: (${lat}, ${lng})` };
      setPassengerMarker(marker);
      setMarkers((prev) => [
        ...prev.filter((m) => m.label !== "Passenger"),
        marker,
      ]);
      // Clear polylines on new passenger input
      setDriverToPickupPolylinePath([]);
      setPassengerToPickupPolylinePath([]);
      setActivePolyline(null);
    }
  };

  const handleRecommendPickup = async () => {
    if (!driverMarker || !passengerMarker) return;
    setPickupLoading(true);
    setPickupError("");
    setPickupMessage("");
    setDriverToPickupPolylinePath([]);
    setPassengerToPickupPolylinePath([]);
    setActivePolyline(null);

    try {
      const response = await fetch(`${FLASK_API_URL}/api/recommend-pickup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          driver: { lat: driverMarker.position.lat, lng: driverMarker.position.lng },
          passenger: { lat: passengerMarker.position.lat, lng: passengerMarker.position.lng },
          landmarks: landmarks,
          scene: scene,
          building_type: buildingType,
          feedback: pickupPreferences,
        }),
      });
      let data;
      try {
        data = await response.json();
      } catch {
        setPickupError("Failed to parse server response.");
        setPickupLoading(false);
        return;
      }
      if (!response.ok) {
        setPickupError(data.error || "Failed to get pickup recommendation");
        setPickupMessage(data.message || "");
        setPickupLoading(false);
        return;
      }
      setPickupPoint(data.pickup_point);
      setPickupMessage(data.message || "");
      setSafetyValidation(data.safety_validation || null);
      setRelevantPlaces(data.relevant_places || []);
      setDirectionsToDriver(data.directions_to_driver || []);
      setDirectionsToPickup(data.directions_to_pickup || []);
      setSelectedRoute(null);
      setMarkers((prev) => [
        ...prev.filter((m) => m.label !== "Pickup Recommendation"),
        { position: data.pickup_point, label: "Pickup Recommendation", info: data.message },
      ]);
      if (map) map.panTo(data.pickup_point);

      if (driverMarker && data.pickup_point) {
        const driverPath = await getRoutePath(driverMarker.position, data.pickup_point);
        setDriverToPickupPolylinePath(driverPath);
      }
      if (passengerMarker && data.pickup_point) {
        const passengerPath = await getRoutePath(passengerMarker.position, data.pickup_point);
        setPassengerToPickupPolylinePath(passengerPath);
        setActivePolyline('passenger'); // Automatically show passenger polyline by default
      }

      setPickupLoading(false);
    } catch (err) {
      console.error("Pickup recommendation error:", err);
      setPickupError("Error getting pickup recommendation");
      setPickupLoading(false);
      alert("Error getting pickup recommendation");
    }
  };

  const handleShowDirections = (route: 'driver' | 'pickup') => {
    setSelectedRoute(route);
    setShowDirectionsModal(true);
    setCurrentStep(0);
    if (route === 'driver') {
      setSteps(directionsToDriver);
      setActivePolyline('driver'); // Set active polyline to driver
    } else {
      setSteps(directionsToPickup);
      setActivePolyline('passenger'); // Set active polyline to passenger
    }
  };

  const handleShowPolyline = (polylineType: 'driver' | 'passenger') => {
    setActivePolyline(polylineType);
  };

  const handleNextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  function getPickupReason() {
    if (pickupMessage?.toLowerCase().includes("llm")) {
      return "This spot was chosen by AI to minimize walking for the passenger and make it easy for the driver.";
    }
    if (pickupMessage?.toLowerCase().includes("driver's location")) {
      return "Fallback: No better spot found, so the driver's location is used.";
    }
    return pickupMessage || "Recommended as the best middle ground.";
  }

  const { isLoaded } = useJsApiLoader({
    googleMapsApiKey: Maps_API_KEY || "",
    libraries: ["places"],
  });

  const onMapLoad = useCallback((mapInstance: google.maps.Map) => {
    setMap(mapInstance);
    // Try to get user's current location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition((position) => {
        const userLoc = {
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        };
        setCenter(userLoc);
        mapInstance.panTo(userLoc);
        setMarkers([{ position: userLoc, label: "You", info: "Current Location" }]);
      });
    }
  }, []);

  const onPlaceChanged = () => {
    if (searchBox) {
      const place = searchBox.getPlace();
      if (place && place.geometry && place.geometry.location) {
        const loc = {
          lat: place.geometry.location.lat(),
          lng: place.geometry.location.lng(),
        };
        setCenter(loc);
        setMarkers((prev) => [
          ...prev,
          { position: loc, label: place.name || "Searched Place", info: place.formatted_address || "" },
        ]);
        // Clear polylines on new place search
        setDriverToPickupPolylinePath([]);
        setPassengerToPickupPolylinePath([]);
        setActivePolyline(null);
      }
    }
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setProcessing(true);
      setLandmarks([]);
      setProcessedImageUrl("");
      setPickupPoint(null);
      setSteps([]);
      setDriverToPickupPolylinePath([]);
      setPassengerToPickupPolylinePath([]);
      setActivePolyline(null);

      
      const formData = new FormData();
      formData.append("image", file);

      if (driverMarker) {
        formData.append("ref_lat", driverMarker.position.lat.toString());
        formData.append("ref_lng", driverMarker.position.lng.toString());
      }

      try {
        const response = await fetch(`${FLASK_API_URL}/api/upload-image`, {
          method: "POST",
          body: formData,
        });
        if (!response.ok) throw new Error(`Error: ${response.statusText}`);
        const data = await response.json();
        setProcessedImageUrl(`${FLASK_API_URL}${data.image_url}`);
        // Ensure landmarks are geocoded before setting them
        setLandmarks(data.landmarks || []);
        setScene(data.scene || "");
        setBuildingType(data.building_type || "");
      } catch (error) {
        console.error("Image upload error:", error);
        alert("Error uploading or processing image. Please try again.");
      } finally {
        setProcessing(false);
      }
    }
  };

  useEffect(() => {
    if (showIntro) setShowIntro(false);
    console.log("Maps API Key:", process.env.NEXT_PUBLIC_Maps_API_KEY);
  }, []);

  // Traffic Layer
  const mapRef = useRef<google.maps.Map | null>(null);
  useEffect(() => {
    if (map && window.google && mapRef.current !== map) {
      mapRef.current = map;
      const trafficLayer = new window.google.maps.TrafficLayer();
      trafficLayer.setMap(map);
    }
  }, [map]);

  return (
    <div className="flex flex-col md:flex-row h-screen bg-gray-50">
      {/* How it works intro modal */}
      {showIntro && (
        <Dialog open={showIntro} onOpenChange={setShowIntro}>
          <DialogContent className="max-w-lg w-full">
            <DialogHeader>
              <DialogTitle>How it works</DialogTitle>
            </DialogHeader>
            <div className="space-y-3">
              <p>
                <b>Middle Ground Pickup:</b> This app finds a mutually convenient, legal, and safe pickup spot between a driver and a passenger. It uses AI to minimize walking for the passenger and avoid detours for the driver.
              </p>
              <ul className="list-disc pl-5 text-sm text-gray-700">
                <li>Enter driver and passenger locations.</li>
                <li>Optionally upload dashcam images for landmark help.</li>
                <li>Click <b>Find Pickup Options</b> to see the best spot and routes for both users.</li>
                <li>Get step-by-step directions and Street View previews.</li>
              </ul>
              <Button className="w-full mt-2" onClick={() => setShowIntro(false)}>Get Started</Button>
            </div>
          </DialogContent>
        </Dialog>
      )}
      
      {/* Left: Compact Control Panel */}
      <div className="w-full md:w-[400px] flex-shrink-0 bg-white border-r shadow-lg z-10 overflow-hidden">
        <div className="h-full flex flex-col">
          {/* Header */}
          <div className="p-4 border-b bg-gradient-to-r from-blue-50 to-indigo-50">
            <h1 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
              <MapPin className="w-5 h-5" />
              Pickup Coordinator
            </h1>
            <p className="text-xs text-gray-600 mt-1">AI-powered ride coordination</p>
          </div>

          {/* Main Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid grid-cols-3 w-full">
                <TabsTrigger value="locations" className="text-xs">Locations</TabsTrigger>
                <TabsTrigger value="landmarks" className="text-xs">Landmarks</TabsTrigger>
                <TabsTrigger value="routes" className="text-xs">Routes</TabsTrigger>
              </TabsList>

              {/* Step 1: Locations */}
              <TabsContent value="locations" className="space-y-3">
                <Card className="border-0 shadow-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Car className="w-4 h-4" />
                      Driver Location
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex gap-2">
                      <Input 
                        type="text" 
                        placeholder="Lat" 
                        value={driverLat} 
                        onChange={e => setDriverLat(e.target.value)}
                        className="text-xs"
                      />
                      <Input 
                        type="text" 
                        placeholder="Lng" 
                        value={driverLng} 
                        onChange={e => setDriverLng(e.target.value)}
                        className="text-xs"
                      />
                      <Button size="sm" onClick={handleSetDriver} className="text-xs">Set</Button>
                    </div>
                    {driverMarker && (
                      <Badge variant="secondary" className="text-xs">
                        ✓ ({driverMarker.position.lat.toFixed(4)}, {driverMarker.position.lng.toFixed(4)})
                      </Badge>
                    )}
                  </CardContent>
                </Card>

                <Card className="border-0 shadow-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <User className="w-4 h-4" />
                      Passenger Location
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex gap-2">
                      <Input 
                        type="text" 
                        placeholder="Lat" 
                        value={passengerLat} 
                        onChange={e => setPassengerLat(e.target.value)}
                        className="text-xs"
                      />
                      <Input 
                        type="text" 
                        placeholder="Lng" 
                        value={passengerLng} 
                        onChange={e => setPassengerLng(e.target.value)}
                        className="text-xs"
                      />
                      <Button size="sm" onClick={handleSetPassenger} className="text-xs">Set</Button>
                    </div>
                    {passengerMarker && (
                      <Badge variant="secondary" className="text-xs">
                        ✓ ({passengerMarker.position.lat.toFixed(4)}, {passengerMarker.position.lng.toFixed(4)})
                      </Badge>
                    )}
                  </CardContent>
                </Card>

                <Button 
                  className="w-full" 
                  onClick={() => setActiveTab("landmarks")} 
                  disabled={!driverMarker || !passengerMarker}
                  size="sm"
                >
                  Next: Landmarks
                </Button>
              </TabsContent>

              {/* Step 2: Landmarks (optional) */}
              <TabsContent value="landmarks" className="space-y-3">
                <Card className="border-0 shadow-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Upload className="w-4 h-4" />
                      Dashcam Image (Optional)
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Input 
                      type="file" 
                      accept="image/*" 
                      onChange={handleImageUpload}
                      className="text-xs"
                    />
                    {processing && <Badge variant="outline" className="text-xs">Processing...</Badge>}
                    {processedImageUrl && (
                      <div className="space-y-2">
                        <img src={processedImageUrl} alt="Processed" className="w-full h-20 object-cover rounded" />
                        <Accordion type="single" collapsible className="w-full">
                          <AccordionItem value="landmarks" className="border-none">
                            <AccordionTrigger className="text-xs py-2">
                              Detected Landmarks ({landmarks.length})
                            </AccordionTrigger>
                            <AccordionContent className="text-xs">
                              <div className="max-h-20 overflow-y-auto space-y-1">
                                {landmarks.length === 0 && <p className="text-gray-400">No landmarks detected.</p>}
                                {landmarks.map((lm, idx) => (
                                  <div key={idx} className="flex items-center gap-2">
                                    <Badge variant="outline" className="text-xs">
                                      {lm.type === "text" ? lm.text : lm.structure_name}
                                    </Badge>
                                  </div>
                                ))}
                              </div>
                            </AccordionContent>
                          </AccordionItem>
                        </Accordion>
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Button 
                  className="w-full" 
                  onClick={() => setActiveTab("routes")} 
                  disabled={!driverMarker || !passengerMarker}
                  size="sm"
                >
                  Next: Routes
                </Button>
              </TabsContent>

              {/* Step 3: Pickup & Route Options */}
              <TabsContent value="routes" className="space-y-3">
                <Card className="border-0 shadow-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Route className="w-4 h-4" />
                      Pickup Preferences
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Input
                      type="text"
                      placeholder="e.g., 'near coffee shop', 'well-lit area'"
                      value={pickupPreferences}
                      onChange={(e) => setPickupPreferences(e.target.value)}
                      className="text-xs"
                    />
                    <p className="text-xs text-gray-500">
                      Examples: &quot;near coffee shop&quot;, &quot;well-lit&quot;, &quot;not crowded&quot;, &quot;rental car&quot;
                    </p>
                    <Button
                      size="sm"
                      className="w-full"
                      onClick={() => { console.log('Pickup options button clicked'); handleRecommendPickup(); }}
                      disabled={!driverMarker || !passengerMarker}
                    >
                      {pickupLoading ? "Finding Options..." : "Find Pickup Options"}
                    </Button>
                  </CardContent>
                </Card>

                {pickupMessage && (
                  <Alert className="text-xs" variant="info">{pickupMessage}</Alert>
                )}
                {pickupError && (
                  <Alert className="text-xs" variant="destructive">{pickupError}</Alert>
                )}

                                {pickupPoint && (
                  <div className="space-y-2">
                    {/* Compact Pickup Info */}
                    <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-3 border border-green-200">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                          <span className="text-sm font-medium text-green-800">Pickup Location</span>
                        </div>
                        <button onClick={() => setShowPickupInfo(true)} className="text-green-600 hover:text-green-800">
                          <Info size={12} />
                        </button>
                      </div>
                      <div className="text-xs text-green-700 font-mono mb-1">
                        ({pickupPoint.lat.toFixed(4)}, {pickupPoint.lng.toFixed(4)})
                      </div>
                      <div className="text-xs text-green-600 italic">{getPickupReason()}</div>
                    </div>

                    {/* Compact Safety & Places Row */}
                    <div className="grid grid-cols-2 gap-2">
                      {/* Safety Score */}
                      {safetyValidation && (
                        <button 
                          onClick={() => setShowSafetyModal(true)}
                          className="bg-white rounded-lg p-2 border border-gray-200 hover:border-gray-300 hover:bg-gray-50 transition-colors w-full text-left"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium text-gray-700">Safety</span>
                            <Badge variant={safetyValidation.safety_score > 0.7 ? "default" : safetyValidation.safety_score > 0.5 ? "secondary" : "destructive"} className="text-xs">
                              {Math.round(safetyValidation.safety_score * 100)}%
                            </Badge>
                          </div>
                          <div className="flex items-center gap-1 text-xs text-gray-600">
                            <div className={`w-1.5 h-1.5 rounded-full ${safetyValidation.is_accessible_by_car ? 'bg-green-500' : 'bg-red-500'}`}></div>
                            <span>{safetyValidation.is_accessible_by_car ? 'Car Accessible' : 'Not Accessible'}</span>
                          </div>
                        </button>
                      )}

                      {/* Nearby Places */}
                      <button 
                        onClick={() => setShowPlacesModal(true)}
                        className="bg-white rounded-lg p-2 border border-gray-200 hover:border-gray-300 hover:bg-gray-50 transition-colors w-full text-left"
                        disabled={!relevantPlaces || relevantPlaces.length === 0}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-medium text-gray-700">Nearby</span>
                          <Badge variant="outline" className="text-xs">
                            {relevantPlaces && relevantPlaces.length > 0 ? relevantPlaces.length : 0}
                          </Badge>
                        </div>
                        <div className="text-xs text-gray-600">
                          {relevantPlaces && relevantPlaces.length > 0 
                            ? relevantPlaces[0]?.name || 'No places found'
                            : 'No places found'
                          }
                        </div>
                      </button>
                    </div>

                    {/* Compact Route Controls */}
                    <div className="bg-gray-50 rounded-lg p-2">
                      <div className="flex gap-1 mb-2">
                        <Button
                          size="sm"
                          onClick={() => handleShowPolyline('driver')}
                          variant={activePolyline === 'driver' ? 'default' : 'outline'}
                          disabled={driverToPickupPolylinePath.length === 0}
                          className="text-xs flex-1 h-6"
                        >
                          Driver
                        </Button>
                        <Button
                          size="sm"
                          onClick={() => handleShowPolyline('passenger')}
                          variant={activePolyline === 'passenger' ? 'default' : 'outline'}
                          disabled={passengerToPickupPolylinePath.length === 0}
                          className="text-xs flex-1 h-6"
                        >
                          Passenger
                        </Button>
                      </div>
                      <div className="flex gap-1">
                        <Button 
                          size="sm" 
                          className="text-xs flex-1 h-6" 
                          onClick={() => handleShowDirections('driver')} 
                          disabled={directionsToDriver.length === 0}
                        >
                          To Driver
                        </Button>
                        <Button 
                          size="sm" 
                          className="text-xs flex-1 h-6" 
                          onClick={() => handleShowDirections('pickup')} 
                          disabled={directionsToPickup.length === 0}
                        >
                          To Pickup
                        </Button>
                      </div>
                    </div>
                  </div>
                )}

                <Button className="w-full" onClick={() => setActiveTab("locations")} size="sm">
                  Start Over
                </Button>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>

      {/* Pickup Info Modal */}
      <Dialog open={showPickupInfo} onOpenChange={setShowPickupInfo}>
        <DialogContent className="max-w-md w-full">
          <DialogHeader>
            <DialogTitle>Why this pickup spot?</DialogTitle>
          </DialogHeader>
          <div className="space-y-2">
            <p>{getPickupReason()}</p>
            <Button className="w-full mt-2" onClick={() => setShowPickupInfo(false)}>Close</Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Safety Analysis Modal */}
      <Dialog open={showSafetyModal} onOpenChange={setShowSafetyModal}>
        <DialogContent className="max-w-lg w-full">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${safetyValidation?.is_safe ? 'bg-green-500' : 'bg-red-500'}`}></div>
              Safety Analysis
            </DialogTitle>
          </DialogHeader>
          {safetyValidation && (
            <div className="space-y-4">
              {/* Safety Score */}
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="font-medium">Safety Score</span>
                <Badge variant={safetyValidation.safety_score > 0.7 ? "default" : safetyValidation.safety_score > 0.5 ? "secondary" : "destructive"} className="text-sm">
                  {Math.round(safetyValidation.safety_score * 100)}%
                </Badge>
              </div>

              {/* Car Accessibility */}
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="font-medium">Car Accessibility</span>
                <Badge variant={safetyValidation.is_accessible_by_car ? "default" : "destructive"} className="text-sm">
                  {safetyValidation.is_accessible_by_car ? 'Accessible' : 'Not Accessible'}
                </Badge>
              </div>

              {/* Lighting Quality */}
              {safetyValidation.lighting_quality && (
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="font-medium">Lighting Quality</span>
                  <span className="capitalize text-sm">{safetyValidation.lighting_quality}</span>
                </div>
              )}

              {/* Recommendations */}
              {safetyValidation.recommendations && safetyValidation.recommendations.length > 0 && (
                <div className="space-y-2">
                  <h4 className="font-medium">Recommendations</h4>
                  <div className="space-y-2">
                    {safetyValidation.recommendations.map((rec: string, idx: number) => (
                      <div key={idx} className="flex items-start gap-2 p-2 bg-blue-50 rounded">
                        <span className="text-blue-600">•</span>
                        <span className="text-sm">{rec}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Safety Features */}
              {safetyValidation.safety_features && safetyValidation.safety_features.length > 0 && (
                <div className="space-y-2">
                  <h4 className="font-medium text-green-700">Safety Features</h4>
                  <div className="space-y-2">
                    {safetyValidation.safety_features.map((feature: string, idx: number) => (
                      <div key={idx} className="flex items-center gap-2 p-2 bg-green-50 rounded">
                        <span className="text-green-600">✅</span>
                        <span className="text-sm">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Issues Found */}
              {safetyValidation.issues_found && safetyValidation.issues_found.length > 0 && (
                <div className="space-y-2">
                  <h4 className="font-medium text-red-700">Issues Detected</h4>
                  <div className="space-y-2">
                    {safetyValidation.issues_found.map((issue: string, idx: number) => (
                      <div key={idx} className="flex items-center gap-2 p-2 bg-red-50 rounded">
                        <span className="text-red-600">⚠️</span>
                        <span className="text-sm">{issue}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          <Button className="w-full mt-4" onClick={() => setShowSafetyModal(false)}>Close</Button>
        </DialogContent>
      </Dialog>

      {/* Nearby Places Modal */}
      <Dialog open={showPlacesModal} onOpenChange={setShowPlacesModal}>
        <DialogContent className="max-w-lg w-full">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <MapPin className="w-5 h-5" />
              Nearby Places
            </DialogTitle>
          </DialogHeader>
          {relevantPlaces && relevantPlaces.length > 0 ? (
            <div className="space-y-3">
              {relevantPlaces.map((place, idx) => (
                <div key={idx} className="p-3 border rounded-lg hover:bg-gray-50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-medium text-sm">{place.name}</h4>
                      <p className="text-xs text-gray-600 mt-1">{place.vicinity}</p>
                      <div className="flex items-center gap-2 mt-2">
                        <Badge variant="outline" className="text-xs">{place.type}</Badge>
                        <span className="text-xs text-gray-500">{place.distance.toFixed(0)}m away</span>
                      </div>
                    </div>
                    <div className="text-xs text-gray-400">
                      ({place.lat.toFixed(4)}, {place.lng.toFixed(4)})
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <MapPin className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Nearby Places Found</h3>
              <p className="text-sm text-gray-600">
                No relevant places were found near the pickup location based on your preferences.
              </p>
            </div>
          )}
          <Button className="w-full mt-4" onClick={() => setShowPlacesModal(false)}>Close</Button>
        </DialogContent>
      </Dialog>

      {/* Right: Map Panel */}
      <div className="flex-1 relative">
        {isLoaded && (
          <GoogleMap
            mapContainerStyle={containerStyle}
            center={center}
            zoom={15}
            onLoad={onMapLoad}
          >
            {/* Floating Search Box */}
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20 w-[350px] max-w-full">
              <Autocomplete
                onLoad={(autocomplete: google.maps.places.Autocomplete) => setSearchBox(autocomplete)}
                onPlaceChanged={onPlaceChanged}
              >
                <Input
                  type="text"
                  placeholder="Search places..."
                  className="w-full p-2 border rounded shadow"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                />
              </Autocomplete>
            </div>
            {/* Markers */}
            {markers.map((marker, idx) => (
              <Marker
                key={idx}
                position={marker.position}
                label={marker.label}
                icon={
                  marker.label === "Pickup Recommendation"
                    ? {
                        url: "http://maps.google.com/mapfiles/ms/icons/red-dot.png",
                        scaledSize: new window.google.maps.Size(44, 44),
                      }
                    : marker.label === "Driver"
                    ? {
                        url: "http://maps.google.com/mapfiles/ms/icons/blue-dot.png",
                        scaledSize: new window.google.maps.Size(40, 40),
                      }
                    : marker.label === "Passenger"
                    ? {
                        url: "http://maps.google.com/mapfiles/ms/icons/green-dot.png",
                        scaledSize: new window.google.maps.Size(40, 40),
                      }
                    : undefined
                }
                onClick={() => setSelectedMarker(marker)}
              />
            ))}
            {/* InfoWindow for selected marker */}
            {selectedMarker && (
              <InfoWindow
                position={selectedMarker.position}
                onCloseClick={() => setSelectedMarker(null)}
              >
                <div>
                  <strong>{selectedMarker.label}</strong>
                  <div>{selectedMarker.info}</div>
                </div>
              </InfoWindow>
            )}
            {/* Only show the active polyline */}
            {activePolyline === 'driver' && driverToPickupPolylinePath.length > 0 && (
              <Polyline
                path={driverToPickupPolylinePath}
                options={{ strokeColor: "#FF0000", strokeWeight: 4, strokeOpacity: 0.7 }} // Red for Driver's route
              />
            )}
            {activePolyline === 'passenger' && passengerToPickupPolylinePath.length > 0 && (
              <Polyline
                path={passengerToPickupPolylinePath}
                options={{ strokeColor: "#0000FF", strokeWeight: 4, strokeOpacity: 0.7 }} // Blue for Passenger's route
              />
            )}
            {/* Pickup Point Marker (highlighted) */}
            {pickupPoint && (
              <Marker
                position={{ lat: pickupPoint.lat, lng: pickupPoint.lng }}
                icon={{
                  url: "http://maps.google.com/mapfiles/ms/icons/red-dot.png",
                  scaledSize: new window.google.maps.Size(44, 44),
                }}
                label="Pickup"
              />
            )}
            {/* Transit Layer */}
            <TransitLayer />
          </GoogleMap>
        )}
      </div>

      {/* Directions Modal as Dialog */}
      <Dialog open={showDirectionsModal} onOpenChange={setShowDirectionsModal}>
        <DialogContent className="max-w-2xl w-full">
          <DialogHeader>
            <DialogTitle>
              {selectedRoute === 'driver' ? 'Directions to Driver' : 'Directions to Pickup Spot'}
            </DialogTitle>
          </DialogHeader>
          {steps.length > 0 && (
            <div className="flex flex-col gap-4">
              <div className="w-full aspect-video bg-gray-200 rounded-lg overflow-hidden shadow-inner">
                <img
                  src={steps[currentStep].photo_url}
                  alt={`Street View for step ${currentStep + 1}`}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex-grow overflow-y-auto p-3 bg-gray-50 rounded-lg border min-h-[120px]">
                <p className="text-lg font-medium text-gray-700 whitespace-pre-line">
                  {steps[currentStep].text}
                </p>
                {/* Show detected landmarks ONLY if selectedRoute is 'driver' */}
                {selectedRoute === 'driver' && landmarks.length > 0 && (
                  <div className="mt-2 text-xs text-gray-500">
                    <b>Landmarks to look for:</b> {landmarks.map((lm) => lm.text || lm.structure_name).filter(Boolean).join(", ")}
                  </div>
                )}
              </div>
              <div className="flex justify-between items-center mt-2">
                <Button onClick={handlePrevStep} disabled={currentStep === 0} variant="outline">
                  Previous
                </Button>
                <div className="text-sm text-gray-500">
                  {currentStep + 1} / {steps.length}
                </div>
                {currentStep < steps.length - 1 ? (
                  <Button onClick={handleNextStep}>
                    Next
                  </Button>
                ) : (
                  <Button onClick={() => setShowDirectionsModal(false)} variant="secondary">
                    Done
                  </Button>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}