import asyncio, sys, time, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Save to cameratests/captures/ regardless of where you run from
OUTPUT_DIR = Path(__file__).resolve().parent / "captures"
OUTPUT_DIR.mkdir(exist_ok=True)

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_pi_camera_manager import qpsPiCameraManager
from quackops_pi.vision.qps_marker_detector import qpsMarkerDetector

async def main():
    config = qpsConfig()
    cam = qpsPiCameraManager(config); cam.start()
    detector = qpsMarkerDetector(config)
    await asyncio.sleep(1)  # let camera settle
    print('Hold marker steady in view...')
    await asyncio.sleep(3)
    
    saved = 0
    while saved < 5:
        frame = cam.get_frame()
        if frame is None: 
            await asyncio.sleep(0.05); continue
        dets = await detector.detect(frame)
        if dets:
            for d in dets:
                pts = d.corners.astype(int)
                cv2.polylines(frame, [pts], True, (0,255,0), 3)
                cx, cy = int(d.center_px[0]), int(d.center_px[1])
                cv2.drawMarker(frame, (cx,cy), (0,0,255), cv2.MARKER_CROSS, 30, 3)
                cv2.putText(frame, f'ID:{d.marker_id}', (cx+10, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            output_path = OUTPUT_DIR / f"proof_{saved}.jpg"
            cv2.imwrite(str(output_path), frame)
            print(f'Saved {output_path} with {len(dets)} marker(s)')
            saved += 1
            await asyncio.sleep(0.5)
        else:
            await asyncio.sleep(0.05)
    cam.stop()

asyncio.run(main())