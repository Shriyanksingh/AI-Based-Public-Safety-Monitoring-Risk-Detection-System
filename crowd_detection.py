import cv2
import numpy as np
from collections import deque
from datetime import datetime
import time
from ultralytics import YOLO
import torch
from pathlib import Path

class CrowdSafetyMonitor:
    """Monitor crowds for safety hazards - works well for malls, transit hubs, events."""
    
    def __init__(self, video_path, modelSize='n', useGPU=True, frameSkip=1):
        """Initialize the monitor.
        
        frameSkip: skip frames for faster processing (tested with mall footage)
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frameSkip = frameSkip
        
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_video = fps
        
        if frameSkip > 1:
            print(f"Skipping frames for performance...")
        
        # GPU or CPU
        self.device = 'cuda' if useGPU and torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        
        # Load model - might take time first run
        try:
            self.model = YOLO(f'yolo11{modelSize}.pt')
            self.model.to(self.device)
        except FileNotFoundError:
            print("Downloading model...")
            self.model = YOLO(f'yolo11{modelSize}.pt')
            self.model.to(self.device)
        except Exception as err:
            raise RuntimeError(f"Failed loading YOLO: {err}")
        
        # Tracking data
        self.frameHist = deque(maxlen=30)
        self.densHist = deque(maxlen=90)
        self.motHist = deque(maxlen=60)
        self.velHist = deque(maxlen=30)
        
        # Thresholds - tuned for mall environments
        # 50 people = low occupancy, 150 = medium, 250+ = crowded (adjust per venue)
        self.baseSamples = 60
        self.confThresh = 0.25
        self.iouThresh = 0.45
        
        # Optical flow setup
        self.prevFrame = None
        self.flow_params = dict(
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Alerts
        self.alerts = []
        self.cooldown = {}  # dont spam alerts
        self.cooldown_frames = 30
        
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.proc_times = deque(maxlen=100)
        
        self.baselineSet = False
        self.baseDensity = None
        self.baseMotion = None
        
    def detect_people_yolo11(self, frame):
        """Find people in frame."""
        results = self.model.predict(
            frame, conf=self.confThresh, iou=self.iouThresh,
            classes=[0], verbose=False, device=self.device
        )
        
        boxes = []
        confidences = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                confidences.append(conf)
        
        return len(boxes), boxes, confidences
    
    def calculate_advanced_optical_flow(self, frame):
        """Compute motion from optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.prevFrame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prevFrame, gray, None, **self.flow_params
            )
            
            magnitude,angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_motion= np.mean(magnitude)
            max_motion= np.max(magnitude)
            std_motion= np.std(magnitude)
            
            # Calculate motion percentiles for robust statistcs
            p90_motion= np.percentile(magnitude,90)
            p95_motion= np.percentile(magnitude,95)
            
            # Directional coheranse (how aligned is the motion)
            flow_x_std= np.std(flow[...,0])
            flow_y_std= np.std(flow[...,1])
            directional_variance= flow_x_std + flow_y_std
            
            # Motion entopy (measure of chaos)
            hist, _= np.histogram(magnitude, bins=20, range=(0, 20))
            hist= hist / hist.sum()
            entropy= -np.sum(hist * np.log2(hist + 1e-10))
            
            self.prevFrame = gray
            
            return {
                'avg_motion': avg_motion,
                'max_motion': max_motion,
                'std_motion': std_motion,
                'p90_motion': p90_motion,
                'p95_motion': p95_motion,
                'directional_variance': directional_variance,
                'entropy': entropy,
                'magnitude_map': magnitude
            }
        
        self.prev_gray= gray
        return None
    
    def calculate_crowd_density_metrics(self, count, boxes, frameShape):
        """Get density from people count."""
        h, w = frameShape[:2]
        area = h * w
        density = count / (area / 10000)
        
        if len(boxes) > 0:
            # Calculate spatial distribution metrics
            centers= []
            areas = []
            
            for box in boxes:
                x,y,w_box,h_box= box
                center_x= x + w_box/2
                center_y= y + h_box/2
                centers.append([center_x,center_y])
                areas.append(w_box*h_box)
            
            centers= np.array(centers)
            
            # Calculate clustering coefficent (standard deviation of distances)
            if len(centers) > 1:
                # Pairwise destances
                from scipy.spatial.distance import pdist
                try:
                    distances = pdist(centers)
                    clustering = np.std(distances) / (np.mean(distances) + 1e-6)
                except:
                    clustering = 0
            else:
                clustering = 0
            
            # Average person sice (can indicate distance from camera)
            avg_person_area = np.mean(areas) if areas else 0
            
            # Coverage ratio (how much of frame is occupied)
            total_person_area = sum(areas)
            coverage_ratio = total_person_area / area
            
        else:
            clustering = 0
            avg_person_area = 0
            coverage_ratio = 0
        
        return {
            'density_ratio': density,
            'count': count,
            'clustering': 0,  # TODO: implement properly
            'avg_person_area': 0,
            'coverage_ratio': 0
        }
    
    def establish_baseline(self):
        """Set baseline from initial frames."""
        if len(self.densHist) >= self.baseSamples and not self.baselineSet:
            self.baseDensity = {
                'mean': np.mean([d['density_ratio'] for d in self.densHist]),
                'std': np.std([d['density_ratio'] for d in self.densHist])
            }
            
            self.baseMotion = {
                'mean': np.mean([m['avg_motion'] for m in self.motHist if m]),
                'std': np.std([m['avg_motion'] for m in self.motHist if m])
            }
            
            self.baselineSet = True
            print(f"Baseline established: Density μ={self.baseDensity['mean']:.2f}, "
                  f"Motion μ={self.baseMotion['mean']:.2f}")
    
    def calculate_z_score(self, val, baseline):
        # quick z-calc
        if baseline['std'] < 0.01:
            return 0
        return (val-baseline['mean'])/baseline['std']
    
    def detect_anomalies(self, density_metrics, motion_metrics):
        """Detect anomalies using statisticle methods."""
        if not self.baselineSet:
            return []
        
        anomalies = []
        current_time = self.frame_count/self.fps_video
        person_count = density_metrics['count']
        
        if person_count < 50:
            if self.check_alert_cooldown('LOW_DENSITY'):
                anomalies.append({
                    'type': 'LOW_DENSITY',
                    'risk': 'LOW',
                    'reason': f'Low crowd desnity: {person_count} people detected',
                    'confidence': 0.6
                })
        elif 50 <= person_count <= 150:
            if self.check_alert_cooldown('MEDIUM_DENSITY'):
                anomalies.append({
                    'type': 'MEDIUM_DENSITY',
                    'risk': 'MEDIUM',
                    'reason': f'Mediun crowd density: {person_count} people detected',
                    'confidence': 0.8
                })
        elif person_count > 150:
            if self.check_alert_cooldown('HIGH_DENSITY'):
                anomalies.append({
                    'type': 'HIGH_DENSITY',
                    'risk': 'HIGH',
                    'reason': f'Hight crowd density: {person_count} people detected',
                    'confidence': 0.95
                })
        
        # 2. MOTION-BASED ANOMALIES
        if motion_metrics:
            motion_z = self.calculate_z_score(
                motion_metrics['avg_motion'],
                self.baseMotion
            )
            
            # Sudden Acceleration
            if motion_z > 3.5:
                if self.check_alert_cooldown('ACCELERATION'):
                    risk = 'HIGH' if motion_z > 5.0 else 'MEDIUM'
                    anomalies.append({
                        'type': 'SUDDEN_ACCELERATION',
                        'risk': risk,
                        'reason': f'Rapid crowd movement: motion={motion_metrics["avg_motion"]:.2f} (z-score: {motion_z:.2f})',
                        'confidence': min(motion_z / 6.0, 1.0)
                    })
            
            # Chaotic Movement (high entropy + high variance)
            if motion_metrics['entropy'] > 3.5 and motion_metrics['directional_variance'] > 50:
                if self.check_alert_cooldown('CHAOS'):
                    anomalies.append({
                        'type': 'CHAOTIC_MOVEMENT',
                        'risk': 'MEDIUM',
                        'reason': f'Irregular movement patterns detected (entropy: {motion_metrics["entropy"]:.2f})',
                        'confidence': 0.75
                    })
            
            # High velocity with high density (critical condition)
            if (motion_metrics['p95_motion'] > 15 and 
                density_metrics['density_ratio'] > self.baseDensity['mean'] * 1.5):
                if self.check_alert_cooldown('CRITICAL'):
                    anomalies.append({
                        'type': 'CRITICAL_CONDITION',
                        'risk': 'HIGH',
                        'reason': 'High-density area with rapid movement - stampede risk',
                        'confidence': 0.9
                    })
        
        # 3. TEMPORAL ANOMALIES
        # Rapid Dispersal
        if len(self.densHist) >= 30:
            recent_densities = [d['density_ratio'] for d in list(self.densHist)[-30:]]
            density_change_rate = (recent_densities[-1] - recent_densities[0]) / 30
            
            if density_change_rate < -2.0 and recent_densities[0] > 20:
                if self.check_alert_cooldown('DISPERSAL'):
                    anomalies.append({
                        'type': 'RAPID_DISPERSAL',
                        'risk': 'MEDIUM',
                        'reason': f'Rapid crowd dispersal detected (rate: {density_change_rate:.2f}/frame)',
                        'confidence': 0.8
                    })
        
        return anomalies
    
    def check_alert_cooldown(self, atype):
        """Cooldown logic."""
        frame = self.frame_count
        
        if atype in self.cooldown:
            if frame - self.cooldown[atype] < self.cooldown_frames:
                return False
        
        self.cooldown[atype] = frame
        return True
    
    def generate_alert(self, anomalies, timestamp):
        """Generate detailled alerts with confidence scores"""
        for anomaly in anomalies:
            alert = {
                'timestamp': timestamp,
                'frame': self.frame_count,
                'risk_level': anomaly['risk'],
                'type': anomaly['type'],
                'explanation': anomaly['reason'],
                'confidence': anomaly.get('confidence', 1.0)
            }
            
            self.alerts.append(alert)
            
            # Console output with color coding
            print("\n" + "="*70)
            print(f"ALERT DETECTED - {anomaly['risk']} RISK")
            print("="*70)
            print(f"Timestamp: {alert['timestamp']}")
            print(f"Frame: {alert['frame']}/{self.total_frames}")
            print(f"Type: {alert['type']}")
            print(f"Confidence: {alert['confidence']:.1%}")
            print(f"Details: {alert['explanation']}")
            print("="*70 + "\n")
    
    def process_video(self, display=False, save_output=False):
        """Main video procesing loop."""
        print("\n" + "="*70)
        print("PUBLIC SAFETY MONITORING SISTEM v2.0 (YOLOv11)")
        print("="*70)
        print(f"Video: {self.video_path}")
        print(f"Total Frames: {self.total_frames}")
        print(f"FPS: {self.fps_video}")
        print(f"Device: {self.device}")
        print("="*70 + "\n")
        
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output_annotated.mp4', fourcc, self.fps_video, 
                                (640, 480))
        
        while True:
            frame_start = time.time()
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            self.frame_count += 1
            
            # Skip frames for CPU optimization
            if self.frame_count % self.frameSkip != 0:
                continue
            
            current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # Resize for processing
            frame = cv2.resize(frame, (640, 480))
            
            # YOLO Detection
            person_count, boxes, confidences = self.detect_people_yolo11(frame)
            
            # Density Analysis
            density_metrics = self.calculate_crowd_density_metrics(
                person_count, boxes, frame.shape
            )
            self.densHist.append(density_metrics)
            
            # Motion Analysis
            motion_metrics = self.calculate_advanced_optical_flow(frame)
            if motion_metrics:
                self.motHist.append(motion_metrics)
            
            # Establish baseline
            self.establish_baseline()
            
            # Anomaly Detection
            if self.baselineSet:
                anomalies = self.detect_anomalies(density_metrics, motion_metrics)
                
                if anomalies:
                    self.generate_alert(anomalies, current_time)
            
            # Performance tracking
            frame_time = time.time() - frame_start
            self.proc_times.append(frame_time)
            
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed
                avg_frame_time = np.mean(self.proc_times)
                
                print(f"Frame {self.frame_count}/{self.total_frames} | "
                      f"People: {person_count} | "
                      f"Density: {density_metrics['density_ratio']:.1f} | "
                      f"Motion: {(motion_metrics['avg_motion'] if motion_metrics else 0):.2f} | "
                      f"FPS: {self.fps:.1f} | "
                      f"Process: {avg_frame_time*1000:.1f}ms")
            
            # Visualization
            if display or save_output:
                vis_frame = self.visualize_frame(
                    frame, person_count, density_metrics, 
                    motion_metrics, boxes
                )
                
                if display:
                    cv2.imshow('Public Safety Monitor', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if save_output:
                    out.write(vis_frame)
        
        self.cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
        self.print_summary()
    
    def visualize_frame(self, frame, count, density_metrics, motion_metrics, boxes):
        """Visualize frame with metrics overllay."""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw detection boxes
        for box in boxes:
            x, y, w_box, h_box = box
            cv2.rectangle(display, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        
        # Create info panel
        panel_height = 150
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Add metrics
        cv2.putText(panel, f"People Count: {count}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(panel, f"Density Ratio: {density_metrics['density_ratio']:.2f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if motion_metrics:
            cv2.putText(panel, f"Avg Motion:{motion_metrics['avg_motion']:.2f}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(panel, f"Motion Entropy: {motion_metrics['entropy']:.2f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(panel, f"Frame: {self.frame_count}/{self.total_frames}", (10, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status indicator
        status_color = (0, 255, 0)  # Green = Normal
        status_text = "NORMAL"
        
        if self.baselineSet and motion_metrics:
            motion_z = self.calculate_z_score(
                motion_metrics['avg_motion'],
                self.baseMotion
            )
            if motion_z > 3.5:
                status_color = (0, 0, 255)
                status_text = "ALERT"
        
        cv2.circle(panel,(w-50,75),30,status_color, -1)
        cv2.putText(panel, status_text,(w-120, 130),
                   cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255), 2)
        
        # Combine frame and panel
        combined = np.vstack([display, panel])
        
        return combined
    
    def print_summary(self):
        print("\nSummary:")
        print(f"Frames: {self.frame_count}")
        print(f"FPS: {self.fps:.1f}")
        print(f"Alerts: {len(self.alerts)}")
        
        if self.alerts:
            print(f"\nFirst few alerts:")
            for a in self.alerts[:3]:
                print(f"  - {a['type']}: {a['explanation']}")
        else:
            print("No alerts triggered")
        
        # Performance metrics
        false_positive_estimate = len([a for a in self.alerts if a['confidence'] < 0.7])
        if self.alerts:
            accuracy_estimate = ((len(self.alerts) - false_positive_estimate) / 
                               len(self.alerts)) * 100
            print(f"\n Estimated Accuracy: {accuracy_estimate:.1f}%")
            print(f"Potential False Positives: {false_positive_estimate}")
        
        print("\n" + "="*70)
        
        self.save_alerts()
    
    def save_alerts(self):
        """Save alert log to fille."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"safety_alerts_{timestamp}.txt"
        
        with open(log_file, 'w') as f:
            f.write("PUBLIC SAFETY MONITORING SYSTEM v2.0 - ALERT LOG\n")
            f.write("="*70 + "\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames: {self.frame_count}\n")
            f.write(f"Total Alerts: {len(self.alerts)}\n")
            f.write("="*70 + "\n\n")
            for i, alert in enumerate(self.alerts, 1):
                f.write(f"Alert #{i}\n")
                f.write(f"Timestamp: {alert['timestamp']}\n")
                f.write(f"Frame: {alert['frame']}/{self.total_frames}\n")
                f.write(f"Risk Level: {alert['risk_level']}\n")
                f.write(f"Type: {alert['type']}\n")
                f.write(f"Confidence: {alert['confidence']:.1%}\n")
                f.write(f"Explanation: {alert['explanation']}\n")
                f.write("-"*70 + "\n\n")
        print(f" Detailed alerts saved to: {log_file}")


if __name__ =="__main__":
    video_path = "crowd_video.mp4"
    monitor = CrowdSafetyMonitor(
        video_path=video_path,
        modelSize='n',
        useGPU=False,
        frameSkip=2
    )
    monitor.process_video(display=False, save_output=False)
    
    print("\nProcessing complet!")
    print("Check the generated log file for detailled alert history")