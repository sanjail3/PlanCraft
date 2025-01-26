# app.py (Flask Backend)
import os
import uuid
import time
import json
from queue import Queue
from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
from agent import PlanAgent
from threading import Thread
from pathlib import Path

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/generated'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory storage for tasks with message queues
tasks = {}

class APIPlanAgent(PlanAgent):
    def __init__(self, prompt, task_id):
        super().__init__(prompt)
        self.task_id = task_id
        self.progress = 0

    def update_progress(self, value):
        self.progress = value
        self._send_message({'type': 'progress', 'progress': value})

    def add_message(self, content, msg_type='info', data=None):
        message = {
            'type': msg_type,
            'content': content,
            'timestamp': time.time(),
            'data': data
        }
        self._send_message(message)

    def _send_message(self, message):
        """Send message to the task's queue"""
        if self.task_id in tasks:
            tasks[self.task_id]['queue'].put(message)

    def save_plot_image(self, image_bytes):
        filename = f"{self.task_id}_plot.png"
        dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(dest, 'wb') as f:
            f.write(image_bytes)
        return f"/static/generated/{filename}"

    def save_file(self, file_path, file_type):
        if not os.path.exists(file_path):
            return None
        ext = Path(file_path).suffix
        filename = f"{self.task_id}_{file_type}{ext}"
        dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.rename(file_path, dest)
        return f"/static/generated/{filename}"

    def run_task(self):
        try:
            self.prepare_agent()
            for event in self.run_agent():
                if event.get('event') == 'plan_generation':
                    self.update_progress(20)
                    self.add_message('Generating initial plan...', 'plan', event['message'])

                elif event.get('event') == 'intermediate_step':
                    tool_name = event.get('tool')
                    metadata = event.get('metadata', {})

                    if tool_name == "plot_generator":
                        self.update_progress(50)
                        plot_data = {
                            "instruction": event["tool_input"].get("plot_instruction"),
                            "code": metadata.get("plot_gen_code"),
                            "error": metadata.get("error")
                        }
                        
                        if not metadata.get("error"):
                            image_bytes = event["step_response"]["image"]["source"]["bytes"]
                            img_path = self.save_plot_image(image_bytes)
                            plot_data["image_url"] = img_path
                        
                        self.add_message("Plot generation update", 'plot', plot_data)

                    elif tool_name == 'house_plan_generator':
                        self.update_progress(40)
                        if not metadata.get('error'):
                            images = []
                            for img in metadata.get('image_path', []):
                                img_path = self.save_file(img['image'], 'image')
                                images.append(img_path)
                            self.add_message('Floor plans generated', 'floorplan', {'images': images})
                        else:
                            self.add_message('Floor plan generation failed', 'error', 
                                          {'error': 'Floor plan generation failed'})

                    elif tool_name == '3d_model_generator':
                        self.update_progress(70)
                        if not metadata.get('error'):
                            model_urls = [self.save_file(url, 'model') for url in metadata.get('model_urls', [])]
                            thumbnail = self.save_file(metadata.get('thumbnail', ''), 'image')
                            self.add_message('3D models generated', '3dmodel', {
                                'model_urls': model_urls,
                                'thumbnail': thumbnail
                            })
                        else:
                            self.add_message('3D model generation failed', 'error', 
                                          {'error': '3D model generation failed'})

            self.update_progress(100)
            self.add_message('Processing complete', 'success')
            self._send_message({'type': 'status', 'status': 'completed'})

        except Exception as e:
            self.add_message('An error occurred', 'error', {'error': str(e)})
            self._send_message({'type': 'status', 'status': 'error', 'error': str(e)})
        finally:
            # Cleanup task after 5 minutes
            Thread(target=self._cleanup_task).start()

    def _cleanup_task(self):
        time.sleep(300)  # Keep task for 5 minutes after completion
        if self.task_id in tasks:
            del tasks[self.task_id]

@app.route('/api/submit', methods=['POST'])
def submit_request():
    data = request.json
    task_id = str(uuid.uuid4())
    
    tasks[task_id] = {
        'queue': Queue(),
        'start_time': time.time()
    }
    
    agent = APIPlanAgent(data['prompt'], task_id)
    thread = Thread(target=agent.run_task)
    thread.start()
    
    return {'task_id': task_id}

@app.route('/api/stream/<task_id>')
def stream_updates(task_id):
    def event_stream():
        task = tasks.get(task_id)
        if not task:
            yield 'event: error\ndata: {"error": "Invalid task ID"}\n\n'
            return

        queue = task['queue']
        while True:
            try:
                message = queue.get(timeout=10)
                if message.get('type') == 'status':
                    yield f"event: {message['status']}\ndata: {json.dumps(message)}\n\n"
                    break
                yield f"data: {json.dumps(message)}\n\n"
            except Exception as e:
                yield 'event: ping\ndata: {"status": "alive"}\n\n'

    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/static/generated/<filename>')
def serve_generated(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)