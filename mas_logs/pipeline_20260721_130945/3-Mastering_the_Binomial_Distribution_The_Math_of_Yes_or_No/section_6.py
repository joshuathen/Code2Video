from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section6Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title = "Real-World Application: Quality Control"
        lecture_lines = [
            "These tools help predict outcomes in manufacturing and science.",
            "Consider a factory checking sensors for potential defects.",
            "The formula calculates the exact risk for every batch."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Colors
        COLOR_FACTORY = "#888888"
        COLOR_SENSOR = "#CCCCCC"
        COLOR_DEFECT = "#FF5555"
        COLOR_SUCCESS = "#55FF55"

        # Assets
        FACTORY_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/factory.svg"
        SENSOR_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg"

        # === Animation for Lecture Line 1 ===
        # Color change for the lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_FACTORY))

        # Factory asset (Issue 24) used as the inspector (Issue 39 calls it drone)
        drone = SVGMobject(FACTORY_ASSET).set_color(COLOR_FACTORY)
        # Fix: Drone/Factory scale to 0.8 at B2 (Issue 39)
        self.place_at_grid(drone, 'B2', scale_factor=0.8)
        
        # 10 sensor icons (Issue 24)
        sensors = VGroup(*[
            SVGMobject(SENSOR_ASSET).set_color(COLOR_SENSOR)
            for _ in range(10)
        ])
        
        # Grid placement for sensors in area C2 to E5
        sensor_positions = ["C2", "C3", "C4", "C5", "D2", "D3", "D4", "D5", "E2", "E3"]
        for i, pos in enumerate(sensor_positions):
            self.place_at_grid(sensors[i], pos, scale_factor=0.5)
        
        # Introduce sensors at their designated animation step (L011)
        self.play(FadeIn(drone))
        self.play(FadeIn(sensors, shift=UP))

        # Scanning animation: move drone across Row B to represent scanning the sensors below
        scan_path = ["B2", "B3", "B4", "B5"]
        for pos in scan_path[1:]:
            self.play(drone.animate.move_to(self.grid[pos]), run_time=0.4)
        
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Color change for the lecture line to match defect highlight
        self.play(self.lecture[1].animate.set_color(COLOR_DEFECT))
        
        # Highlight a 5% defect rate on one sensor (sensor at index 5 is at D3)
        target_sensor = sensors[5] 
        
        # Defect label (Issue 38 fix: move to F4)
        # Storyboard says label 'p=0.05'
        defect_label = MathTex("p=0.05", color=COLOR_DEFECT, font_size=24)
        self.place_at_grid(defect_label, 'F4', scale_factor=0.8)
        
        self.play(
            target_sensor.animate.set_color(COLOR_DEFECT),
            Write(defect_label),
            Indicate(target_sensor, color=COLOR_DEFECT) # Correct animation class (L004)
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Color change for the lecture line to match success/result color
        self.play(self.lecture[2].animate.set_color(COLOR_SUCCESS))

        # Display result (Issue 37 fix: area A1-A4, scale 0.8)
        # Outline says: "59.9% chance the whole batch is perfect"
        result_text = MathTex("P(X=0) = 59.9\\%", color=COLOR_SUCCESS, font_size=36)
        success_box = SurroundingRectangle(result_text, color=COLOR_SUCCESS, buff=0.2)
        result_group = VGroup(success_box, result_text)
        
        self.place_in_area(result_group, 'A1', 'A4', scale_factor=0.8)
        
        self.play(FadeIn(result_group, shift=DOWN))
        self.wait(2)
