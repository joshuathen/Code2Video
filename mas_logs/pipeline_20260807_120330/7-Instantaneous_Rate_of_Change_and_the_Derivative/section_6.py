from manim import *
import numpy as np

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
        self.setup_layout("Real-World Application: The Speedometer", [
            "- A car's speedometer shows speed at every heartbeat.",
            "- It continuously calculates the derivative of position.",
            "- Calculus connects abstract math to movement."
        ])

        # Assets
        speed_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/speed.svg"
        car_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Speedometer SVG
        speedometer = SVGMobject(speed_path).set_color(WHITE)
        # Fix layout: Place the speedometer in area B3-F6 with scale 0.8 (Issues #40, #41)
        self.place_in_area(speedometer, "B3", "F6", scale_factor=0.8)
        
        # Create a Needle
        pivot = speedometer.get_center()
        # Initial needle position (pointing to 0, approx 135 deg in standard polar)
        needle = Line(pivot, pivot + rotate_vector(UP * 0.7, 135 * DEGREES), color=RED, stroke_width=4)
        
        self.play(FadeIn(speedometer), Create(needle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Car image (Issue #27, #47)
        car = ImageMobject(car_path)
        self.place_at_grid(car, "F3", scale_factor=0.5)
        
        # Synchronization Tracker
        progress = ValueTracker(0)
        
        # Updaters for synchronized motion
        # Needle sweep from 135 deg to -135 deg (270 deg total)
        def update_needle(m):
            angle = (135 - progress.get_value() * 270) * DEGREES
            m.put_start_and_end_on(pivot, pivot + rotate_vector(UP * 0.7, angle))
            
        needle.add_updater(update_needle)
        
        # Car moves horizontally across the bottom
        start_car_x = self.grid["F3"][0]
        end_car_x = self.grid["F6"][0]
        car.add_updater(lambda m: m.set_x(start_car_x + progress.get_value() * (end_car_x - start_car_x)))
        
        self.play(FadeIn(car))
        self.play(progress.animate.set_value(1), run_time=4, rate_func=slow_into)
        self.wait(1)
        
        needle.clear_updaters()
        car.clear_updaters()

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Derivative label
        derivative_text = Text("Instantaneous Speed = Derivative", font_size=20, color="#00FFFF")
        # Position label in Row A to avoid occlusion (Belief B005 consideration vs Issue 41)
        self.place_in_area(derivative_text, "A3", "A6", scale_factor=0.8)
        
        self.play(Write(derivative_text))
        self.wait(2)
        
        # Reset color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
