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
        self.setup_layout("Summary: The Snapshot of Change", [
            "- The derivative is a snapshot of change.",
            "- It bridges the gap between two points and one.",
            "- Now you know the secret behind the speedometer."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Draw a winding hilly road curve in #95A5A6 incorporating [Asset: ...road.svg].
        self.lecture[0].set_color(YELLOW)
        
        # Road function: y = f(x)
        # Shifted slightly to keep the curve mostly in the upper half (Rows A-D)
        def road_func(t):
            x = 3.0 + 2.5 * t
            y = 0.5 + 1.2 * np.sin(3 * t) + 0.5 * np.cos(5 * t)
            return np.array([x, y, 0])

        road_curve = ParametricFunction(road_func, t_range=[-1, 1], color="#95A5A6")
        
        # Road Asset Integration (Issue 20)
        road_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/road.svg")
        self.place_at_grid(road_asset, "A1", scale_factor=0.6)
        road_asset.set_color("#95A5A6")

        self.play(Create(road_curve), FadeIn(road_asset), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw three distinct Tangent Lines (#F1C40F) at different points of the road.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        def get_tangent_line(t_val, length=1.2):
            point = road_func(t_val)
            # Derivative of y = 0.5 + 1.2*sin(3t) + 0.5*cos(5t)
            # dy/dt = 3.6 * cos(3t) - 2.5 * sin(5t)
            # dx/dt = 2.5
            slope_dy_dt = 3.6 * np.cos(3 * t_val) - 2.5 * np.sin(5 * t_val)
            slope_dx_dt = 2.5
            m = slope_dy_dt / slope_dx_dt
            
            # Direction vector
            direction = np.array([1, m, 0])
            direction = direction / np.linalg.norm(direction)
            
            start = point - direction * (length / 2)
            end = point + direction * (length / 2)
            return Line(start, end, color="#F1C40F", stroke_width=4)

        tangents = VGroup(
            get_tangent_line(-0.8),
            get_tangent_line(0.0),
            get_tangent_line(0.7)
        )
        
        for tg in tangents:
            self.play(Create(tg), run_time=0.8)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display the concluding text and [Asset: ...speedometer.svg].
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        concluding_text = Text("Derivative:\nThe Snapshot of Change", color="#FFFFFF", font_size=32)
        
        # Fix for Issue 34 (Repositioning) and Issue 35 (Scaling)
        self.place_in_area(concluding_text, "D2", "F5", scale_factor=0.7)
        
        # Speedometer Asset Integration (Issue 20)
        speed_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        self.place_at_grid(speed_asset, "D1", scale_factor=0.6)
        speed_asset.set_color(WHITE)
        
        bg_rect = BackgroundRectangle(concluding_text, color=BLACK, fill_opacity=0.7, buff=0.3)
        
        self.play(FadeIn(bg_rect), Write(concluding_text), FadeIn(speed_asset))
        self.wait(3)
        
        # Final Highlight
        self.lecture[2].set_color(WHITE)
        self.wait(2)
