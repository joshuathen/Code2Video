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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Solution: The Cycloid", ["The optimal curve is a cycloid.", "It traces a rolling circle's rim point.", "Cycloids balance path length and speed gains."])
        
        # Setup Cycloid animation
        r = 0.5
        line = Line(start=self.grid['F1'], end=self.grid['F6'], color=WHITE)
        circle = Circle(radius=r, color=BLUE)
        
        # Point on rim
        rim_point = Dot(color=RED)
        
        # Rolling motion setup
        def get_circle_pos(t):
            x = -1.5 + r * t
            y = self.grid['F1'][1] + r
            return np.array([x, y, 0])
            
        def get_rim_pos(t):
            angle = -t
            cp = get_circle_pos(t)
            return cp + np.array([r * np.sin(angle), -r * np.cos(angle), 0])

        self.current_time = 0.0
        circle.add_updater(lambda m: m.move_to(get_circle_pos(self.current_time)))
        rim_point.add_updater(lambda m: m.move_to(get_rim_pos(self.current_time)))
        
        # Tracer path
        path = TracedPath(rim_point.get_center, stroke_color=GREEN, stroke_width=4)
        
        def update_time(m, dt):
            self.current_time += dt * 2
            
        self.add(line, circle, rim_point, path)
        self.add_updater(update_time)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=1)
        self.wait(2)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"), run_time=1)
        self.wait(5)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"), run_time=1)
        
        # Cleanup
        self.remove_updater(update_time)
        self.wait(2)