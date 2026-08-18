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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Hook: Speed vs. Distance", [
            "A robot runs: speed is the instantaneous derivative.", 
            "Area under the speed curve is total distance.", 
            "These processes are inversely connected, remarkably."
        ])
        
        # Assets
        car_img_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png"
        car_a = ImageMobject(car_img_path).set_color(RED)
        car_b = ImageMobject(car_img_path).set_color(BLUE)
        
        label_a = Text("Constant Speed", color=RED, font_size=18).scale(0.7)
        label_b = Text("Acceleration", color=BLUE, font_size=18).scale(0.7)
        formula = Text("Distance = Area under Curve", color=YELLOW, font_size=24).scale(0.8)
        
        # Position initial elements per feedback
        self.place_at_grid(car_a, "B2", scale_factor=0.6)
        label_a.next_to(car_a, RIGHT, buff=0.2)
        
        self.place_at_grid(car_b, "D4", scale_factor=0.6)
        label_b.next_to(car_b, RIGHT, buff=0.2)
        
        self.add(car_a, label_a, car_b, label_b)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Move cars
        self.play(
            car_a.animate.shift(RIGHT * 2),
            car_b.animate.shift(RIGHT * 2),
            run_time=2
        )

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_in_area(formula, "E4", "F6", scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
