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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Vectors underpin physics and gaming.", "They define motion and positions.", "These basics build complex systems."]
        self.setup_layout("Summary and Real-World Application", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        summary = Text("Vectors: The Language of Motion", font_size=32, color="#FFD700")
        self.place_at_grid(summary, "B2", scale_factor=0.8)
        self.play(Write(summary))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(self.lecture[1].animate.set_color("#00CED1"))
        
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        velocity_arrow = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#00CED1")
        car_group = Group(car, velocity_arrow)
        self.place_in_area(car_group, "C4", "D5", scale_factor=0.9)
        self.play(FadeIn(car_group))
        self.play(car_group.animate.shift(RIGHT*0.5), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(self.lecture[2].animate.set_color("#FF6347"))
        complex_systems = Text("Physics, Games, AI", font_size=36, color="#FF6347")
        self.place_at_grid(complex_systems, "E5", scale_factor=0.8)
        self.play(FadeIn(complex_systems))
        self.wait(2)
        
        self.play(FadeOut(self.lecture), FadeOut(self.title), FadeOut(summary), FadeOut(car_group), FadeOut(complex_systems))
