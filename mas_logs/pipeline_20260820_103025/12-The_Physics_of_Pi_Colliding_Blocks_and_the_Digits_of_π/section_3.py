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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Plot velocities in phase space.", "Collisions trace a circular arc.", "Angle relates to mass ratio.", "Each step is one collision.", "Steps reveal digits of π."]
        self.setup_layout("Mapping Collisions to Geometry", lecture_lines)
        
        # Define objects
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False, "color": "#95A5A6"})
        circle = Arc(radius=1.5, start_angle=0, angle=PI/2, color=WHITE)
        point = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg")
        point.set_color("#2ECC71")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.5)
        self.play(Create(axes))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.place_in_area(circle, 'B2', 'E5', scale_factor=0.5)
        self.play(Create(circle))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        label = MathTex(r"tan(\theta) = \sqrt{m_1/m_2}", font_size=24)
        self.place_at_grid(label, 'D4', scale_factor=0.7)
        self.play(Write(label))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        self.place_at_grid(point, 'C4', scale_factor=0.6)
        self.play(FadeIn(point))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(ORANGE)
        path = VMobject()
        path.set_points_smoothly([self.grid['C4'], self.grid['C5'], self.grid['B5']])
        path.set_stroke("#E67E22", 2)
        self.play(MoveAlongPath(point, path), run_time=2)
        self.wait(1)
