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
        lecture_lines = [
            "Slope fields show the landscape of change.",
            "Every point dictates a tangent vector's direction.",
            "Follow the stream to visualize the solution."
        ]
        self.setup_layout("Visualizing the Solution: Direction Fields", lecture_lines)
        
        # Create header
        header_text = Text("Direction Field", font_size=24, color=YELLOW)
        self.place_at_grid(header_text, 'A3', scale_factor=0.9)
        self.add(header_text)
        
        # Define field
        field = VGroup()
        for i in range(-2, 3):
            for j in range(-2, 3):
                arrow = Arrow(start=ORIGIN, end=RIGHT*0.3, buff=0, color="#808080", stroke_width=2)
                angle = np.arctan2(j, i+1)
                arrow.rotate(angle, about_point=ORIGIN)
                arrow.shift(self.grid['C3'] + np.array([i*0.6, j*0.6, 0]))
                field.add(arrow)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(Create(field))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FFD700"))
        # Color vectors based on angle
        self.play(*[vec.animate.set_color(interpolate_color(BLUE, RED, (np.arctan2(vec.get_center()[1], vec.get_center()[0]) + PI)/(2*PI))) for vec in field])

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFD700"))
        
        # Asset: tracer point
        tracer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stream.svg")
        tracer.set_color(WHITE)
        tracer.scale(0.05)
        tracer.move_to(self.grid['C1'])
        self.add(tracer)
        
        path = TracedPath(tracer.get_center, stroke_width=3, stroke_color="#FFFFFF")
        self.add(path)
        
        # Group to allow place_in_area as requested by critic
        animation_group = VGroup(tracer, path)
        self.place_in_area(animation_group, 'C3', 'E5', scale_factor=0.6)
        
        # Animate tracer
        self.play(tracer.animate.shift(RIGHT*2 + UP*1), run_time=3, rate_func=linear)
