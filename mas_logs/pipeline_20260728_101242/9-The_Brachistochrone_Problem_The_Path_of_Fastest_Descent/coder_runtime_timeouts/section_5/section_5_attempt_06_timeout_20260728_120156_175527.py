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
        lecture_lines = [
            "The solution to this problem is the cycloid.",
            "A cycloid is traced by a point on a rolling circle.",
            "It provides the perfect balance of speed and distance.",
            "It starts steeply to gain rapid initial velocity.",
            "Then, it curves efficiently toward the finish point."
        ]
        self.setup_layout("Revealing the Cycloid", lecture_lines)

        # Colors
        HIGHLIGHT_COLOR = "#FFFF00"
        CYCLOID_COLOR = "#FF0000"
        CIRCLE_COLOR = "#FFFFFF"
        LABEL_COLOR = "#FFFFFF"
        GROUND_COLOR = GRAY_D

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Draw ground line at Row D
        ground_line = Line(
            start=self.grid["D1"] + LEFT * 0.5,
            end=self.grid["D6"] + RIGHT * 0.5,
            color=GROUND_COLOR
        )
        self.play(Create(ground_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)

        # Circle Asset
        radius = 0.4
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg]
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        circle.set_color(CIRCLE_COLOR)
        circle.height = 2 * radius
        
        # Initial position: Sits on ground line (Row D) at Col 1
        start_center = self.grid["D1"] + UP * radius
        circle.move_to(start_center)
        circle.save_state()
        
        # Dot on circumference (bottom point at t=0)
        dot = Dot(color=CYCLOID_COLOR, radius=0.06)
        dot.move_to(self.grid["D1"])
        
        # Trace function for performance
        trace_func = lambda t: start_center + np.array([
            radius * (t - np.sin(t)),
            -radius * np.cos(t),
            0
        ])
        
        cycloid_path = ParametricFunction(
            trace_func,
            t_range=[0, 2*PI],
            color=CYCLOID_COLOR,
            stroke_width=4
        )
        
        # Rolling tracker
        theta = ValueTracker(0)
        
        def update_circle(c):
            t = theta.get_value()
            c.restore()
            c.move_to(start_center + RIGHT * radius * t)
            c.rotate(-t)
            
        def update_dot(d):
            t = theta.get_value()
            d.move_to(trace_func(t))

        circle.add_updater(update_circle)
        dot.add_updater(update_dot)
        
        self.add(circle, dot)
        self.play(
            theta.animate.set_value(2*PI),
            Create(cycloid_path),
            run_time=4,
            rate_func=linear
        )
        circle.remove_updater(update_circle)
        dot.remove_updater(update_dot)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Rotate the cycloid arc 180 degrees downwards
        self.play(
            FadeOut(circle),
            FadeOut(dot),
            Rotate(cycloid_path, angle=PI, axis=RIGHT, about_point=cycloid_path.get_start()),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Label the curve
        label = Text("The Cycloid", font_size=24, color=LABEL_COLOR)
        self.place_at_grid(label, "E3", scale_factor=0.9)
        self.play(Write(label))
        self.wait(2)

        # End of section
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        self.wait(1)
