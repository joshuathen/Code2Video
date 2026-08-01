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
        CYCLOID_COLOR = "#FF0000"
        CIRCLE_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FF0000"
        GROUND_COLOR = GRAY_D

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Ground line at Row D
        ground = Line(self.grid["D1"] + LEFT*0.5, self.grid["D6"] + RIGHT*0.5, color=GROUND_COLOR)
        self.play(Create(ground))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(CIRCLE_COLOR) # Circle is white

        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg
        radius = 0.5
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        circle.set_color(CIRCLE_COLOR)
        circle.height = 2 * radius
        
        # Start at D1
        x0, y0 = self.grid["D1"][0], self.grid["D1"][1]
        start_center = np.array([x0, y0 + radius, 0])
        circle.move_to(start_center)
        
        dot = Dot(radius=0.06, color=CYCLOID_COLOR)
        dot.move_to(self.grid["D1"]) # Bottom
        
        # Cycloid Path
        cycloid_path = ParametricFunction(
            lambda t: np.array([
                x0 + radius * (t - np.sin(t)),
                y0 + radius * (1 - np.cos(t)),
                0
            ]),
            t_range=[0, 2*PI, 0.1], # Optimized step size
            color=CYCLOID_COLOR,
            stroke_width=5
        )
        
        theta = ValueTracker(0)
        self.last_t = 0
        
        def update_circle(c):
            t = theta.get_value()
            dt = t - self.last_t
            c.shift(RIGHT * radius * dt)
            c.rotate(-dt)
            self.last_t = t
            
        def update_dot(d):
            t = theta.get_value()
            d.move_to(np.array([
                x0 + radius * (t - np.sin(t)),
                y0 + radius * (1 - np.cos(t)),
                0
            ]))

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
        
        # Rotate the cycloid arc 180 degrees to face downwards (Brachistochrone orientation)
        # We also fade out the construction elements
        self.play(
            FadeOut(circle),
            FadeOut(dot),
            FadeOut(ground),
            Rotate(cycloid_path, angle=PI, axis=RIGHT, about_point=cycloid_path.get_start()),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Label the curve
        label = Text("The Cycloid", font_size=24, color=WHITE)
        # Issue 30: Place in area E4-E5
        self.place_in_area(label, 'E4', 'E5', scale_factor=0.8)
        self.play(Write(label))
        self.wait(2)

        # Highlight final line
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        self.wait(1)
