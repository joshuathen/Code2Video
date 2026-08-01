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
        TEXT_HIGHLIGHT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(TEXT_HIGHLIGHT)
        
        # Horizontal line for rolling
        ground_line = Line(
            start=self.grid["C1"] + LEFT * 0.5,
            end=self.grid["C6"] + RIGHT * 0.5,
            color=GRAY
        )
        self.play(Create(ground_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(TEXT_HIGHLIGHT)

        # Rolling Circle setup using provided Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg]
        radius = 0.5
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        circle.set_color(CIRCLE_COLOR)
        circle.height = 2 * radius
        
        # Starting position at C1, sitting on the line
        start_center = self.grid["C1"] + UP * radius
        circle.move_to(start_center)
        
        # Red dot on circumference (starts at bottom point of the circle)
        dot = Dot(color=CYCLOID_COLOR, radius=0.08)
        dot.move_to(start_center + DOWN * radius)
        
        # Tracker for angle theta (rotation and translation)
        theta = ValueTracker(0)
        
        # Reference circle for rotation calculation to avoid cumulative errors
        circle_template = circle.copy()

        def update_circle(c):
            t = theta.get_value()
            center = start_center + RIGHT * radius * t
            # Move and rotate the template
            c.become(circle_template.copy().move_to(center).rotate(-t))

        def update_dot(d):
            t = theta.get_value()
            center = start_center + RIGHT * radius * t
            # Cycloid parametric equations relative to start: 
            # x = r(t - sin(t)), y = r(1 - cos(t))
            # Current dot position relative to start ground point (C1):
            # pos = C1 + [r*t - r*sin(t), r - r*cos(t)]
            # Which is: center + [-r*sin(t), -r*cos(t)]
            d.move_to(center + np.array([-radius * np.sin(t), -radius * np.cos(t), 0]))

        circle.add_updater(update_circle)
        dot.add_updater(update_dot)
        
        # Path tracing
        path = TracedPath(dot.get_center, stroke_color=CYCLOID_COLOR, stroke_width=4)
        
        self.add(circle, dot, path)
        # Roll circle for exactly one full revolution (2*pi)
        self.play(theta.animate.set_value(2 * PI), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(TEXT_HIGHLIGHT)
        
        # Finalize the path as a static VMobject for subsequent transformation
        start_floor = self.grid["C1"]
        cycloid_points = [
            start_floor + np.array([radius * (t - np.sin(t)), radius * (1 - np.cos(t)), 0])
            for t in np.linspace(0, 2 * PI, 100)
        ]
        static_cycloid = VMobject(color=CYCLOID_COLOR).set_points_as_corners(cycloid_points)
        static_cycloid.set_stroke(width=4)
        
        # Transition from dynamic updaters to static path
        circle.remove_updater(update_circle)
        dot.remove_updater(update_dot)
        self.remove(circle, dot, path)
        self.add(static_cycloid)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(TEXT_HIGHLIGHT)
        
        # Rotate/Flip the cycloid 180 degrees to face downwards (The Brachistochrone curve)
        # Pivot around the start point (grid C1) using RIGHT axis (X-axis flip)
        pivot = self.grid["C1"]
        self.play(
            Rotate(static_cycloid, angle=PI, axis=RIGHT, about_point=pivot),
            ground_line.animate.set_stroke(opacity=0.3),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(TEXT_HIGHLIGHT)
        
        # Label the curve - Centered between E4 and E5 as per Issue 30
        label = Text("The Cycloid", font_size=24, color=WHITE)
        self.place_in_area(label, 'E4', 'E5', scale_factor=0.8)
        
        self.play(Write(label))
        self.play(Indicate(static_cycloid, color=CYCLOID_COLOR))
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
        self.wait(2)
