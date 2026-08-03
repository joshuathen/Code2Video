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
        title = "The Solution: The Cycloid"
        lines = [
            "The fastest path is called the cycloid.",
            "It is the path of a rolling wheel.",
            "This curve balances acceleration and distance perfectly.",
            "Johann Bernoulli challenged the world with this problem.",
            "The solution revealed a new branch of calculus."
        ]
        self.setup_layout(title, lines)

        # Colors
        ORANGE = "#FFA500"
        RED = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.lecture[0].set_color(ORANGE)
        
        # Define Points A and B for visual reference
        # Start A at A3 as per Issue 38
        point_a = Dot(self.grid["A3"], color=WHITE)
        label_a = Text("A", font_size=18).next_to(point_a, UP, buff=0.1)
        point_b = Dot(self.grid["E5"], color=WHITE)
        label_b = Text("B", font_size=18).next_to(point_b, DOWN, buff=0.1)
        
        self.add(point_a, label_a, point_b, label_b)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2 (dim Line 1)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WHITE)
        
        # Wheel setup using Asset (Issue 27)
        wheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg")
        # Position wheel at A3, scale 0.5 (Issue 38)
        self.place_at_grid(wheel, "A3", scale_factor=0.5)
        
        # Adjust wheel position so its rim point is at A3 (ceiling rolling)
        radius = wheel.height / 2
        start_center = self.grid["A3"] + DOWN * radius
        wheel.move_to(start_center)
        
        # A point on the rim (initially at the contact point A3)
        rim_point = Dot(self.grid["A3"], color=RED)
        
        # Group for coordination
        wheel_group = VGroup(wheel, rim_point)
        
        self.play(FadeIn(wheel), FadeIn(rim_point))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3 (dim Line 2)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ORANGE)
        
        # Rolling animation using ValueTracker
        roll_tracker = ValueTracker(0)
        self.last_theta = 0
        
        # The path tracing
        path = TracedPath(rim_point.get_center, stroke_color=ORANGE, stroke_width=4)
        self.add(path)
        
        def update_wheel(m):
            theta = roll_tracker.get_value()
            d_theta = theta - self.last_theta
            
            # Center moves linearly
            new_center = start_center + RIGHT * radius * theta
            m[0].move_to(new_center)
            # Wheel rotates
            m[0].rotate(-d_theta)
            
            # Point on rim (starting at top relative to center: theta=0 -> [0, R, 0])
            offset = np.array([
                radius * np.sin(theta),
                radius * np.cos(theta),
                0
            ])
            m[1].move_to(new_center + offset)
            self.last_theta = theta

        wheel_group.add_updater(update_wheel)
        
        # Roll for one full rotation
        self.play(roll_tracker.animate.set_value(TAU), run_time=4, rate_func=linear)
        wheel_group.remove_updater(update_wheel)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight Line 4 (dim Line 3)
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(WHITE)
        
        # Static cycloid curve from A to B
        p1 = self.grid["A3"]
        p2 = self.grid["E5"]
        
        def cycloid_parametric(t):
            # t from 0 to PI
            # Interpolates between A3 and E5 following cycloid shape
            nx = (t - np.sin(t)) / PI
            ny = (1 - np.cos(t)) / 2
            return np.array([
                p1[0] + (p2[0] - p1[0]) * nx,
                p1[1] + (p2[1] - p1[1]) * ny,
                0
            ])

        brach_curve = ParametricFunction(cycloid_parametric, t_range=[0, PI], color=ORANGE)
        
        # Label 'Cycloid' at D4, scale 0.8 (Issue 37)
        label_cycloid = Text("Cycloid", font_size=20, color=ORANGE)
        self.place_at_grid(label_cycloid, "D4", scale_factor=0.8)
        
        self.play(
            FadeOut(wheel_group),
            FadeOut(path),
            Create(brach_curve),
            Write(label_cycloid)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight Line 5 (dim Line 4)
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(ORANGE)
        
        # Flash the text 'Brachistochrone'
        brach_text = Text("Brachistochrone", font_size=36, color=ORANGE)
        # Position at B4-B6, scale 0.7 (Issue 36)
        self.place_in_area(brach_text, "B4", "B6", scale_factor=0.7)
        
        self.play(Flash(brach_text, color=ORANGE, flash_radius=1.5))
        self.play(Write(brach_text))
        self.wait(2)
        
        # Cleanup
        self.play(FadeOut(brach_text), FadeOut(label_cycloid))
        self.wait(1)
