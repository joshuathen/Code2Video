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
        
        # Horizontal ground line - shifted to be consistent with C row
        ground_line = Line(
            start=self.grid["C1"] + LEFT * 0.5,
            end=self.grid["C6"] + RIGHT * 0.5,
            color=GRAY_D
        )
        self.play(Create(ground_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(TEXT_HIGHLIGHT)

        # Setup rolling circle
        radius = 0.4
        # Using the specified SVG asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg]
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        circle.set_color(CIRCLE_COLOR)
        circle.height = 2 * radius
        
        # Initial position: sits on ground at C1
        start_center = self.grid["C1"] + UP * radius
        circle.move_to(start_center)
        circle.save_state()
        
        # Dot on circumference (bottom point relative to center at t=0)
        dot = Dot(color=CYCLOID_COLOR, radius=0.06)
        dot.move_to(self.grid["C1"])
        
        # Tracker for rolling distance (angle theta)
        theta = ValueTracker(0)
        
        # Optimized updaters: absolute positioning to avoid drift or double-transformations
        def update_circle(c):
            t = theta.get_value()
            c.restore()
            c.move_to(start_center + RIGHT * radius * t)
            c.rotate(-t)
            
        def update_dot(d):
            t = theta.get_value()
            curr_center = start_center + RIGHT * radius * t
            # Cycloid parametric eq relative to center: x = -r*sin(t), y = -r*cos(t)
            d.move_to(curr_center + np.array([-radius * np.sin(t), -radius * np.cos(t), 0]))

        circle.add_updater(update_circle)
        dot.add_updater(update_dot)
        
        # Pre-calculate path points for efficiency (avoiding TracedPath which is heavier)
        cycloid_points = [
            start_center + np.array([radius * (t - np.sin(t)), -radius * np.cos(t), 0])
            for t in np.linspace(0, 2 * PI, 60)
        ]
        cycloid_path = VMobject(color=CYCLOID_COLOR, stroke_width=4)
        cycloid_path.set_points_as_corners(cycloid_points)
        
        self.add(circle, dot)
        # Create the path while circle rolls. Since both use linear rate_func, they stay in sync.
        self.play(
            theta.animate.set_value(2 * PI),
            Create(cycloid_path),
            run_time=4,
            rate_func=linear
        )
        
        # Static state transition: remove updaters to save render time
        circle.clear_updaters()
        dot.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(TEXT_HIGHLIGHT)
        
        # Fade the construction elements to focus on the curve
        self.play(
            circle.animate.set_stroke(opacity=0.2).set_fill(opacity=0.1),
            FadeOut(dot),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(TEXT_HIGHLIGHT)
        
        # Flip the cycloid 180 degrees to face downwards (The Brachistochrone curve)
        # Pivot is the starting point on the ground
        pivot = self.grid["C1"]
        self.play(
            Rotate(cycloid_path, angle=PI, axis=RIGHT, about_point=pivot),
            FadeOut(circle, ground_line),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(TEXT_HIGHLIGHT)
        
        # Label the curve - Centered in area E4-E5 for better alignment (Resolves Issue 30)
        label = Text("The Cycloid", font_size=24, color=WHITE)
        self.place_in_area(label, 'E4', 'E5', scale_factor=0.8)
        
        self.play(Write(label))
        # Brief highlight of the curve
        self.play(Indicate(cycloid_path, color=CYCLOID_COLOR), run_time=1.5)
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
        self.wait(1)

# Final issue updates (integrated into tool calls by the MAS system)
# update_issue(21, under_review=True, resolution_note="Integrated circle.svg asset and optimized animation with synchronized Create() and ValueTracker.")
# update_issue(30, under_review=True, resolution_note="Used place_in_area('E4', 'E5') for label centering.")
