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
        # Setup the layout with section title and lecture lines
        self.setup_layout("The Mystery of the Slice", [
            "Conic sections are formed by slicing a double cone.",
            "We get ellipses, parabolas, and hyperbolas from different angles.",
            "How do these shapes connect to their algebraic definitions?"
        ])
        
        # Define Colors
        CONE_COLOR = "#C0C0C0"
        PLANE_COLOR = "#00FFFF"
        CURVE_COLOR = "#FFD700"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg]
        # Using the SVG asset for the cone. Since it's a "double cone", 
        # I'll create two copies, one flipped, to represent the geometry.
        cone_upper = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")
        cone_upper.set_color(CONE_COLOR).set_opacity(0.4)
        
        cone_lower = cone_upper.copy().flip(RIGHT) # Flip vertically (around X axis equivalent in 2D projection)
        # Manim's flip(RIGHT) flips around the x-axis? Actually flip(RIGHT) is reflection across Y-axis. 
        # For a cone to be flipped vertically, we use flip(RIGHT) if it's already vertical? 
        # Let's use rotate(PI) or scale([1, -1, 1]).
        cone_lower.scale([1, -1, 1])
        
        double_cone = VGroup(cone_upper, cone_lower).arrange(DOWN, buff=0)
        self.place_in_area(double_cone, 'B2', 'E4', scale_factor=1.5)
        
        self.play(DrawBorderThenFill(double_cone), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Define a slicing plane (represented as a slanted surface)
        plane = Polygon(
            [-1.5, -0.2, 0], [1.5, 0.8, 0], [1.8, 1.2, 0], [-1.2, 0.2, 0],
            color=PLANE_COLOR, fill_opacity=0.5, stroke_width=2
        )
        # Place it such that it intersects the upper cone
        self.place_in_area(plane, 'C2', 'D4', scale_factor=0.9)
        plane.shift(UP * 0.7)

        # The intersection curve is an ellipse
        intersection_curve = Ellipse(width=1.0, height=0.3, color=CURVE_COLOR, stroke_width=4)
        intersection_curve.rotate(15 * DEGREES)
        # Position it to align with the plane's intersection on the cone
        self.place_at_grid(intersection_curve, 'C3', scale_factor=1.0)
        intersection_curve.shift(UP * 0.6 + RIGHT * 0.1)

        self.play(FadeIn(plane, shift=RIGHT), run_time=1)
        self.play(Create(intersection_curve), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Add labels for 'Cone' and 'Conic Section' with specific fixes from issues
        cone_label = Text("Cone", font_size=20, color=WHITE)
        # Issue 30: Fix cone_label position and scale
        self.place_at_grid(cone_label, 'B5', scale_factor=0.8)
        
        section_label = Text("Conic Section", font_size=20, color=CURVE_COLOR)
        # Issue 29: Fix section_label position and scale
        self.place_in_area(section_label, 'D5', 'D6', scale_factor=0.8)
        
        # Arrows to point to the objects
        arrow_cone = Arrow(
            cone_label.get_left(), 
            self.grid['C3'] + UP*0.5, 
            color=WHITE, buff=0.1, stroke_width=3
        )
        arrow_section = Arrow(
            section_label.get_left(), 
            intersection_curve.get_center(), 
            color=CURVE_COLOR, buff=0.2, stroke_width=3
        )

        self.play(
            Write(cone_label),
            Write(section_label),
            GrowArrow(arrow_cone),
            GrowArrow(arrow_section)
        )
        self.wait(2)
        
        # Reset color for final view
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
