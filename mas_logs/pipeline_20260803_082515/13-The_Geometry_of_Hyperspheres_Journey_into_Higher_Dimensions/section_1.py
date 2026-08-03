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
        # Initializing layout with section title and lecture lines from storyboard
        self.setup_layout("The Dimension Ladder: From Lines to Spheres", [
            "- A point is zero-dimensional with no length or width.",
            "- Moving a point creates a one-dimensional line segment.",
            "- Rotating a line creates a two-dimensional circle.",
            "- Rotating a circle creates a three-dimensional sphere.",
            "- All points are equidistant from a central center."
        ])

        # Color definitions for visual elements and corresponding lecture lines
        COLOR_0D = RED_A
        COLOR_1D = GREEN_A
        COLOR_2D = BLUE_A
        COLOR_3D = PURPLE_A
        COLOR_LABEL = "#00FFFF"
        COLOR_EQUIDISTANT = ORANGE

        # Main visual anchor point in the grid
        visual_center_key = 'C4'
        visual_center = self.grid[visual_center_key]
        radius = 1.3

        # === Animation for Lecture Line 1 ===
        # A point is zero-dimensional with no length or width.
        self.lecture[0].set_color(COLOR_0D)
        center_dot = Dot(color=COLOR_0D)
        self.place_at_grid(center_dot, visual_center_key)
        self.play(Create(center_dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Moving a point creates a one-dimensional line segment.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_1D)
        
        # Create segment and flanking dots relative to center
        point_l = Dot(visual_center, color=COLOR_1D)
        point_r = Dot(visual_center, color=COLOR_1D)
        segment = Line(visual_center + LEFT * radius, visual_center + RIGHT * radius, color=COLOR_1D)
        
        self.play(
            point_l.animate.shift(LEFT * radius),
            point_r.animate.shift(RIGHT * radius),
            Create(segment),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Rotating a line creates a two-dimensional circle.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_2D)
        
        # Define circle and a helper segment to show rotation
        circle = Circle(radius=radius, color=COLOR_2D).move_to(visual_center)
        rotating_line = segment.copy().set_color(COLOR_2D)
        
        self.play(
            Rotate(rotating_line, angle=PI, about_point=visual_center),
            Create(circle),
            run_time=2
        )
        self.remove(rotating_line)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Rotating a circle creates a three-dimensional sphere.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_3D)
        
        # Integrate Asset: sphere.svg as per Issue 19
        sphere_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere_svg.set_color(COLOR_3D)
        sphere_svg.scale_to_fit_width(2 * radius)
        self.place_at_grid(sphere_svg, visual_center_key, scale_factor=1.0)
        
        # Use ellipses to represent 3D volume in 2D projection
        ellipse_h = Ellipse(width=2*radius, height=0.6*radius, color=COLOR_3D).move_to(visual_center)
        ellipse_v = Ellipse(width=0.6*radius, height=2*radius, color=COLOR_3D).move_to(visual_center)
        
        # Labels for different dimensions
        label_1d = Text("1D: Segment", font_size=18, color=COLOR_LABEL)
        label_2d = Text("2D: Circle", font_size=18, color=COLOR_LABEL)
        label_3d = Text("3D: Sphere", font_size=18, color=COLOR_LABEL)
        
        # Grid positioning fixes for Issue 22 and 23
        self.place_at_grid(label_1d, 'A2', scale_factor=0.8)
        self.place_at_grid(label_2d, 'A4', scale_factor=0.8) # Fix for horizontal overlap (Issue 22)
        self.place_at_grid(label_3d, 'A6', scale_factor=0.8) # Fix for horizontal overlap (Issue 23)

        self.play(
            FadeIn(sphere_svg),
            Create(ellipse_h),
            Create(ellipse_v),
            circle.animate.set_color(COLOR_3D),
            run_time=2
        )
        self.play(
            Write(label_1d),
            Write(label_2d),
            Write(label_3d)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # All points are equidistant from a central center.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_EQUIDISTANT)
        
        # Highlight center and show radius lines to surface points
        # Keep sphere_svg visible from previous step (Asset requirement for Issue 19)
        center_dot.set_color(COLOR_EQUIDISTANT)
        num_points = 10
        surface_dots = VGroup()
        radius_lines = VGroup()
        
        for i in range(num_points):
            angle = i * (2 * PI / num_points)
            # Mixed projection points to show surface distribution
            if i % 3 == 0:
                p_pos = visual_center + np.array([radius * np.cos(angle), 0.3 * radius * np.sin(angle), 0])
            elif i % 3 == 1:
                p_pos = visual_center + np.array([0.3 * radius * np.cos(angle), radius * np.sin(angle), 0])
            else:
                p_pos = visual_center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
                
            dot = Dot(p_pos, radius=0.06, color=COLOR_EQUIDISTANT)
            line = DashedLine(visual_center, p_pos, color=COLOR_EQUIDISTANT, stroke_width=2)
            surface_dots.add(dot)
            radius_lines.add(line)
            
        self.play(
            LaggedStart(*[FadeIn(d) for d in surface_dots], lag_ratio=0.1),
            LaggedStart(*[Create(l) for l in radius_lines], lag_ratio=0.1),
            run_time=2
        )
        self.wait(3)
