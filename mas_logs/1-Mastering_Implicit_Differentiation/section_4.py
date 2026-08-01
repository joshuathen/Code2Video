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

class Section4Scene(TeachingScene):
    def construct(self):
        # Define lecture lines exactly as per the provided section snapshot
        lecture_lines = [
            "Let's verify our formula on the circle's graph.",
            "At point three four, the slope is negative three fourths.",
            "The formula matches the visual tangent line perfectly."
        ]
        
        # Set up the scene layout with title and lecture sidebars
        self.setup_layout("Graphical Interpretation & Verification", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Use orange for the circle components to match its lecture line
        color_circle = "#FFA500" # Orange
        color_accent = "#FFFFFF" # White
        
        # Color-code the first lecture line
        self.play(self.lecture[0].animate.set_color(color_circle))
        
        # Define a coordinate system (Axes)
        # x_range: [-6, 6] (12 units), y_range: [-6, 6] (12 units)
        # x_length/y_length: 4.5. Unit size = 4.5 / 12 = 0.375
        axes = Axes(
            x_range=[-6, 6, 2],
            y_range=[-6, 6, 2],
            x_length=4.5,
            y_length=4.5,
            axis_config={"color": WHITE, "include_tip": True}
        )
        
        # Create the circle x^2 + y^2 = 25 (Radius 5)
        # Scale radius to match axes units: 5 * 0.375 = 1.875
        circle = Circle(radius=1.875, color=color_circle)
        
        # Group axes and circle to place them as a single unit on the right side
        graph_box = VGroup(axes, circle)
        self.place_in_area(graph_box, 'A1', 'F6')
        
        # Drawing animations
        self.play(Create(axes), Create(circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(self.lecture[1].animate.set_color(color_accent))
        
        # Calculate coordinate for point (3, 4) in the scene
        point_p = axes.c2p(3, 4)
        dot_p = Dot(point_p, color=color_accent)
        
        # Label for the point (3, 4) using the grid system
        label_p = Text("(3, 4)", font_size=18, color=color_accent)
        # Fix for Issue 24: Scale reduced to 0.6 for better visual balance
        self.place_at_grid(label_p, 'B5', scale_factor=0.6)
        
        # Tangent line at (3, 4) with slope m = -3/4
        # Equation: y - 4 = -0.75(x - 3) => y = -0.75x + 6.25
        # Line segment spanning across the circle area
        tangent_start = axes.c2p(-0.5, 6.625)
        tangent_end = axes.c2p(6.5, 1.375)
        tangent_line = Line(start=tangent_start, end=tangent_end, color=color_accent)
        
        # Display the slope formula dy/dx = -3/4
        slope_tex = Text("dy/dx = -3/4", font_size=24, color=color_accent)
        # Fix for Issues 22 & 23: Relocated and scaled to avoid overlap and edge crowding
        self.place_in_area(slope_tex, 'C5', 'C6', scale_factor=0.6)

        # Sequential creation
        self.play(FadeIn(dot_p), FadeIn(label_p))
        self.play(Create(tangent_line))
        self.play(Write(slope_tex))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(self.lecture[2].animate.set_color(color_accent))
        
        # Snail icon: Represented by a triangle sliding down the incline
        snail = Triangle(color=color_accent).scale(0.12)
        
        # Orient the snail to match the negative slope
        # angle of tangent is atan(-0.75)
        slope_angle = np.arctan2(-3, 4)
        snail.rotate(slope_angle - PI/2) # Triangle normally points UP (PI/2)
        
        # Define trajectory along the tangent line
        slide_start = axes.c2p(0.5, 5.875)
        slide_end = axes.c2p(5.5, 2.125)
        
        # Initialize snail position using shift
        initial_pos_vector = slide_start - snail.get_center()
        snail.shift(initial_pos_vector)
        
        # Snail sliding animation
        self.play(FadeIn(snail))
        self.play(
            snail.animate.shift(slide_end - slide_start), 
            run_time=3, 
            rate_func=linear
        )
        self.wait(2)
