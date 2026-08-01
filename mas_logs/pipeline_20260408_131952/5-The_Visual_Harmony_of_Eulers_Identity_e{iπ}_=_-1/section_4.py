from manim import *
import numpy as np

# Fix for KeyError caused by curly braces in the file path
config.input_file = "section_4.py"

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
        # Setup layout
        lecture_lines = [
            'This circular motion is described by Euler’s general formula.',
            'e^ix defines a point on the unit circle.',
            'The variable x represents the angle in radians.',
            'The horizontal component is given by the cosine function.',
            'The vertical component is mapped using the sine function.'
        ]
        self.setup_layout("The Unit Circle and Euler's Formula", lecture_lines)

        # Mathematical parameters
        angle_val = 1.0  # Approx 57 degrees
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        formula = Text("e^ix = cos(x) + i sin(x)", color=WHITE, font_size=24)
        # Fix for Issue 41: Reposition formula to avoid crowding
        self.place_in_area(formula, 'A3', 'A6', scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Fix for Issue 42: Anchor complex plane to grid area
        complex_plane = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": GREY_C}
        )
        self.place_in_area(complex_plane, 'C3', 'F6', scale_factor=0.9)
        axes = complex_plane # for calculation reference
        
        real_label = Text("Re", font_size=16, color=GREY_C)
        self.place_at_grid(real_label, 'D6', scale_factor=1.0)
        imag_label = Text("Im", font_size=16, color=GREY_C)
        self.place_at_grid(imag_label, 'C4', scale_factor=1.0)
        
        # Unit Circle
        unit_radius = axes.coords_to_point(1, 0)[0] - axes.coords_to_point(0, 0)[0]
        circle = Circle(radius=unit_radius, color=WHITE, stroke_width=2)
        circle.move_to(axes.get_center())
        
        # Point on the circle at angle x
        pt_coords = axes.coords_to_point(np.cos(angle_val), np.sin(angle_val))
        
        # Highlight vector from origin to the point as e^ix (Cyan)
        vector = Arrow(axes.get_center(), pt_coords, buff=0, color="#00FFFF", stroke_width=4)
        vector_label = Text("e^ix", color="#00FFFF", font_size=18)
        self.place_at_grid(vector_label, 'C5', scale_factor=1.0)
        
        self.play(Create(complex_plane), Create(circle), Write(real_label), Write(imag_label))
        self.play(GrowArrow(vector), Write(vector_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Angle arc labeled x
        angle_arc = Arc(
            radius=0.4, 
            start_angle=0, 
            angle=angle_val, 
            arc_center=axes.get_center(), 
            color=YELLOW
        )
        angle_label = Text("x", color=YELLOW, font_size=20)
        self.place_at_grid(angle_label, 'D4', scale_factor=1.0)
        
        self.play(Create(angle_arc), Write(angle_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Vertical projection to the Real axis labeled cos(x)
        real_pt = [pt_coords[0], axes.get_center()[1], 0]
        v_proj = DashedLine(pt_coords, real_pt, color=WHITE)
        
        # Horizontal distance highlight (cosine)
        cos_line = Line(axes.get_center(), real_pt, color=GREEN, stroke_width=6)
        cos_label = Text("cos(x)", color=GREEN, font_size=22)
        self.place_at_grid(cos_label, 'E5', scale_factor=1.0)
        
        self.play(Create(v_proj))
        self.play(Create(cos_line), Write(cos_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Horizontal projection to the Imaginary axis labeled i sin(x)
        imag_pt = [axes.get_center()[0], pt_coords[1], 0]
        h_proj = DashedLine(pt_coords, imag_pt, color=WHITE)
        
        # Vertical distance highlight (sine)
        sin_line = Line(axes.get_center(), imag_pt, color=PINK, stroke_width=6)
        sin_label = Text("i sin(x)", color=PINK, font_size=22)
        # Fix for Issue 43: Position sin_label to avoid obstruction
        self.place_at_grid(sin_label, 'D5', scale_factor=0.7)
        
        self.play(Create(h_proj))
        self.play(Create(sin_line), Write(sin_label))
        self.wait(2)
