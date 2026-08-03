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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "The Math of the Pattern: Complex Gratings",
            [
                "- Fringes act as a complex grating.",
                "- Light bends following the grating equation.",
                "- Variable spacing encodes the object's shape."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Fringes act as a complex grating.
        # Show a cross-section of plate with vertical lines (#FFFFFF).
        self.lecture[0].set_color(YELLOW)
        
        plate = Rectangle(width=4.0, height=3.0, color=WHITE, stroke_width=2)
        # Fix for Issue 30: Changed from 'C2'-'E5' to 'C3'-'E6' for better margin.
        self.place_in_area(plate, 'C3', 'E6')
        
        fringes = VGroup()
        num_fringes = 11
        for i in range(num_fringes):
            # Positioning fringes manually within the plate relative to its center
            offset = (i - (num_fringes - 1) / 2) * 0.35
            line = Line(
                plate.get_top() + RIGHT * offset,
                plate.get_bottom() + RIGHT * offset,
                color=WHITE,
                stroke_width=1.5
            )
            fringes.add(line)
            
        self.play(Create(plate), Create(fringes), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Light bends following the grating equation.
        # Display 'd sin(theta) = n*lambda' (#FFFF00)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        equation = MathTex(r"d \sin(\theta) = n\lambda", color=YELLOW)
        # Fix for Issue 29: Changed from 'A2'-'B5' to 'A3'-'B5' and scale to 1.0.
        self.place_in_area(equation, 'A3', 'B5', scale_factor=1.0)
        
        # Spacing label 'd'
        d_brace = BraceBetweenPoints(fringes[4].get_top(), fringes[5].get_top(), direction=UP, color=YELLOW, buff=0.1)
        d_label = MathTex("d", color=YELLOW, font_size=22).next_to(d_brace, UP, buff=0.05)
        
        # Theta label with a ray
        ray_origin = plate.get_center()
        normal = DashedLine(ray_origin, ray_origin + RIGHT * 1.5, color=GREY, stroke_width=2)
        diffracted_ray = Line(ray_origin, ray_origin + rotate_vector(RIGHT * 1.5, 35 * DEGREES), color=RED, stroke_width=3)
        arc = Arc(radius=0.6, start_angle=0, angle=35 * DEGREES, arc_center=ray_origin, color=YELLOW)
        theta_label = MathTex(r"\theta", color=YELLOW, font_size=22).next_to(arc, RIGHT, buff=0.1)
        
        self.play(Write(equation))
        self.play(FadeIn(d_brace), Write(d_label))
        self.play(Create(normal), Create(diffracted_ray), Create(arc), Write(theta_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Variable spacing encodes the object's shape.
        # Show light rays (#FF0000) bending through variable spacing 'd'.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Variable spacing fringes
        new_fringes = VGroup()
        for i in range(num_fringes):
            # Variable spacing logic: compressing on one side, expanding on the other
            # Use a non-linear offset
            norm_i = (i / (num_fringes - 1)) * 2 - 1 # range -1 to 1
            offset = 1.8 * (norm_i + 0.2 * norm_i**2)
            line = Line(
                plate.get_top() + RIGHT * offset,
                plate.get_bottom() + RIGHT * offset,
                color=WHITE,
                stroke_width=1.5
            )
            new_fringes.add(line)
            
        # Rays bending through variable spacing
        rays_in = VGroup()
        rays_out = VGroup()
        
        y_positions = np.linspace(-1.0, 1.0, 5)
        # Angles corresponding to the local spacing (approximate visual effect)
        # Narrow spacing (right side in the offset logic) => higher angle
        angles = [15, 25, 35, 45, 55] 
        
        for i, y in enumerate(y_positions):
            # Ray enters from the left
            start_point = plate.get_left() + LEFT * 1.0 + UP * y
            entry_point = plate.get_left() + UP * y
            rays_in.add(Line(start_point, entry_point, color=RED, stroke_width=2))
            
            # Ray exits bent at an angle
            angle_rad = angles[i] * DEGREES
            exit_point = entry_point + rotate_vector(RIGHT * 1.5, angle_rad)
            rays_out.add(Line(entry_point, exit_point, color=RED, stroke_width=2))
            
        self.play(
            FadeOut(d_brace), FadeOut(d_label), 
            FadeOut(normal), FadeOut(diffracted_ray), FadeOut(arc), FadeOut(theta_label),
            Transform(fringes, new_fringes)
        )
        self.play(Create(rays_in), run_time=1)
        self.play(Create(rays_out), run_time=1)
        self.wait(3)
