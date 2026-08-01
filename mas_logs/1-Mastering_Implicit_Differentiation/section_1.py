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
        # Initial Setup using TeachingScene layout
        title_text = "The 'Hidden' Variable: Explicit vs. Implicit"
        lecture_lines = [
            "Explicit functions like y equals f of x are clear.",
            "Implicit relations like circles tangle x and y together.",
            "We still need the slope, dy dx, everywhere."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Color palette
        COLOR_EXPLICIT = "#00AAFF"
        COLOR_IMPLICIT = "#FFA500"
        COLOR_Y = "#FF0000"
        COLOR_TANGENT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_EXPLICIT)
        
        # Display 'y = x^2'
        # Issue 19: Scale factor adjusted to 0.8 to avoid crowding
        eq_explicit = Text("y = x^2", color=COLOR_EXPLICIT)
        self.place_at_grid(eq_explicit, "A2", scale_factor=0.8)
        
        # Draw a simple blue curve (representing the function y = x^2)
        blue_curve = FunctionGraph(
            lambda x: 0.5 * x**2 - 0.4, 
            x_range=[-1.5, 1.5], 
            color=COLOR_EXPLICIT
        )
        self.place_in_area(blue_curve, "B1", "D3", scale_factor=0.8)
        
        self.play(Write(eq_explicit))
        self.play(Create(blue_curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_IMPLICIT)
        
        # Display 'x^2 + y^2 = 25'
        # Issue 20: Scale factor adjusted to 0.7 to handle length and edge proximity
        eq_implicit = VGroup(Text("x^2 + "), Text("y"), Text("^2 = 25")).arrange(RIGHT, buff=0).set_color(COLOR_IMPLICIT)
        self.place_at_grid(eq_implicit, "A5", scale_factor=0.7)
        
        # Draw orange circle
        # Issue 21: Scale factor adjusted to 0.7 for better visual breathing room
        circle = Circle(radius=1.1, color=COLOR_IMPLICIT)
        self.place_in_area(circle, "B4", "D6", scale_factor=0.7)
        
        self.play(Write(eq_implicit))
        self.play(Create(circle))
        
        # Highlight 'y' in the circle equation with a pulsing red
        y_mob = eq_implicit[1]
        self.play(Indicate(y_mob, color=COLOR_Y, scale_factor=1.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_TANGENT)
        
        # Add a white tangent line on the orange circle at a point representing (3, 4)
        angle_at_34 = np.arctan2(4, 3)
        point_at_34 = circle.point_at_angle(angle_at_34)
        
        # Tangent direction is perpendicular to the radial direction
        tan_dir = np.array([-np.sin(angle_at_34), np.cos(angle_at_34), 0])
        tangent_line = Line(
            point_at_34 - tan_dir * 1.2,
            point_at_34 + tan_dir * 1.2,
            color=COLOR_TANGENT,
            stroke_width=2
        )
        
        # Small white triangle (drone)
        drone = Triangle(color=COLOR_TANGENT, fill_opacity=1).scale(0.12)
        drone.move_to(point_at_34)
        # Rotate drone so it points along the tangent
        drone.rotate(angle_at_34 + PI/2)

        self.play(Create(tangent_line))
        self.play(FadeIn(drone))
        self.wait(0.5)
        
        # Drone moves along the circle path
        # Rotate around the circle's center maintains the tangent orientation relative to path
        self.play(
            Rotate(drone, angle=2*PI, about_point=circle.get_center()),
            run_time=5,
            rate_func=linear
        )
        self.wait(2)
