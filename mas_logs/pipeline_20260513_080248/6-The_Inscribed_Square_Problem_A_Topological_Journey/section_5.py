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
        # Setup the layout with the specific title and lines for section 5
        self.setup_layout(
            "The Intersection: Proving the Rectangle exists", 
            [
                "A rectangle's diagonals have equal length and midpoint.", 
                "The surface's topology forces a self-intersection point.", 
                "This intersection guarantees an inscribed rectangle exists."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Draw a rectangle and its diagonals
        rect_color = "#888888"
        diag_color = "#3399FF"
        
        rectangle = Rectangle(width=3, height=2, color=rect_color)
        diag1 = Line(rectangle.get_corner(DL), rectangle.get_corner(UR), color=diag_color)
        diag2 = Line(rectangle.get_corner(UL), rectangle.get_corner(DR), color=diag_color)
        midpoint_dot = Dot(rectangle.get_center(), color=WHITE)
        
        diag_label1 = Text("d", font_size=16, color=diag_color)
        diag_label2 = Text("d", font_size=16, color=diag_color)
        diag_label1.next_to(diag1.get_center(), UR, buff=0.1)
        diag_label2.next_to(diag2.get_center(), UL, buff=0.1)
        
        line1_group = VGroup(rectangle, diag1, diag2, midpoint_dot, diag_label1, diag_label2)
        # Resolved Issue 43: Vertical alignment fix
        self.place_in_area(line1_group, "B2", "E5", scale_factor=0.8)
        
        self.play(Create(rectangle))
        self.play(Create(diag1), Create(diag2))
        self.play(FadeIn(midpoint_dot), FadeIn(diag_label1), FadeIn(diag_label2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Represent the Möbius strip surface with a twist and an intersection
        # We'll use a stylized loop to represent the surface in 2D
        mobius_color = "#00FFFF"
        intersection_color = "#FF3333"
        
        # Create a "figure 8" style twist to simulate the self-intersection visually
        path = ParametricFunction(
            lambda t: np.array([2 * np.cos(t), np.sin(2 * t), 0]),
            t_range=[0, TAU],
            color=mobius_color
        )
        
        intersection_dot = Dot(path.get_center(), color=intersection_color, radius=0.1)
        intersection_text = Text("(M, d)", font_size=18, color=intersection_color)
        intersection_text.next_to(intersection_dot, DOWN, buff=0.2)
        
        surface_group = VGroup(path, intersection_dot, intersection_text)
        # Resolved Issue 44: Balanced area positioning
        self.place_in_area(surface_group, "B2", "E5", scale_factor=0.9)
        
        self.play(FadeOut(line1_group))
        self.play(Create(path))
        self.play(FadeIn(intersection_dot), Write(intersection_text))
        self.play(Indicate(intersection_dot, color=intersection_color))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Draw a 2D loop and two equal chords forming a rectangle
        loop_color = WHITE
        chord_color = "#3399FF"
        rect_final_color = "#FFD700" # GOLD
        
        # A simple oval to serve as the curve
        loop = Ellipse(width=4, height=2.5, color=loop_color)
        
        # Two chords of same length sharing same midpoint
        # For an ellipse x^2/a^2 + y^2/b^2 = 1, diagonals of an inscribed rectangle work
        # Let's just draw them manually for clarity
        p1 = loop.point_at_angle(30 * DEGREES)
        p2 = loop.point_at_angle(210 * DEGREES)
        p3 = loop.point_at_angle(150 * DEGREES)
        p4 = loop.point_at_angle(330 * DEGREES)
        
        chord1 = Line(p1, p2, color=chord_color)
        chord2 = Line(p3, p4, color=chord_color)
        
        final_rectangle = Polygon(p1, p3, p2, p4, color=rect_final_color, stroke_width=4)
        
        loop_group = VGroup(loop, chord1, chord2, final_rectangle)
        # Resolved Issue 45: Avoid cramped vertical span
        self.place_in_area(loop_group, "B2", "E5", scale_factor=0.8)
        
        self.play(FadeOut(surface_group))
        self.play(Create(loop))
        self.play(Create(chord1), Create(chord2))
        self.wait(0.5)
        self.play(Create(final_rectangle))
        self.play(Indicate(final_rectangle, color=rect_final_color))
        self.wait(2)
        
        # Cleanup colors
        self.lecture[2].set_color(WHITE)
