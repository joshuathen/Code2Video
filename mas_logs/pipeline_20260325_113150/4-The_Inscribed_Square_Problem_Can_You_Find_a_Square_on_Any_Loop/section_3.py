from manim import *

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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "The Strategy: From Square to Rectangle"
        lines = [
            "Proving a square exists is a hard challenge.",
            "Instead, let's look for an inscribed rectangle first.",
            "Rectangles have two equal diagonals sharing a midpoint."
        ]
        self.setup_layout(title, lines)

        # === Define Objects ===
        # Create a blob that passes through specific points to ensure the rectangle can be "inscribed"
        # Rectangle vertices (for a 2.4 x 1.6 rectangle)
        v1 = np.array([1.2, 0.8, 0])
        v2 = np.array([1.2, -0.8, 0])
        v3 = np.array([-1.2, -0.8, 0])
        v4 = np.array([-1.2, 0.8, 0])
        
        # Blob geometry
        blob = Polygon(
            v1, [1.6, 0.1, 0], v2, [0.2, -1.3, 0], 
            v3, [-1.5, -0.2, 0], v4, [-0.1, 1.4, 0],
            color=GREEN, stroke_width=4
        ).round_corners(radius=0.5)

        # Rectangle for Line 2
        rect = Rectangle(width=2.4, height=1.6, color="#00FFFF", stroke_width=4)
        
        # Diagonals for Line 3
        diag1 = Line(v3, v1, color="#FF8C00")
        diag2 = Line(v2, v4, color="#FF8C00")
        
        # Midpoint for Line 3
        midpoint = Dot(ORIGIN, color="#FFFFFF", radius=0.08)
        
        # Non-square points for Line 1
        p1 = Dot(v1 + [0.3, 0.2, 0], color=GREEN)
        p2 = Dot([1.6, 0.1, 0], color=GREEN)
        p3 = Dot(v3, color=GREEN)
        p4 = Dot([-0.1, 1.4, 0], color=GREEN)
        non_square_pts = VGroup(p1, p2, p3, p4)

        # Group all elements for consistent positioning via grid
        animation_elements = VGroup(blob, rect, diag1, diag2, midpoint, non_square_pts)
        self.place_in_area(animation_elements, "A2", "F5")

        # === Animation for Lecture Line 1 ===
        # "Proving a square exists is a hard challenge."
        self.play(self.lecture[0].animate.set_color(GREEN))
        self.play(Create(blob), run_time=1.5)
        self.play(FadeIn(non_square_pts))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Instead, let's look for an inscribed rectangle first."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        self.play(FadeOut(non_square_pts))
        # Show rectangle appearing and snapping to blob
        self.play(Create(rect), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Rectangles have two equal diagonals sharing a midpoint."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF8C00")
        )
        self.play(Create(diag1), Create(diag2))
        self.play(FadeIn(midpoint))
        self.wait(2)
