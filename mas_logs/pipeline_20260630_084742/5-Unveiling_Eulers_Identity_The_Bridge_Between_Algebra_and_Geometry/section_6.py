from manim import *

# Assuming TeachingScene is a base class that provides grid layout functionality
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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize the layout with mandatory lecture lines
        lecture_lines = [
            'We can rearrange this into its most famous form.',
            'It unites the five fundamental constants of mathematics.',
            'This identity bridges algebra, geometry, and calculus together.'
        ]
        self.setup_layout("Summary: The Most Beautiful Equation", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Equation forms in white
        # Indices for "e^iπ=-1": e(0), ^(1), i(2), π(3), =(4), -(5), 1(6)
        eq_start = Text("e^iπ=-1", color=WHITE)
        # Indices for "e^iπ+1=0": e(0), ^(1), i(2), π(3), +(4), 1(5), =(6), 0(7)
        eq_final = Text("e^iπ+1=0", color=WHITE)

        # Positioning using corrected grid and scale (Issues 43, 44, 45)
        self.place_in_area(eq_start, "B2", "E5", scale_factor=1.2)
        self.place_in_area(eq_final, "B2", "E5", scale_factor=1.2)

        # Display start and transform to final
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Write(eq_start))
        self.wait(1)
        self.play(TransformMatchingShapes(eq_start, eq_final))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the five constants (e, i, pi, 1, 0) with distinct colors
        # Colors: e(YELLOW), i(BLUE), pi(GREEN), 1(RED), 0(PURPLE)
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(
            eq_final[0].animate.set_color(YELLOW),   # e
            eq_final[2].animate.set_color(BLUE),     # i
            eq_final[3].animate.set_color(GREEN),    # π
            eq_final[5].animate.set_color(RED),      # 1
            eq_final[7].animate.set_color(PURPLE),   # 0
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Fade out background (title/lecture) and zoom into the identity
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.play(
            FadeOut(self.title),
            FadeOut(self.lecture),
            eq_final.animate.scale(1.5).move_to(ORIGIN),
            run_time=3
        )
        self.wait(2)
