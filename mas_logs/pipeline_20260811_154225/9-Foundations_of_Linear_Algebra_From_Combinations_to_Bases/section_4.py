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
        self.setup_layout("Synthesis and Summary", ["Span defines our reach.", "Independence ensures efficiency.", "Basis is the space's DNA."])
        
        # Prepare visual elements
        # B020: Use 0.7-0.8 scale factor for labels
        # B011: Tether labels to objects
        span_label = Text("Span", font_size=24).scale(0.7)
        span_shape = Circle(radius=0.4, color=BLUE)
        span_group = VGroup(span_shape, span_label).arrange(DOWN)
        
        ind_label = Text("Independence", font_size=24).scale(0.7)
        ind_shape = Square(side_length=0.6, color=GREEN)
        ind_group = VGroup(ind_shape, ind_label).arrange(DOWN)
        
        basis_label = Text("Basis", font_size=24).scale(0.7)
        basis_shape = Triangle(color=YELLOW).scale(0.4)
        basis_group = VGroup(basis_shape, basis_label).arrange(DOWN)
        
        # === Animation for Lecture Line 1 ===
        # B004: Restrict anchor to columns 4-6 if possible, but adjust for layout
        self.place_at_grid(span_group, 'B4', scale_factor=1.0)
        self.play(FadeIn(span_group), self.lecture[0].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(ind_group, 'C4', scale_factor=1.0)
        self.play(FadeIn(ind_group), self.lecture[1].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(basis_group, 'E4', scale_factor=1.0)
        self.play(FadeIn(basis_group), self.lecture[2].animate.set_color(YELLOW))
        self.wait(1)
        
        # Pulse animation
        pulse = VGroup(span_group, ind_group, basis_group)
        self.play(Indicate(pulse))
        
        # Concluding text
        final_text = Text("Linear Algebra Foundation", font_size=32, color=WHITE).move_to(self.grid['D3'])
        self.play(FadeOut(span_group), FadeOut(ind_group), FadeOut(basis_group))
        self.play(Write(final_text))
        self.wait(2)
