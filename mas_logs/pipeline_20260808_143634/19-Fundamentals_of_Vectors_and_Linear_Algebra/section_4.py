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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Linear Combinations: Spanning the Space", [
            "Any vector can reach any coordinate point.",
            "We use basis vectors i and j.",
            "Linear combinations span the 2D space."
        ])
        
        # Assets (Using placeholders/simple shapes as assets exist but are generic/path-based)
        # Note: The provided paths exist but contain simple icons.
        
        # Define base vectors i, j
        i_vec = Arrow(start=ORIGIN, end=RIGHT, color=WHITE)
        j_vec = Arrow(start=ORIGIN, end=UP, color=WHITE)
        i_label = MathTex("\\vec{i}", color=WHITE)
        j_label = MathTex("\\vec{j}", color=WHITE)
        basis = VGroup(i_vec, j_vec, i_label, j_label)
        
        # === Animation for Lecture Line 1 ===
        # Addressing Critic Issue 29: Use place_in_area B2-D4
        self.place_in_area(basis, 'B2', 'D4', scale_factor=0.6)
        
        # Position labels relative to objects
        i_label.next_to(i_vec.get_end(), DOWN, buff=0.1)
        j_label.next_to(j_vec.get_end(), LEFT, buff=0.1)
        
        self.play(Create(basis), self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        i_scaled = Arrow(start=ORIGIN, end=2*RIGHT, color="#00FF00")
        j_scaled = Arrow(start=ORIGIN, end=3*UP, color="#00FF00")
        i_scaled_label = MathTex("2\\vec{i}", color="#00FF00")
        j_scaled_label = MathTex("3\\vec{j}", color="#00FF00")
        
        # Addressing Critic Issue 30: Place labels in C5 and B3
        self.place_at_grid(i_scaled_label, 'C5', scale_factor=0.7)
        self.place_at_grid(j_scaled_label, 'B3', scale_factor=0.7)
        
        self.play(
            ReplacementTransform(i_vec.copy(), i_scaled),
            ReplacementTransform(j_vec.copy(), j_scaled),
            FadeIn(i_scaled_label),
            FadeIn(j_scaled_label),
            self.lecture[1].animate.set_color("#00FF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        sum_vec = Arrow(start=ORIGIN, end=2*RIGHT + 3*UP, color="#FFFF00")
        sum_label = MathTex("2\\vec{i} + 3\\vec{j}", color="#FFFF00")
        
        # Addressing Critic Issue 31: Place formula in C6
        self.place_at_grid(sum_label, 'C6', scale_factor=0.8)
        
        self.play(
            Create(sum_vec),
            Write(sum_label),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        self.wait(2)
