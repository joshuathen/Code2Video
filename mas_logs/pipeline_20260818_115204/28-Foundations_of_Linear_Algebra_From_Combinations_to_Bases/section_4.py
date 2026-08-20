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
        self.setup_layout("Bases: The Minimal Blueprint", [
            "A basis spans the entire space.",
            "Basis vectors are also linearly independent.",
            "A basis is the minimal blueprint required."
        ])
        
        # Define vectors
        v1 = Vector([1, 1], color=BLUE)
        v2 = Vector([1, -1], color=GREEN)
        v3 = Vector([2, 0], color=YELLOW)
        basis_vectors = VGroup(v1, v2)
        
        # Blueprint icon
        blueprint = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blueprint.svg")
        basis_label = Text("Basis", color=WHITE)
        
        group_basis_elements = VGroup(blueprint, basis_label)

        # === Animation for Lecture Line 1 ===
        # A basis spans the entire space.
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(basis_vectors, 'B3', 'D4', scale_factor=0.9)
        self.play(FadeIn(v1), FadeIn(v2), FadeIn(v3))
        self.play(
            v1.animate.shift(RIGHT*0.2),
            v2.animate.shift(LEFT*0.2),
            run_time=1.5
        )

        # === Animation for Lecture Line 2 ===
        # Basis vectors are also linearly independent.
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(FadeOut(v3))
        self.play(Indicate(v1), Indicate(v2))

        # === Animation for Lecture Line 3 ===
        # A basis is the minimal blueprint required.
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        self.place_at_grid(basis_label, 'D5', scale_factor=0.8)
        self.place_in_area(group_basis_elements, 'B2', 'E5', scale_factor=0.8)
        
        self.play(FadeIn(blueprint), Write(basis_label))
        self.wait(2)
