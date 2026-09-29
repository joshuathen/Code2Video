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
        lecture_lines = [
            "Primes act as the atoms of arithmetic.",
            "Numbers grow, yet primes become rarer.",
            "Prime density follows 1/ln(x) approximately.",
            "Visualizing prime thinning in a grid.",
            "The grid shows this clearly."
        ]
        self.setup_layout("Prerequisite: The Nature of Primes", lecture_lines)
        
        atom_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF4500")
        atoms = VGroup(*[SVGMobject(atom_path).scale(0.2) for _ in range(16)])
        # Use grid to arrange atoms
        grid_pos = [f"{row}{col}" for row in "BCDE" for col in "2345"]
        for i, pos in enumerate(grid_pos):
            self.place_at_grid(atoms[i], pos, scale_factor=0.5)
        self.play(FadeIn(atoms))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00CED1")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        # Highlight some as primes
        prime_indices = [0, 2, 4, 7, 11, 14]
        for idx in prime_indices:
            atoms[idx].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#ADFF2F")
        # Fade non-primes
        non_primes = [i for i in range(16) if i not in prime_indices]
        self.play(FadeOut(VGroup(*[atoms[i] for i in non_primes])))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF69B4")
        # Final atom representing building blocks
        final_atom = SVGMobject(atom_path).scale(1.5)
        self.place_in_area(final_atom, "C3", "D4", scale_factor=1.0)
        self.play(FadeIn(final_atom))
        self.wait(2)
