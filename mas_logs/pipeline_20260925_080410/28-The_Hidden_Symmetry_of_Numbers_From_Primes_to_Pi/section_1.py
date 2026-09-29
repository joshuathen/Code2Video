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
        self.setup_layout("Introduction: The Quest for Order", [
            "Prime numbers are the atoms of mathematics.",
            "They appear random, but hide deep structure.",
            "We seek patterns within this chaos."
        ])
        
        # --- Create animation elements ---
        atom_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg"
        
        # Create scattered atoms
        atoms = VGroup(*[SVGMobject(atom_path, color=WHITE).scale(0.2) for _ in range(25)])
        for atom in atoms:
            atom.move_to(np.array([np.random.uniform(2, 6), np.random.uniform(-2.5, 2.5), 0]))
            
        # Target positions in 5x5 grid
        target_atoms = VGroup(*[SVGMobject(atom_path, color=WHITE).scale(0.2) for _ in range(25)])
        self.place_in_area(target_atoms, 'A2', 'F6', scale_factor=0.9)
        
        # For precise layout of target atoms
        grid_positions = [self.grid[f"{row}{col}"] for row in ["B", "C", "D", "E", "F"] for col in ["2", "3", "4", "5", "6"]]
        for i, atom in enumerate(target_atoms):
            atom.move_to(grid_positions[i])

        pattern_label = Text("Mathematical Order", font_size=24, color=WHITE)
        self.place_at_grid(pattern_label, "F3", scale_factor=0.7)
        pattern_label.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"), FadeIn(atoms))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF5733"),
            Transform(atoms, target_atoms)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFD700"),
            atoms.animate.set_color("#FFD700"),
            FadeIn(pattern_label)
        )
        self.wait(2)
