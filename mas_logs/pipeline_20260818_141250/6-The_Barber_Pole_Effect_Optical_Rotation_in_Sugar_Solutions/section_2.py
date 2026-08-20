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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Chirality of Sugar Molecules", [
            "Chiral sugar molecules possess a specific handedness.",
            "They act like spiral staircases for light.",
            "Interaction twists the light's polarization plane."
        ])
        
        # Load assets
        sugar_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sugar.svg"
        mol1 = SVGMobject(sugar_asset_path, color=WHITE)
        mol2 = SVGMobject(sugar_asset_path, color=WHITE).flip(axis=RIGHT)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture_texts[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(mol1, 'D2', scale_factor=0.75)
        self.place_at_grid(mol2, 'D5', scale_factor=0.75)
        self.play(FadeIn(mol1), FadeIn(mol2))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture_texts[1].animate.set_color("#FFCC00"))
        # Rotate molecules to demonstrate non-superimposable nature
        self.play(Rotate(mol1, angle=PI/2), Rotate(mol2, angle=-PI/2))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture_texts[2].animate.set_color("#FF0000"))
        # Highlight chiral centers
        center1 = Dot(color="#FF0000", radius=0.15).move_to(mol1.get_center())
        center2 = Dot(color="#FF0000", radius=0.15).move_to(mol2.get_center())
        self.play(FadeIn(center1), FadeIn(center2))
        self.wait(2)
