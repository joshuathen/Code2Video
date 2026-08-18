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
        self.setup_layout("Ownership: Elliptic Curve Cryptography", [
            "Elliptic curve cryptography secures user ownership.",
            "Private keys sign, public keys verify identity.",
            "Deriving private keys is computationally infeasible."
        ])
        
        # Define visual elements
        # Line 1: Main Title
        l1_text = Text("Public Key Cryptography", font_size=36, color=WHITE)
        self.place_at_grid(l1_text, 'B3', scale_factor=0.6)
        
        # Line 2: Private Key icon
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/document.svg
        l2_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/document.svg", color="#FFD700")
        l2_label = Text("Private Key", font_size=24, color="#FFD700")
        l2_group = VGroup(l2_icon, l2_label).arrange(DOWN)
        
        # Line 3: Public Sensor icon
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg
        l3_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg", color="#00BFFF")
        l3_label = Text("Public Sensor", font_size=24, color="#00BFFF")
        l3_group = VGroup(l3_icon, l3_label).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(l1_text))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(l2_group, 'C2', scale_factor=0.5)
        self.play(FadeIn(l2_group))
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(l3_group, 'C5', scale_factor=0.5)
        self.play(FadeIn(l3_group))
        self.play(self.lecture[2].animate.set_color("#00BFFF"))
        self.wait(1)
