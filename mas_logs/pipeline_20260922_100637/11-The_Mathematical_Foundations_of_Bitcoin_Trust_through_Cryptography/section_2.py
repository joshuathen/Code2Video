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
        self.setup_layout("Digital Signatures: Asymmetric Cryptography", [
            "Public-Key Infrastructure uses a key pair system.",
            "Private key signs, public key verifies.",
            "Math allows verification without revealing the secret."
        ])
        
        # Define assets
        priv_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg").set_color(WHITE)
        pub_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg").set_color(WHITE)
        doc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/document.svg").set_color(WHITE)
        
        priv_label = Text("Private Key", color=WHITE, font_size=20)
        pub_label = Text("Public Key", color=WHITE, font_size=20)
        sig_label = Text("Signature", font_size=20, color=PURPLE)
        check = Tex(r"$\checkmark$", color=GREEN, font_size=60)
        
        # Groupings
        priv_group = VGroup(priv_icon, priv_label).arrange(DOWN, buff=0.1)
        pub_group = VGroup(pub_icon, pub_label).arrange(DOWN, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(priv_group, "B2", scale_factor=0.7)
        self.place_at_grid(pub_group, "B5", scale_factor=0.7)
        self.play(FadeIn(priv_group), FadeIn(pub_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_in_area(doc, "D1", "D2", scale_factor=0.6)
        self.play(FadeIn(doc))
        self.play(doc.animate.move_to(self.grid["C2"]))
        self.play(Write(sig_label))
        self.place_at_grid(sig_label, "C3", scale_factor=0.8)
        self.play(sig_label.animate.move_to(self.grid["C5"]))
        self.play(FadeIn(check))
        self.place_at_grid(check, "C4", scale_factor=0.6)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.play(check.animate.set_color(GREEN))
        self.wait(2)
