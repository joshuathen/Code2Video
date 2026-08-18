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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Synthesis", [
            "Hashing, Proof-of-Work, and ECC form Bitcoin.",
            "These primitives enable decentralized trust systems.",
            "The network acts as a global ledger."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display the beehive.svg representing the network
        beehive = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beehive.svg", color="#FFD700")
        self.place_at_grid(beehive, "C4", scale_factor=0.5)
        self.play(FadeIn(beehive))
        self.lecture[0].set_color("#FFD700")
        
        # === Animation for Lecture Line 2 ===
        # Label 'Hashing', 'Proof-of-Work', and 'Cryptography' around the beehive
        hashing_label = Text("Hashing", font_size=20, color=WHITE)
        pow_label = Text("Proof-of-Work", font_size=20, color=WHITE)
        ecc_label = Text("Cryptography", font_size=20, color=WHITE)
        
        self.place_at_grid(hashing_label, "B3")
        self.place_at_grid(pow_label, "D3")
        self.place_at_grid(ecc_label, "C5")
        
        self.play(FadeIn(hashing_label), FadeIn(pow_label), FadeIn(ecc_label))
        self.lecture[1].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 3 ===
        # Animate all components merging into a single 'Trust' icon
        trust_icon = Circle(radius=0.5, color="#00FF00", fill_opacity=0.5)
        trust_text = Text("Trust", font_size=24, color="#00FF00")
        trust_group = VGroup(trust_icon, trust_text)
        
        self.place_at_grid(trust_group, "C4")
        
        self.play(
            FadeOut(beehive),
            FadeOut(hashing_label),
            FadeOut(pow_label),
            FadeOut(ecc_label),
            FadeIn(trust_group)
        )
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
