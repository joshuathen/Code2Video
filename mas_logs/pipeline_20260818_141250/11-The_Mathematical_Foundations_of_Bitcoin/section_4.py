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
        self.setup_layout("Digital Signatures: Proving Ownership", [
            "Private keys seal digital transactions.",
            "Public keys verify the signatures.",
            "Originality remains protected throughout."
        ])
        
        # Elements
        data_rect = Rectangle(width=2, height=1.5, color=WHITE)
        data_lock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg").scale(0.5)
        data_group = VGroup(data_rect, data_lock)
        data_label = Text("Transaction Data", font_size=20).next_to(data_group, UP)
        data_full = VGroup(data_group, data_label)
        
        priv_key = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg").scale(0.8)
        priv_label = Text("Private Key", font_size=18).next_to(priv_key, DOWN)
        priv_group = VGroup(priv_key, priv_label)
        
        signature = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/seal.svg").scale(0.8).set_color("#FF00FF")
        sig_label = Text("Signature", font_size=18).next_to(signature, DOWN)
        sig_group = VGroup(signature, sig_label)

        # === Animation for Lecture Line 1 ===
        self.place_in_area(data_full, "B2", "C3", scale_factor=0.6)
        self.play(FadeIn(data_full))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(priv_group, "D2", scale_factor=0.7)
        self.play(FadeIn(priv_group))
        self.play(priv_group.animate.move_to(data_rect.get_center()))
        self.play(ReplacementTransform(priv_group, sig_group))
        self.place_in_area(sig_group, "D4", "E5", scale_factor=0.6)
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        self.play(Flash(sig_group, color="#FF00FF"))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
