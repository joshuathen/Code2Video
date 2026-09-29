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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Each block contains the previous block's hash.",
            "Altering old data breaks the entire chain.",
            "This structure ensures mathematical immutability."
        ]
        self.setup_layout("The Structure: Chaining Blocks", lecture_lines)
        
        # Load Assets
        blockchain_chain_link = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chain.svg")
        tamper_invalidation_chain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        # Address Issue 28: Use place_in_area for block chain
        self.place_in_area(blockchain_chain_link, 'A2', 'B5', scale_factor=0.9)
        self.play(DrawBorderThenFill(blockchain_chain_link))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        # Address Issue 29: Use place_in_area for tamper warning
        self.place_in_area(tamper_invalidation_chain, 'D2', 'D5', scale_factor=0.6)
        self.play(FadeIn(tamper_invalidation_chain), tamper_invalidation_chain.animate.set_color(RED))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Address Issue 27: Updated grid placement for checkmark
        integrity_check = Text("✓", color="#00FF00", font_size=72)
        self.place_at_grid(integrity_check, 'E5', scale_factor=0.8)
        self.play(Write(integrity_check))
        self.play(blockchain_chain_link.animate.set_color("#FFFF00"))
