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
        self.setup_layout("The Mathematical Security of the Chain", [
            "Blocks are linked using previous block hashes.",
            "This creates an immutable history of records.",
            "Tampering with one block invalidates the chain."
        ])
        
        # Load assets
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        chain_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chain.svg")

        # Create block mobjects using asset
        block1 = block_icon.copy()
        label1 = Text("Block 1", font_size=24)
        group1 = VGroup(block1, label1).arrange(DOWN)
        self.place_at_grid(group1, "B2", scale_factor=0.6)

        block2 = block_icon.copy()
        label2 = Text("Block 2", font_size=24)
        group2 = VGroup(block2, label2).arrange(DOWN)
        self.place_at_grid(group2, "D5", scale_factor=0.6)

        # Connection line
        line = Line(group1.get_right(), group2.get_left(), color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.play(Create(group1), Create(group2), Create(line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        block3 = block_icon.copy()
        label3 = Text("Block 3", font_size=24)
        group3 = VGroup(block3, label3).arrange(DOWN)
        self.place_at_grid(group3, "F3", scale_factor=0.6)
        
        line2 = Line(group2.get_bottom(), group3.get_top(), color=WHITE)
        self.play(Create(group3), Create(line2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        # Visualizing hash dependency: changing color to indicate security
        chain_symbol = chain_icon.copy().scale(0.5).next_to(self.lecture[2], RIGHT)
        self.play(
            group1.animate.set_color("#00FF00"),
            group2.animate.set_color("#00FF00"),
            group3.animate.set_color("#00FF00"),
            line.animate.set_color("#00FF00"),
            line2.animate.set_color("#00FF00"),
            FadeIn(chain_symbol)
        )
        self.wait(2)
