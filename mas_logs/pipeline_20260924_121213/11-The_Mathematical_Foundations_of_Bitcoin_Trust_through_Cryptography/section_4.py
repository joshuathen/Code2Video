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
        self.setup_layout("Proof of Work: Mathematical Consensus", [
            "Miners compete to find a specific hash.",
            "The hash must start with many zeros.",
            "This process is like a lottery search."
        ])
        
        # Assets
        miner = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/miner.svg")
        
        # === Animation for Lecture Line 1 ===
        # Representing mathematical block
        block = Square(side_length=1.5, color="#FFFFFF")
        block_text = MathTex("H(Data + Nonce)", color="#FFFFFF").scale(0.8)
        puzzle = VGroup(block, block_text)
        self.place_in_area(puzzle, 'A2', 'B5', scale_factor=0.9)
        self.play(FadeIn(puzzle))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Display zeros requirement
        zeros_req = Text("0000... (Prefix Req)", color="#FFFF00").scale(0.6)
        self.place_at_grid(zeros_req, 'C3', scale_factor=0.8)
        self.play(Write(zeros_req))
        
        # Miner icon for computational effort
        self.place_at_grid(miner, 'D3', scale_factor=0.5)
        self.play(FadeIn(miner))
        self.play(miner.animate.shift(RIGHT * 0.5).shift(LEFT * 0.5))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Miner success
        success_text = Text("Nonce Found!", color="#00FF00").scale(0.8)
        self.place_at_grid(success_text, 'D3', scale_factor=0.9)
        self.play(ReplacementTransform(puzzle.copy(), success_text))
        self.play(Indicate(success_text))
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
