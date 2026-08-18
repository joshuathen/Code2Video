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
        self.setup_layout("Proof of Work: The Mathematical Puzzle", [
            "Mining is a difficult computational lottery.", 
            "Miners guess a nonce for valid hashes.", 
            "Successful guessing adds a new block."
        ])
        
        # Objects
        puzzle = MathTex(r"\\text{Hash}( \\text{Data} + \\text{Nonce} ) < \\text{Target}", color=BLUE)
        self.place_at_grid(puzzle, "B4", scale_factor=0.6)
        
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg]
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(computer, "D5", scale_factor=0.5)
        
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pickaxe.svg]
        pickaxe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pickaxe.svg")
        
        new_block = Rectangle(width=1.5, height=1.0, color=WHITE)
        new_block_label = Text("New Block", font_size=18)
        new_block_group = VGroup(new_block, new_block_label)
        self.place_at_grid(new_block_group, "E2", scale_factor=0.5)
        new_block_group.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(puzzle), Write(self.lecture[0]))
        self.play(self.lecture[0].animate.set_color("#F1C40F"))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(computer), Write(self.lecture[1]))
        self.play(self.lecture[1].animate.set_color("#F1C40F"))
        # Animate computation process with #F1C40F
        self.play(computer.animate.set_color("#F1C40F"), run_time=2)
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(Write(self.lecture[2]))
        self.play(self.lecture[2].animate.set_color("#E74C3C"))
        # Show successful solution finding block creation using pickaxe.svg
        self.play(FadeIn(pickaxe.move_to(new_block_group.get_center()).shift(UP*0.5)))
        self.play(new_block_group.animate.set_opacity(1), run_time=1)
        self.play(new_block.animate.set_color("#E74C3C"))
        self.wait(2)
