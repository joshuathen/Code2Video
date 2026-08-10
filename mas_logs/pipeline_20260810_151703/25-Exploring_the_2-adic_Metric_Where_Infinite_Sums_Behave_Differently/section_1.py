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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Standard distance measures physical gaps on a line.",
            "A frog jumps, halving its distance to target.",
            "The sum converges as jumps get smaller."
        ]
        self.setup_layout("Prerequisite: The Usual Notion of Convergence", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#44AAFF")
        number_line = NumberLine(x_range=[0, 4, 1], length=5, include_numbers=True)
        # Using place_in_area as requested per Issue 19
        self.place_in_area(number_line, 'C3', 'C6', scale_factor=1.0)
        self.play(Create(number_line))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFAA44")
        target = Dot(number_line.n2p(4), color=RED)
        # Using the asset reference as required in Issue 13
        frog = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/frog.svg", color=GREEN)
        self.place_at_grid(frog, 'B3', scale_factor=0.3)
        frog.move_to(number_line.n2p(0))
        
        self.add(target, frog)
        
        curr_pos = 0
        for _ in range(4):
            dist = 4 - curr_pos
            curr_pos += dist / 2
            self.play(frog.animate.move_to(number_line.n2p(curr_pos)), run_time=0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#AAFF44")
        limit_text = Text("Sum -> 4", font_size=20, color=WHITE)
        # Using place_at_grid for the limit text as requested per Issue 18/29
        self.place_at_grid(limit_text, 'D4', scale_factor=1.0)
        self.play(Write(limit_text))
        self.wait(1)
