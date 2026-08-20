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
            "Two blocks slide toward each other.",
            "Small block hits a massive block.",
            "Collisions continue until they separate.",
            "Total count surprises us all.",
            "How many collisions will occur?"
        ]
        self.setup_layout("The π-Collision Mystery", lecture_lines)
        
        # Assets
        block_m = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE)
        block_M = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=RED)
        wall = Line(start=UP*1.5, end=DOWN*1.5, color=WHITE, stroke_width=8)
        ground = Line(start=LEFT*3, end=RIGHT*3, color=WHITE, stroke_width=4)
        
        # Layout based on feedback (using recommended positions for better visibility)
        self.place_at_grid(block_m, 'C2', scale_factor=0.5)
        self.place_at_grid(block_M, 'C5', scale_factor=1.5)
        self.place_at_grid(wall, 'E6', scale_factor=1.0)
        self.place_at_grid(ground, 'D4', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(block_m), FadeIn(block_M), FadeIn(wall), FadeIn(ground))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.play(block_m.animate.move_to(self.grid['C4']))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.play(block_m.animate.move_to(self.grid['C3']), block_M.animate.move_to(self.grid['C5']))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        counter = Text("1", color=GREEN, font_size=36)
        self.place_at_grid(counter, 'A5')
        self.play(FadeIn(counter))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        self.play(block_m.animate.set_color(GREEN), block_M.animate.set_color(GREEN))
        self.wait(2)
