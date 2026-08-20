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
            "Two blocks slide on a frictionless surface.",
            "They collide with each other and the wall.",
            "Counting collisions mysteriously reveals digits of pi.",
            "A mouse block hits an elephant block.",
            "Let's count these mechanical impacts!"
        ]
        self.setup_layout("Introduction: The Impossible Counting Puzzle", lecture_lines)

        # Assets
        block_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        # Using SVG asset
        mouse = SVGMobject(block_asset, color=BLUE).set_fill(BLUE, opacity=0.5)
        elephant = SVGMobject(block_asset, color=GREEN).set_fill(GREEN, opacity=0.5)
        blocks = VGroup(mouse, elephant).arrange(RIGHT)
        self.place_at_grid(blocks, 'C1', scale_factor=0.9)
        self.play(FadeIn(blocks))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#ADD8E6"))
        wall = Line(start=UP*2, end=DOWN*2, color=GRAY, stroke_width=10)
        self.place_at_grid(wall, 'C6', scale_factor=1.0)
        self.play(Create(wall))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        pi_symbol = MathTex(r"\\pi", color=YELLOW, font_size=72)
        self.place_at_grid(pi_symbol, 'B5', scale_factor=0.7)
        self.play(Write(pi_symbol))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        mouse_label = Text("Mouse", font_size=18).scale(0.7)
        mouse_label.next_to(mouse, UP)
        elephant_label = Text("Elephant", font_size=18).scale(0.7)
        elephant_label.next_to(elephant, UP)
        self.add(mouse_label, elephant_label)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF4500"))
        question_mark = Text("?", font_size=96, color="#FF4500")
        self.place_at_grid(question_mark, 'D5', scale_factor=0.7)
        self.play(FadeIn(question_mark, scale=2))
        self.play(Indicate(blocks))
