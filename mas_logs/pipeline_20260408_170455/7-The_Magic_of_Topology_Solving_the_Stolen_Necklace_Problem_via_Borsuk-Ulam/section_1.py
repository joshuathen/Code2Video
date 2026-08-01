from manim import *
import random

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
            'Two thieves steal a necklace with different bead types.',
            'They want to share every type exactly in half.',
            'What is the minimum number of cuts needed?'
        ]
        self.setup_layout("The Hook: The Fair Thief Dilemma", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.lecture[0].set_color(YELLOW)

        # Necklace base
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/necklace.svg]
        necklace_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/necklace.svg")
        self.place_in_area(necklace_svg, 'C1', 'C6', scale_factor=1.2)
        
        line_start = self.grid['C1']
        line_end = self.grid['C6']
        necklace_line = Line(line_start, line_end, color=GREY_B, stroke_width=2)
        
        # Create beads (10 red, 8 blue)
        bead_colors = ["#FF0000"] * 10 + ["#0000FF"] * 8
        random.seed(42) # Deterministic randomness
        random.shuffle(bead_colors)
        
        beads = VGroup()
        for i, color in enumerate(bead_colors):
            pos = line_start + (i / (len(bead_colors) - 1)) * (line_end - line_start)
            bead = Circle(radius=0.1, fill_color=color, fill_opacity=1, stroke_width=1, stroke_color=WHITE)
            bead.move_to(pos)
            beads.add(bead)

        self.play(FadeIn(necklace_svg), Create(necklace_line))
        self.play(LaggedStart(*[FadeIn(b) for b in beads], lag_ratio=0.05))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/thief.svg]
        thief_a_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/thief.svg")
        thief_a_label = Text("Thief A", font_size=16)
        thief_a = VGroup(thief_a_svg, thief_a_label).arrange(UP, buff=0.1)

        thief_b_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/thief.svg")
        thief_b_label = Text("Thief B", font_size=16)
        thief_b = VGroup(thief_b_svg, thief_b_label).arrange(UP, buff=0.1)

        # Fix issues 28 and 29: Position at D1 and D6
        self.place_at_grid(thief_a, 'D1', scale_factor=0.8)
        self.place_at_grid(thief_b, 'D6', scale_factor=0.8)

        self.play(FadeIn(thief_a), FadeIn(thief_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        question_text = Text("Fair split with minimal cuts?", font_size=24, color="#FFFFFF")
        question_mark = Text("?", font_size=48, color=YELLOW)
        
        # Combine and place
        q_group = VGroup(question_mark, question_text).arrange(DOWN, buff=0.2)
        # Fix issue 27: Place in A1-B6 with scale 0.8
        self.place_in_area(q_group, 'A1', 'B6', scale_factor=0.8)

        self.play(Write(q_group))
        self.wait(2)
