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
            "Elastic collisions conserve momentum and kinetic energy.",
            "A small block hits a large block.",
            "The large block slides toward a wall.",
            "These laws govern simple mechanical motion.",
        ]
        self.setup_layout("Prerequisite: Elastic Collisions", lecture_lines)
        
        # Mobjects
        m_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE, fill_opacity=0.6).scale(0.3)
        M_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=GREEN, fill_opacity=0.6).scale(0.6)
        m_label = Text("m", color="#E0FFFF", font_size=24).next_to(m_block, UP, buff=0.1)
        M_label = Text("M", color="#E0FFFF", font_size=24).next_to(M_block, UP, buff=0.1)
        
        blocks = VGroup(m_block, m_label, M_block, M_label)
        self.place_in_area(blocks, 'B4', 'E6', scale_factor=0.7)
        self.add(blocks)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        momentum_txt = Text("Momentum: m*v1 + M*v2 = Const", font_size=18, color=WHITE)
        energy_txt = Text("Energy: 0.5*m*v1^2 + 0.5*M*v2^2 = Const", font_size=18, color=WHITE)
        equations = VGroup(momentum_txt, energy_txt).arrange(DOWN)
        self.place_at_grid(equations, 'D1', scale_factor=0.75)
        self.play(Write(equations))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(m_block.animate.shift(RIGHT * 1.5))
        flash = Flash(m_block.get_right(), color=WHITE, line_length=0.2)
        self.play(flash)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(M_block.animate.shift(RIGHT * 1.0))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.wait(2)
