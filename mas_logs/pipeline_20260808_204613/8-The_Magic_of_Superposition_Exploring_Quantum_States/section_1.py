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
        self.setup_layout("Introduction: The Binary Limit", [
            "Classical bits are strictly zero or one.",
            "Like a simple light switch.",
            "Quantum states exist in both at once."
        ])
        
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg]
        bit0 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg", color=WHITE).scale(0.5)
        bit1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg", color=WHITE).scale(0.5)
        
        # Repositioning bits as requested (Line 55/56)
        self.place_at_grid(bit0, 'C3', scale_factor=1.0)
        self.place_at_grid(bit1, 'C4', scale_factor=1.0)
        
        self.add(bit0, bit1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(bit0.animate.set_color("#FF0000"), bit1.animate.set_color("#0000FF"))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        # Visual asset placement and grouping (Line 68)
        switch_label = Text("ON/OFF", font_size=24, color=GRAY)
        self.place_at_grid(switch_label, 'D3', scale_factor=0.8)
        self.play(FadeIn(switch_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        # Label placement (Line 74)
        q_mark = Tex("?", color=WHITE).scale(2)
        self.place_at_grid(q_mark, 'C3', scale_factor=0.9)
        self.play(FadeIn(q_mark))
        self.wait(2)
